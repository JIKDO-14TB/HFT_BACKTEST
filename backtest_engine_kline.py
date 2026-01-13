from __future__ import annotations
from typing import Dict
import itertools
import pandas as pd

from hft_backtest_engine.data_loader import DataLoader
from hft_backtest_engine.execution import ExecutionEngine, Order as ExecOrder
from hft_backtest_engine.strategy_closez import CloseZStrategy, StrategyState, Position, Order
from hft_backtest_engine.feature_store_kline import FeatureStoreKline

COMPUTE_DELAY_MS = 10


class BacktestEngineKline:
    def __init__(
        self,
        loader: DataLoader,
        strategy: CloseZStrategy,
        execution: ExecutionEngine,
        feature_store: FeatureStoreKline,
        initial_capital: float = 100.0,
    ):
        self.loader = loader
        self.strategy = strategy
        self.execution = execution
        self.feature_store = feature_store

        self.state = StrategyState(
            cash=initial_capital,
            positions={},
            current_ts=None,
        )

        self.order_id_gen = itertools.count(1)
        self.active_orders: Dict[int, ExecOrder] = {}

        self.fills = []
        self.closed_trades = []

    def run_day(self, symbol: str, ymd: str):
        klines = self.loader.load_klines_1m_day(symbol, ymd)
        if klines is None or klines.empty:
            print(f"[WARN] no klines: {symbol} {ymd}")
            return

        klines = klines.sort_values("open_ts").reset_index(drop=True)

        for bar in klines.itertuples():
            ts = bar.open_ts
            self.state.current_ts = ts

            # (A) feature update: 현재 bar를 버퍼에 push
            k = klines.loc[[bar.Index], ["open_ts", "close"]]
            self.feature_store.update_kline(k)

            # (B) 기존 주문 체결 시도 (bar close를 “거래가격”으로 사용)
            pseudo_trade = type("T", (), {"ts": ts, "price": float(bar.close)})
            self._process_active_orders(pseudo_trade)

            # (C) 전략 호출
            orders = self.strategy.on_bar(bar, self.state)

            for o in orders:
                self._submit_order(o, ts)

            # (D) 같은 bar에서 바로 체결 시도(단순화)
            self._process_active_orders(pseudo_trade)

    def _submit_order(self, proto: Order, ts: pd.Timestamp):
        oid = next(self.order_id_gen)
        created_ts = ts + pd.Timedelta(milliseconds=COMPUTE_DELAY_MS)

        expire_ts = None
        if proto.order_type == "limit":
            expire_ts = created_ts + pd.Timedelta(seconds=5)

        order = ExecOrder(
            order_id=oid,
            symbol=proto.symbol,
            side=proto.side,
            size=proto.size,
            order_type=proto.order_type,
            price=proto.price,
            created_ts=created_ts,
            expire_ts=expire_ts,
        )
        self.active_orders[oid] = order

    def _process_active_orders(self, trade):
        to_remove = []

        for oid, order in list(self.active_orders.items()):
            if trade.ts < self.execution.order_active_ts(order):
                continue

            # 지정가 만료 → 취소 후 시장가로
            if order.expire_ts and trade.ts >= order.expire_ts:
                to_remove.append(oid)

                market_oid = next(self.order_id_gen)
                market = ExecOrder(
                    order_id=market_oid,
                    symbol=order.symbol,
                    side=order.side,
                    size=order.size,
                    order_type="market",
                    price=None,
                    created_ts=trade.ts,
                )
                self.active_orders[market_oid] = market
                continue

            fill = self.execution.try_fill(order, trade)
            if fill.filled:
                self._apply_fill(order, float(fill.fill_price), trade.ts)
                to_remove.append(oid)

        for oid in to_remove:
            self.active_orders.pop(oid, None)

    def _apply_fill(self, order: ExecOrder, price: float, ts: pd.Timestamp):
        symbol = order.symbol

        if symbol not in self.state.positions:
            self.state.positions[symbol] = Position(
                symbol=symbol,
                size=order.size,
                side=order.side,
                entry_price=price,
                entry_ts=ts,
            )
            self.state.cash -= order.size

            self.fills.append({
                "ts": ts,
                "symbol": symbol,
                "fill_type": "ENTER",
                "side": order.side,
                "price": price,
                "size": order.size,
                "order_type": order.order_type,
                "order_id": order.order_id,
            })

        else:
            pos = self.state.positions.pop(symbol)
            pnl = pos.side * (price - pos.entry_price) / pos.entry_price * pos.size
            self.state.cash += pos.size + pnl

            self.fills.append({
                "ts": ts,
                "symbol": symbol,
                "fill_type": "EXIT",
                "side": -pos.side,
                "price": price,
                "size": pos.size,
                "order_type": order.order_type,
                "order_id": order.order_id,
                "pnl": pnl,
            })

            self.closed_trades.append({
                "symbol": symbol,
                "entry_ts": pos.entry_ts,
                "exit_ts": ts,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "size": pos.size,
                "pnl": pnl,
                "holding_sec": (ts - pos.entry_ts).total_seconds(),
                "win": pnl > 0,
            })
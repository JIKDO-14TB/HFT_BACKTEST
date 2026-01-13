# hft_backtest_engine/backtest_engine.py
from __future__ import annotations

from typing import Dict, Optional
import itertools
import pandas as pd

from hft_backtest_engine.data_loader import DataLoader
from hft_backtest_engine.strategy_base import (
    Strategy,
    StrategyState,
    Position,
    Order as StrategyOrder,   # ✅ 전략 주문
)
from hft_backtest_engine.execution import (
    ExecutionEngine,
    Order as ExecOrder,       # ✅ 실행 주문
    FillResult,
)
from hft_backtest_engine.feature_store import FeatureStore

COMPUTE_DELAY_MS = 10


class BacktestEngine:
    def __init__(
        self,
        loader: DataLoader,
        strategy: Strategy,
        execution: ExecutionEngine,
        feature_store: FeatureStore,
        initial_capital: float = 100.0,
        verbose: bool = True,
    ):
        self.loader = loader
        self.strategy = strategy
        self.execution = execution
        self.feature_store = feature_store
        self.verbose = verbose

        self.state = StrategyState(
            cash=initial_capital,
            positions={},
            current_ts=None,
        )

        # ✅ active_orders에는 "ExecOrder"만 들어간다
        self.active_orders: Dict[int, ExecOrder] = {}
        self.order_id_gen = itertools.count(1)

        self.fills = []
        self.closed_trades = []

        # book snapshot 중복 push 방지
        self._last_pushed_book_ts: Optional[pd.Timestamp] = None
        self._last_seen_book_ts: Optional[pd.Timestamp] = None

    def run_day(self, symbol: str, ymd: str):
        # 0) load
        df = self.loader.load_aggtrades_day(symbol, ymd)
        if df is None or df.empty:
            if self.verbose:
                print(f"[WARN] no aggTrades: {symbol} {ymd}")
            return

        try:
            book_df = self.loader.load_bookdepth_day(symbol, ymd)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] no bookDepth: {symbol} {ymd} ({e})")
            book_df = pd.DataFrame()

        try:
            klines = self.loader.load_klines_1m_day(symbol, ymd)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] no klines_1m: {symbol} {ymd} ({e})")
            klines = pd.DataFrame()

        trades = df.sort_values("ts").reset_index(drop=True)

        # 1) book snapshot index (ts -> row indices)
        book_ts_list = []
        book_groups = None
        if not book_df.empty:
            # 정렬 + groupby indices
            book_df = book_df.sort_values(["ts", "percentage"]).reset_index(drop=True)
            book_groups = book_df.groupby("ts", sort=True).indices  # dict: ts -> np.array idx
            book_ts_list = sorted(book_groups.keys())
        book_ptr = 0

        # 2) kline index (open_ts -> row) : 룩어헤드 방지(직전 분만 push)
        kline_map = None
        if not klines.empty:
            klines = klines.sort_values("open_ts").reset_index(drop=True)
            kline_map = klines.set_index("open_ts", drop=False)

        last_min = None  # 현재 tick의 minute

        for trade in trades.itertuples(index=False):
            ts = trade.ts
            self.state.current_ts = ts

            # (A) FeatureStore tick update
            self.feature_store.update_trade(trade)

            # (B) kline push (룩어헤드 방지: 새 minute 진입 시 직전 minute만 push)
            if kline_map is not None:
                cur_min = ts.floor("1min")
                if last_min is None:
                    last_min = cur_min
                elif cur_min > last_min:
                    last_completed_min = last_min
                    last_min = cur_min

                    if last_completed_min in kline_map.index:
                        self.feature_store.update_kline(kline_map.loc[[last_completed_min]])

            # (C) book snapshot push (ts까지 도달한 snapshot들 중 마지막 1개만 push)
            if book_groups is not None and book_ts_list:
                while book_ptr < len(book_ts_list) and book_ts_list[book_ptr] <= ts:
                    self._last_seen_book_ts = book_ts_list[book_ptr]
                    book_ptr += 1

                last_seen = self._last_seen_book_ts
                if last_seen is not None and self._last_pushed_book_ts != last_seen:
                    idxs = book_groups[last_seen]
                    snapshot = book_df.loc[idxs]
                    self.feature_store.update_book(snapshot)
                    self._last_pushed_book_ts = last_seen

            # (D) fill existing orders
            self._process_active_orders(trade)

            # (E-1) tick exit only
            exit_orders = self.strategy.on_tick(trade=trade, state=self.state)
            for o in exit_orders:
                self._submit_order(o, ts)

            # (E-2) signal entry only
            entry_orders = self.strategy.on_signal(trade=trade, state=self.state)
            for o in entry_orders:
                self._submit_order(o, ts)

    # =====================================================
    # 주문 제출: StrategyOrder -> ExecOrder로 변환
    # =====================================================
    def _submit_order(self, proto_order: StrategyOrder, ts: pd.Timestamp):
        oid = next(self.order_id_gen)
        created_ts = ts + pd.Timedelta(milliseconds=COMPUTE_DELAY_MS)

        expire_ts = None
        if proto_order.order_type == "limit":
            expire_ts = created_ts + pd.Timedelta(seconds=5)

        # ✅ ExecOrder는 execution.py의 dataclass 구조를 따른다
        order = ExecOrder(
            order_id=oid,
            symbol=proto_order.symbol,
            side=proto_order.side,
            size=float(proto_order.size),
            order_type=proto_order.order_type,
            price=None if proto_order.price is None else float(proto_order.price),
            created_ts=created_ts,
            expire_ts=expire_ts,
            status="PENDING",
        )

        self.active_orders[oid] = order

        if self.verbose:
            print(
                f"[ORDER] {created_ts} {order.symbol} "
                f"{order.order_type.upper()} id={oid} (compute+{COMPUTE_DELAY_MS}ms)"
            )

    # =====================================================
    # active_orders 처리
    # =====================================================
    def _process_active_orders(self, trade):
        to_remove = []

        for oid, order in list(self.active_orders.items()):
            if trade.ts < self.execution.order_active_ts(order):
                continue

            # limit 만료 → 취소 후 market 강제
            if order.expire_ts and trade.ts >= order.expire_ts:
                order.status = "CANCELED"
                to_remove.append(oid)

                if self.verbose:
                    print(f"[CANCEL] {trade.ts} {order.symbol} LIMIT id={oid}")

                market_oid = next(self.order_id_gen)
                market_order = ExecOrder(
                    order_id=market_oid,
                    symbol=order.symbol,
                    side=order.side,
                    size=order.size,
                    order_type="market",
                    price=None,
                    created_ts=trade.ts,
                    expire_ts=None,
                    status="PENDING",
                )
                self.active_orders[market_oid] = market_order

                if self.verbose:
                    print(f"[FORCE MARKET] {trade.ts} {order.symbol}")

                continue

            fill: FillResult = self.execution.try_fill(order, trade)
            if fill.filled:
                self._apply_fill(order, fill, trade.ts)
                to_remove.append(oid)

        for oid in to_remove:
            self.active_orders.pop(oid, None)

    # =====================================================
    # 체결 반영 (fee 포함)
    # =====================================================
    def _apply_fill(self, order: ExecOrder, fill: FillResult, ts: pd.Timestamp):
        symbol = order.symbol
        price = float(fill.fill_price)

        # ExecutionEngine이 아직 fee 필드를 안 넣어줘도 죽지 않게 방어
        fee_bp = float(getattr(fill, "fee_bp", 0.0))
        fee_amt = float(getattr(fill, "fee_amt", 0.0))

        # =========================
        # 진입
        # =========================
        if symbol not in self.state.positions:
            pos = Position(
                symbol=symbol,
                size=order.size,
                side=order.side,
                entry_price=price,
                entry_ts=ts,
            )
            # ✅ entry fee 저장(나중에 exit에서 net_pnl 계산에 포함)
            setattr(pos, "entry_fee_amt", fee_amt)

            self.state.positions[symbol] = pos
            self.state.cash -= (order.size + fee_amt)

            self.fills.append({
                "ts": ts,
                "symbol": symbol,
                "fill_type": "ENTER",
                "side": order.side,
                "price": price,
                "size": order.size,
                "order_type": order.order_type,
                "order_id": order.order_id,
                "fee_bp": fee_bp,
                "fee_amt": fee_amt,
                "gross_pnl": 0.0,
                "net_pnl": -fee_amt,
            })

            if self.verbose:
                print(f"[FILL ENTER] {ts} {symbol} price={price:.4f} fee_bp={fee_bp:.2f}")

        # =========================
        # 청산
        # =========================
        else:
            pos = self.state.positions.pop(symbol)

            entry_fee_amt = float(getattr(pos, "entry_fee_amt", 0.0))

            gross_pnl = (
                pos.side * (price - pos.entry_price)
                / pos.entry_price * pos.size
            )
            total_fee_amt = entry_fee_amt + fee_amt
            net_pnl = gross_pnl - total_fee_amt

            # cash update:
            # entry 때 이미 (size + entry_fee) 빠졌고,
            # exit 때는 (size + gross_pnl - exit_fee) 더해주면 최종이 net 반영됨
            self.state.cash += (pos.size + gross_pnl - fee_amt)

            self.fills.append({
                "ts": ts,
                "symbol": symbol,
                "fill_type": "EXIT",
                "side": -pos.side,
                "price": price,
                "size": pos.size,
                "order_type": order.order_type,
                "order_id": order.order_id,
                "fee_bp": fee_bp,
                "fee_amt": fee_amt,
                "entry_fee_amt": entry_fee_amt,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
            })

            self.closed_trades.append({
                "symbol": symbol,
                "entry_ts": pos.entry_ts,
                "exit_ts": ts,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "size": pos.size,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "entry_fee_amt": entry_fee_amt,
                "exit_fee_amt": fee_amt,
                "total_fee_amt": total_fee_amt,
                "holding_sec": (ts - pos.entry_ts).total_seconds(),
                "win": net_pnl > 0,
            })

            if self.verbose:
                print(f"[FILL EXIT ] {ts} {symbol} gross={gross_pnl:.4f} net={net_pnl:.4f} fee_bp={fee_bp:.2f}")



from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from hft_backtest_engine.feature_store import FeatureStore


@dataclass
class Position:
    symbol: str
    size: float
    side: int
    entry_price: float
    entry_ts: pd.Timestamp


@dataclass
class StrategyState:
    cash: float
    positions: Dict[str, Position]
    current_ts: pd.Timestamp
    last_signal_ts: Optional[pd.Timestamp] = None
    last_score: float = 0.0
    score_history: List[float] = field(default_factory=list)


@dataclass
class Order:
    symbol: str
    side: int
    size: float
    order_type: str
    price: float | None


class Strategy:
    def __init__(
        self,
        symbol: str,
        feature_store: FeatureStore,
        initial_capital: float = 100.0,
        signal_interval_seconds: int = 5 * 60,
        max_holding_seconds: int = 4 * 60,
        target_bp: float = 30.0, #20bp로 수정했음.
        max_leverage_notional: float = 1.0,
        score_window: int = 288,
    ):
        self.symbol = symbol
        self.feature_store = feature_store
        self.initial_capital = initial_capital

        self.signal_interval_seconds = signal_interval_seconds
        self.max_holding_seconds = max_holding_seconds
        self.target_bp = target_bp

        self.max_leverage_notional = max_leverage_notional
        self.score_window = score_window
        self.feature_logs = []

    def compute_score(self, trade, state: StrategyState) -> float:
        ts = trade.ts
        feats = self.feature_store.get_features(ts)

        vpin_cdf = feats.get("vpin_cdf", float("nan"))
        z_ofi = feats.get("z_ofi", float("nan"))
        qr = feats.get("qr", float("nan"))
        n_cdf = feats.get("n_cdf", float("nan"))

        # 데이터 부족/결측이면 관망
        if pd.isna(vpin_cdf) or pd.isna(z_ofi) or pd.isna(qr) or pd.isna(n_cdf):
            score = 0.0
        else:
            # vpin gate는 필요 시 다시 켜면 됨
            #if vpin_cdf <= 0.7: score = 0.0
            v_score = 0.7 * z_ofi + 0.3 * qr
            score = v_score * n_cdf

        self.feature_logs.append({
            "ts": ts,
            "vpin_cdf": vpin_cdf,
            "z_ofi": z_ofi,
            "qr": qr,
            "n_cdf": n_cdf,
            "score": float(score),
        })

        return float(score)

    def _should_recompute_signal(self, ts: pd.Timestamp, state: StrategyState) -> bool:
        if state.last_signal_ts is None:
            return True
        return (ts - state.last_signal_ts).total_seconds() >= self.signal_interval_seconds

    def _score_to_weight(self, score: float, state: StrategyState) -> float:
        state.score_history.append(float(score))
        if len(state.score_history) > self.score_window:
            state.score_history = state.score_history[-self.score_window:]

        mn = min(state.score_history)
        mx = max(state.score_history)
        if mx == mn:
            return 0.0

        norm01 = (score - mn) / (mx - mn)
        weight = 2.0 * norm01 - 1.0
        return max(-1.0, min(1.0, weight))

    # =========================
    # NEW: tick 청산 전용
    # =========================
    def on_tick(self, trade, state: StrategyState) -> List[Order]:
        orders: List[Order] = []
        ts = trade.ts
        price = float(trade.price)

        pos = state.positions.get(self.symbol)
        if pos is None:
            return orders

        holding_time = (ts - pos.entry_ts).total_seconds()
        pnl_bp = pos.side * (price - pos.entry_price) / pos.entry_price * 1e4

        if pnl_bp >= self.target_bp:
            orders.append(Order(self.symbol, -pos.side, pos.size, "limit", price))
        elif holding_time >= self.max_holding_seconds:
            orders.append(Order(self.symbol, -pos.side, pos.size, "market", None))

        return orders

    # =========================
    # NEW: 5분 시그널 진입 전용
    # =========================
    def on_signal(self, trade, state: StrategyState) -> List[Order]:
        orders: List[Order] = []
        ts = trade.ts

        # 포지션 있으면 진입 금지
        if self.symbol in state.positions:
            return orders

        if not self._should_recompute_signal(ts, state):
            return orders

        score = self.compute_score(trade, state)
        state.last_signal_ts = ts
        state.last_score = score

        weight = self._score_to_weight(score, state)
        if weight == 0.0:
            return orders

        base_cash = max(state.cash, 0.0)
        size = base_cash * self.max_leverage_notional * abs(weight)
        side = 1 if weight > 0 else -1

        orders.append(Order(self.symbol, side, size, "market", None))
        return orders

    # =========================
    # 기존 호환용: on_trade는 wrapper
    # =========================
    def on_trade(self, trade, state: StrategyState) -> List[Order]:
        out = []
        out.extend(self.on_tick(trade, state))
        # 청산 주문이 나가면 같은 tick에 진입은 금지
        if out:
            return out
        out.extend(self.on_signal(trade, state))
        return out




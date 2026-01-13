from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from hft_backtest_engine.feature_store_kline import FeatureStoreKline


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


@dataclass
class Order:
    symbol: str
    side: int
    size: float
    order_type: str
    price: float | None


class CloseZStrategy:
    def __init__(
        self,
        symbol: str,
        feature_store: FeatureStoreKline,
        initial_capital: float = 100.0,
        window: int = 60,
        entry_z: float = 1.0,             # 임계값 (예: |z|>=1.0 진입)
        max_holding_seconds: int = 240,    # 240초
        target_bp: float = 10.0,           # 익절 bp
        max_notional: float = 1.0,         # 최대 비중(초기자본 대비)
        weight_clip: float = 2.5,          # z를 [-2.5,2.5]로 클립 후 스케일
    ):
        self.symbol = symbol
        self.fs = feature_store
        self.initial_capital = initial_capital

        self.window = window
        self.entry_z = entry_z
        self.max_holding_seconds = max_holding_seconds
        self.target_bp = target_bp

        self.max_notional = max_notional
        self.weight_clip = weight_clip

        self.feature_logs: List[dict] = []

    def _z_to_weight(self, z: float) -> float:
        if np.isnan(z):
            return 0.0
        z = float(np.clip(z, -self.weight_clip, self.weight_clip))
        # [-clip, clip] -> [-1,1]
        return z / self.weight_clip

    def on_bar(self, bar, state: StrategyState) -> List[Order]:
        """
        bar: klines_1m row (itertuples) with open_ts, close, ...
        """
        ts = bar.open_ts
        price = float(bar.close)

        orders: List[Order] = []

        pos = state.positions.get(self.symbol)

        # =========================
        # 1) Exit logic (매 bar)
        # =========================
        if pos is not None:
            holding = (ts - pos.entry_ts).total_seconds()
            pnl_bp = pos.side * (price - pos.entry_price) / pos.entry_price * 1e4

            if pnl_bp >= self.target_bp:
                orders.append(Order(self.symbol, -pos.side, pos.size, "limit", price))
                return orders

            if holding >= self.max_holding_seconds:
                orders.append(Order(self.symbol, -pos.side, pos.size, "market", None))
                return orders

            return orders  # 포지션 있으면 신규진입 없음

        # =========================
        # 2) Entry logic (포지션 0일 때만)
        # =========================
        feats = self.fs.get_features(ts)
        z = feats["z_close"]

        # 로그
        self.feature_logs.append({"ts": ts, "close": feats["close"], "z_close": z})

        if np.isnan(z) or abs(z) < self.entry_z:
            return orders

        weight = self._z_to_weight(z)  # [-1,1]
        side = 1 if weight > 0 else -1

        # 단리: sizing은 initial_capital 기준(원하면 state.cash로 바꿔도 됨)
        size = self.initial_capital * self.max_notional * abs(weight)
        if size <= 0:
            return orders

        orders.append(Order(self.symbol, side, size, "market", None))
        return orders
# new_backtester.py
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

Action = Optional[str]  # "LONG" | "SHORT" | None


@dataclass
class BacktestConfig:
    initial_capital: float = 100.0

    maker_fee_bp: float = 5.0
    taker_fee_bp: float = 2.0

    latency_ms: int = 10
    decision_ms: int = 10  # 🔥 전략 판단 주기
    max_holding_seconds: int = 240

    limit_ratio: float = 0.7
    target_bp: float = 3.0
    partial_bp: float = 1.0
    stop_bp: float = 10.0


class SimpleBacktester:
    def __init__(self, quotes: pd.DataFrame, *, config: BacktestConfig):
        if quotes.empty:
            raise ValueError("quotes empty")

        quotes = quotes.copy()
        ts = pd.to_datetime(quotes["ts"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        quotes["ts"] = ts.astype("datetime64[ns]")

        self.quotes = quotes.sort_values("ts").reset_index(drop=True)
        self.cfg = config

        self.ts_ns = self.quotes["ts"].values.astype("int64")
        self.price = self.quotes["price"].values
        self.bid_qty = self.quotes["best_bid_qty"].values
        self.ask_qty = self.quotes["best_ask_qty"].values

        self.latency_ns = config.latency_ms * 1_000_000
        self.decision_ns = config.decision_ms * 1_000_000
        self.max_hold_ns = config.max_holding_seconds * 1_000_000_000

    @staticmethod
    def _fee(notional, bp):
        return notional * bp / 10000.0

    def run(
        self,
        signal_func: Callable[[Dict, Dict], Action],
        *,
        start_capital: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, float, float]:

        state = {
            "position": None,
            "entry_price": 0.0,
            "entry_ts": 0,
            "capital": self.cfg.initial_capital if start_capital is None else start_capital,
            "turnover": 0.0,
            "last_decision_ts": -1,
            "trades": [],
        }

        exec_i = 0
        n = len(self.ts_ns)

        for i in range(n):
            decision_ts = self.ts_ns[i]

            # latency
            target_ts = decision_ts + self.latency_ns
            while exec_i < n and self.ts_ns[exec_i] < target_ts:
                exec_i += 1
            if exec_i >= n:
                break

            mid = self.price[exec_i]

            # ======================
            # STOP LOSS
            # ======================
            if state["position"] is not None:
                side = state["position"]
                entry = state["entry_price"]
                ret_bp = (
                    (mid - entry) / entry * 10000
                    if side == "LONG"
                    else (entry - mid) / entry * 10000
                )
                if ret_bp <= -self.cfg.stop_bp:
                    self._exit(exec_i, state)
                    continue

            # TIME STOP
            if state["position"] is not None:
                if self.ts_ns[exec_i] - state["entry_ts"] >= self.max_hold_ns:
                    self._exit(exec_i, state)
                    continue

            # ======================
            # ENTRY (10ms gate)
            # ======================
            if state["position"] is None:
                if decision_ts - state["last_decision_ts"] < self.decision_ns:
                    continue

                action = signal_func({"i": i}, state)
                state["last_decision_ts"] = decision_ts

                if action in ("LONG", "SHORT"):
                    self._enter(exec_i, action, state)

        return pd.DataFrame(state["trades"]), state["capital"], state["turnover"]

    def _enter(self, i, side, state):
        cap = state["capital"]
        d = 1 if side == "LONG" else -1
        mid = self.price[i]

        ln = cap * self.cfg.limit_ratio
        mn = cap - ln

        limit_p = mid * (1 - d * self.cfg.target_bp / 10000)
        market_p = mid * (1 + d * self.cfg.partial_bp / 10000)

        fee = self._fee(ln, self.cfg.maker_fee_bp) + self._fee(
            mn, self.cfg.taker_fee_bp
        )
        avg_p = (ln * limit_p + mn * market_p) / cap

        state["capital"] -= fee
        state["position"] = side
        state["entry_price"] = avg_p
        state["entry_ts"] = self.ts_ns[i]
        state["turnover"] += cap

        state["trades"].append(
            {"ts": self.quotes["ts"].iloc[i], "type": "ENTER", "side": side, "capital": state["capital"]}
        )

    def _exit(self, i, state):
        cap = state["capital"]
        d = 1 if state["position"] == "LONG" else -1
        mid = self.price[i]

        ret = d * (mid - state["entry_price"]) / state["entry_price"]
        pnl = cap * ret - self._fee(cap, self.cfg.taker_fee_bp)

        state["capital"] += pnl
        state["turnover"] += cap

        state["trades"].append(
            {"ts": self.quotes["ts"].iloc[i], "type": "EXIT", "capital": state["capital"]}
        )

        state["position"] = None

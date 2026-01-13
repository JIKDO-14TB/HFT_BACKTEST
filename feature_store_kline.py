from __future__ import annotations
from typing import Dict, Optional
import pandas as pd

from hft_backtest_engine.features import compute_close_zscore

class FeatureStoreKline:
    def __init__(self, window: int = 60):
        self.window = window
        self.kline_buffer = pd.DataFrame()
        self.cached: Optional[Dict[str, float]] = None
        self.last_ts: Optional[pd.Timestamp] = None

    def update_kline(self, k: pd.DataFrame):
        # k: 1-row DataFrame expected
        if "open_ts" not in k.columns or "close" not in k.columns:
            raise ValueError(f"kline missing columns: {k.columns}")

        self.kline_buffer = pd.concat([self.kline_buffer, k], ignore_index=True)

        # 최근 window+5 정도만 유지 (메모리 절약)
        keep = self.window + 5
        if len(self.kline_buffer) > keep:
            self.kline_buffer = self.kline_buffer.iloc[-keep:].reset_index(drop=True)

    def get_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        # 같은 분에 여러 번 부르면 캐시 반환
        if self.last_ts is not None and ts == self.last_ts and self.cached is not None:
            return self.cached

        z = compute_close_zscore(self.kline_buffer, window=self.window)
        feats = {"close": z["close"], "z_close": z["z_close"], "cdf": z["cdf"]}

        self.last_ts = ts
        self.cached = feats
        return feats
# hft_backtest_engine/feature_store.py
from __future__ import annotations

from typing import Dict, Optional, Deque, List
from collections import deque

import pandas as pd

from hft_backtest_engine.features import (
    VPINCalculator,
    OFICalculator,
    compute_trade_count_spike,
    compute_qr,
)


class FeatureStore:
    """
    FeatureStore (STATEFUL + FAST)
    ------------------------------
    - tick에서는 append만 수행 (no pd.concat)
    - 5분 시그널 시점에 필요한 만큼만 "작게" DataFrame 변환
    """

    def __init__(
        self,
        symbol: str,
        signal_interval_seconds: int = 5 * 60,
        vpin_bucket_volume: float = 1e6,
        vpin_history: int = 100,
        tc_window: int = 60,
        ofi_window_minutes: int = 5,
        ofi_z_window: int = 30,
        # 성능/메모리 튜닝
        keep_book_minutes: int = 10,   # book은 최근 N분만 들고 간다 (>= ofi_window_minutes)
        keep_agg_debug_rows: int = 0,  # 0이면 agg debug 저장 안 함(추천)
    ):
        self.symbol = symbol
        self.signal_interval_seconds = signal_interval_seconds

        # =========================
        # Stateful calculators
        # =========================
        self.vpin_calc = VPINCalculator(
            bucket_volume=vpin_bucket_volume,
            history=vpin_history,
        )
        self.ofi_calc = OFICalculator(
            z_window=ofi_z_window,
        )

        self.tc_window = tc_window
        self.ofi_window_minutes = ofi_window_minutes
        self.keep_book_minutes = max(keep_book_minutes, ofi_window_minutes + 1)

        # =========================
        # FAST buffers
        # =========================
        # kline: 최근 (tc_window + 2)개만 유지
        self._kline_rows: Deque[dict] = deque(maxlen=self.tc_window + 2)

        # book: snapshot df를 ts와 함께 들고 가되, 오래된 건 prune
        # (deque maxlen은 넉넉히, 실제 prune은 시간 기준으로)
        self._book_snaps: Deque[pd.DataFrame] = deque(maxlen=50_000)
        self._last_book_ts: Optional[pd.Timestamp] = None
        self._last_book_snap: Optional[pd.DataFrame] = None  # QR용

        # agg: VPIN은 stateful이라 사실 저장 불필요 (디버깅용 옵션)
        self.keep_agg_debug_rows = int(keep_agg_debug_rows)
        self._agg_debug: Deque[dict] = deque(maxlen=self.keep_agg_debug_rows) if self.keep_agg_debug_rows > 0 else deque(maxlen=1)

        # cache
        self.last_feature_ts: Optional[pd.Timestamp] = None
        self.cached_features: Optional[Dict[str, float]] = None

    # =====================================================
    # Update API
    # =====================================================

    def update_trade(self, trade_row) -> None:
        """aggTrades tick 1개"""
        # ✅ VPIN 증분 업데이트 (중복 없는 구조)
        self.vpin_calc.update_trade(trade_row)

        # (선택) 디버깅용 저장
        if self.keep_agg_debug_rows > 0:
            self._agg_debug.append({
                "ts": trade_row.ts,
                "price": float(trade_row.price),
                "quantity": float(trade_row.quantity),
                "is_buyer_maker": bool(trade_row.is_buyer_maker),
            })

    def update_kline(self, kline_df: pd.DataFrame) -> None:
        """
        1분봉 DataFrame (BacktestEngine에서 분 바뀔 때 k=klines[open_ts==minute] 전달하는 구조)
        - 여기서는 "한 분(row 1개)"만 들어오는 걸 전제로 가장 최근 row만 append
        """
        if kline_df is None or kline_df.empty:
            return
        if "open_ts" not in kline_df.columns:
            raise ValueError(f"kline_df must contain open_ts. cols={list(kline_df.columns)}")
        if "trades" not in kline_df.columns:
            raise ValueError(f"kline_df must contain trades. cols={list(kline_df.columns)}")

        row = kline_df.iloc[-1]
        self._kline_rows.append({
            "open_ts": row["open_ts"],
            "trades": float(row["trades"]),
        })

    def update_book(self, book_df: pd.DataFrame) -> None:
        """
        bookDepth snapshot(ts 동일한 여러 row) 업데이트
        - BacktestEngine이 snapshot = book_df[book_df["ts"] == last_book_ts]로 넘겨줌
        """
        if book_df is None or book_df.empty:
            return
        if "ts" not in book_df.columns:
            raise ValueError(f"book_df must contain ts. cols={list(book_df.columns)}")

        snap_ts = book_df["ts"].iloc[0]
        # 같은 ts가 중복 push되는 경우 방지
        if self._last_book_ts is not None and snap_ts == self._last_book_ts:
            return

        self._book_snaps.append(book_df)
        self._last_book_ts = snap_ts
        self._last_book_snap = book_df

        # ✅ 시간 기준 prune (최근 keep_book_minutes만 유지)
        cutoff = snap_ts - pd.Timedelta(minutes=self.keep_book_minutes)
        while self._book_snaps and self._book_snaps[0]["ts"].iloc[0] < cutoff:
            self._book_snaps.popleft()

    # =====================================================
    # Feature computation
    # =====================================================

    def should_compute(self, ts: pd.Timestamp) -> bool:
        if self.last_feature_ts is None:
            return True
        return (ts - self.last_feature_ts).total_seconds() >= self.signal_interval_seconds

    def compute_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        # -------------------------
        # 1) VPIN (stateful)
        # -------------------------
        vpin = self.vpin_calc.get_value()
        vpin_cdf = vpin.get("vpin_cdf", 0.0)
        if pd.isna(vpin_cdf):
            vpin = {"vpin_raw": float("nan"), "vpin_cdf": 0.0}

        # -------------------------
        # 2) Trade Count Spike
        # -------------------------
        if len(self._kline_rows) < self.tc_window + 1:
            tc = {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}
        else:
            kline_df = pd.DataFrame(list(self._kline_rows)).sort_values("open_ts")
            tc = compute_trade_count_spike(kline_df, window=self.tc_window)
            if pd.isna(tc.get("n_cdf", float("nan"))):
                tc = {"tc": float(tc.get("tc", 0.0)), "z_tc": 0.0, "n_cdf": 0.5}

        # -------------------------
        # 3) OFI (최근 5분 slice만)
        #    - book 전체 concat 금지: deque에 남아있는 것만 합친다(이미 10분 이하)
        # -------------------------
        if not self._book_snaps:
            ofi = {"ofi_raw": 0.0, "z_ofi": 0.0}
        else:
            # deque는 이미 최근 N분만 유지하므로 여기 concat은 "작다"
            book_df = pd.concat(list(self._book_snaps), ignore_index=True).sort_values("ts")

            start = ts - pd.Timedelta(minutes=self.ofi_window_minutes)
            book_slice = book_df[(book_df["ts"] >= start) & (book_df["ts"] <= ts)]

            if book_slice.empty or book_slice["ts"].nunique() < 2:
                ofi = {"ofi_raw": 0.0, "z_ofi": 0.0}
            else:
                ofi = self.ofi_calc.update(book_slice)

        # -------------------------
        # 4) QR (마지막 snapshot으로 바로 계산)
        # -------------------------
        if self._last_book_snap is None:
            qr = 0.0
        else:
            qr = compute_qr(self._last_book_snap)

        features = {
            "ts": ts,
            "vpin_raw": float(vpin.get("vpin_raw", float("nan"))),
            "vpin_cdf": float(vpin.get("vpin_cdf", 0.0)),
            "tc": float(tc.get("tc", 0.0)),
            "z_tc": float(tc.get("z_tc", 0.0)),
            "n_cdf": float(tc.get("n_cdf", 0.5)),
            "ofi_raw": float(ofi.get("ofi_raw", 0.0)),
            "z_ofi": float(ofi.get("z_ofi", 0.0)),
            "qr": float(qr),
        }

        self.last_feature_ts = ts
        self.cached_features = features
        return features

    def get_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        if self.cached_features is None or self.should_compute(ts):
            return self.compute_features(ts)
        return self.cached_features


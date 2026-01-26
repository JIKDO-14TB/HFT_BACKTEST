# new_features.py
from collections import deque
import numpy as np


class RollingZTime:
    """
    Time-based rolling z-score.
    Window is defined in seconds (converted internally to ns).
    """

    def __init__(self, window_sec: int):
        self.window_ns = int(window_sec * 1_000_000_000)
        self.buf = deque()  # (ts_ns, value)

    def update(self, ts_ns: int, x: float) -> float:
        if np.isnan(x):
            return np.nan

        self.buf.append((ts_ns, float(x)))

        # drop old
        while self.buf and (ts_ns - self.buf[0][0]) > self.window_ns:
            self.buf.popleft()

        if len(self.buf) < 2:
            return np.nan

        vals = np.fromiter((v for _, v in self.buf), dtype=np.float64)
        std = vals.std()
        if std == 0:
            return np.nan
        return (vals[-1] - vals.mean()) / std

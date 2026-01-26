import numpy as np
import numba as nb


# =========================
# Rolling Z-score (Numba)
# =========================
@nb.njit
def rolling_z_numba(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)

    buf = np.zeros(window)
    s = 0.0
    ss = 0.0
    cnt = 0

    for i in range(n):
        xi = x[i]
        if np.isnan(xi):
            continue

        if cnt < window:
            buf[cnt] = xi
            s += xi
            ss += xi * xi
            cnt += 1
        else:
            j = i % window
            old = buf[j]
            buf[j] = xi
            s += xi - old
            ss += xi * xi - old * old

        if cnt == window:
            mean = s / window
            var = ss / window - mean * mean
            if var > 0:
                out[i] = (xi - mean) / np.sqrt(var)

    return out


# =========================
# Run-length (Numba)
# =========================
@nb.njit
def run_length_numba(sign: np.ndarray) -> np.ndarray:
    n = len(sign)
    out = np.zeros(n, dtype=np.int32)

    prev = 0
    cnt = 0

    for i in range(n):
        s = sign[i]
        if s == 0:
            cnt = 0
            out[i] = 0
        elif s == prev:
            cnt += 1
            out[i] = cnt
        else:
            cnt = 1
            out[i] = 1
            prev = s

    return out


# =========================
# Tick spread in bp
# =========================
@nb.njit
def tick_spread_bp(price: np.ndarray) -> np.ndarray:
    n = len(price)
    out = np.full(n, np.nan)

    for i in range(1, n):
        if price[i] > 0 and price[i - 1] > 0:
            out[i] = abs(price[i] - price[i - 1]) / price[i - 1] * 10000.0

    return out

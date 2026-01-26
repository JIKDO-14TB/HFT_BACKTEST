"""Common statistical indicators and helper functions.

This module provides rolling z‑scores and ordinary least squares (OLS)
regressions computed on sliding windows.  The functions operate on
``pandas.Series`` objects and return results aligned to the input
index.  They are intentionally lightweight and do not depend on
heavyweight third party libraries such as ``statsmodels``.  If you
prefer to use a specialised library, feel free to replace these
implementations or wrap them to fit your needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute the z‑score of a series over a rolling window.

    The z‑score is defined as ``(x - mean) / std``, where the mean and
    standard deviation are computed over the trailing ``window``
    observations.  ``NaN`` values are propagated in the result for
    periods where insufficient data exists.

    Parameters
    ----------
    series : pandas.Series
        Input data.  Must be numeric.
    window : int
        Size of the trailing window.

    Returns
    -------
    pandas.Series
        Z‑scores aligned to ``series``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)
    z = (series - rolling_mean) / rolling_std
    return z


def rolling_ols(
    y: pd.Series, x: pd.Series, window: int
) -> tuple[pd.Series, pd.Series]:
    """Compute rolling ordinary least squares regression of ``y`` on ``x``.

    For each rolling window the slope (beta) and intercept (alpha) are
    estimated by minimising squared residuals.  Missing values in the
    inputs are ignored, but if fewer than ``window`` valid observations
    are available the result for that window is ``NaN``.

    Parameters
    ----------
    y, x : pandas.Series
        Dependent and independent variables.
    window : int
        Number of observations to use in each regression.

    Returns
    -------
    (beta, alpha) : tuple of pandas.Series
        Series containing the slope and intercept estimates for each
        window.  Both series are aligned to the input index.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    # Prepare containers for results
    betas = pd.Series(index=y.index, dtype="float64")
    alphas = pd.Series(index=y.index, dtype="float64")
    # Precompute cumulative sums for efficiency
    sum_x = x.rolling(window=window, min_periods=window).sum()
    sum_y = y.rolling(window=window, min_periods=window).sum()
    sum_x2 = (x * x).rolling(window=window, min_periods=window).sum()
    sum_xy = (x * y).rolling(window=window, min_periods=window).sum()
    n = window
    # Compute beta and alpha using the closed form solution
    denom = n * sum_x2 - sum_x * sum_x
    # Avoid division by zero: where denom==0 set to NaN
    beta_series = (n * sum_xy - sum_x * sum_y) / denom
    alpha_series = (sum_y - beta_series * sum_x) / n
    betas.update(beta_series)
    alphas.update(alpha_series)
    return betas, alphas
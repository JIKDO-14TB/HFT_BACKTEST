"""Top‑level package for a simple, modular backtesting engine.

This module exposes a handful of classes and utilities to load market data
from parquet files, compute common statistics (e.g. tick spreads) and
provide a foundation for building strategy logic and execution loops.  The
design philosophy is to keep each component small and reusable so that
additional features or asset types can be plugged in without rewriting the
whole system.  See the individual classes for more details.
"""

from .new_data_loader import DataLoader

__all__ = ["DataLoader"]
"""Reusable analytical metrics for the dashboard.

Kept dependency-light (numpy + pandas) and pure so each can be unit-checked
in isolation and reused across the Growth / Impact / Concentration tabs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def gini(values) -> float:
    """Gini coefficient of a non-negative distribution.

    0 = perfect equality, 1 = maximal concentration. Used for citation
    inequality (Impact / Concentration tabs).
    """
    x = np.asarray([v for v in values if v is not None], dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(np.clip(x, 0, None))
    n = x.size
    cum = np.cumsum(x)
    # Mean absolute difference formulation.
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * cum[-1]) / (n * cum[-1]))


def lorenz_curve(values):
    """Return (x, y) points of the Lorenz curve, both starting at (0, 0)."""
    x = np.sort(np.asarray([v for v in values if v is not None], dtype=float))
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cum = np.cumsum(x)
    cum_share = np.insert(cum / cum[-1], 0, 0.0)
    pop_share = np.linspace(0.0, 1.0, x.size + 1)
    return pop_share, cum_share


def shannon_entropy(counts) -> float:
    """Shannon entropy (in bits) of a categorical distribution.

    Higher = attention spread evenly across topics; lower = concentrated in a
    few topics. Used for the Concentration tab's "Topic Entropy" KPI.
    """
    c = np.asarray([v for v in counts if v and v > 0], dtype=float)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def cagr(start_value: float, end_value: float, periods: int) -> float:
    """Compound annual growth rate as a fraction (0.25 == 25%)."""
    if periods <= 0 or start_value <= 0 or end_value <= 0:
        return float("nan")
    return float((end_value / start_value) ** (1.0 / periods) - 1.0)


def yoy_growth(series: pd.Series) -> pd.Series:
    """Year-over-year growth rate (%) for a year-indexed count series."""
    return series.sort_index().pct_change() * 100.0


def yoy_acceleration(series: pd.Series) -> pd.Series:
    """Change in YoY growth rate vs the previous year, in percentage points."""
    return yoy_growth(series).diff()


def top_n_share(values, n: int) -> float:
    """Share (%) of the total held by the largest ``n`` entries."""
    x = np.sort(np.asarray([v for v in values if v is not None], dtype=float))[::-1]
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    return float(100.0 * x[:n].sum() / x.sum())

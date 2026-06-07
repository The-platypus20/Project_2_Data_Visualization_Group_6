"""Cached readers for the Tab 3 "What drives impact" ML cache files.

Mirrors the narrative_data pattern: the Shiny app only ever reads the small
precomputed ``tab3_*.csv`` files produced offline by
``src/preprocess/build_impact_ml_cache.py`` so every chart renders instantly.
"""
from __future__ import annotations

import functools

import pandas as pd

from .narrative_data import _cache_csv


@functools.lru_cache(maxsize=1)
def lorenz() -> pd.DataFrame:
    return _cache_csv("tab3_lorenz.csv", ["paper_frac", "citation_frac"])


@functools.lru_cache(maxsize=1)
def funnel() -> pd.DataFrame:
    return _cache_csv("tab3_funnel.csv", ["stage", "count"])


@functools.lru_cache(maxsize=1)
def concentration() -> dict[str, float]:
    df = _cache_csv("tab3_concentration.csv", ["metric", "value"])
    if df.empty:
        return {}
    return dict(zip(df["metric"], pd.to_numeric(df["value"], errors="coerce")))


@functools.lru_cache(maxsize=1)
def correlations() -> pd.DataFrame:
    return _cache_csv(
        "tab3_correlations.csv",
        ["dimension", "group", "order", "n_papers", "high_impact_rate"],
    )


@functools.lru_cache(maxsize=1)
def drivers() -> pd.DataFrame:
    return _cache_csv("tab3_drivers.csv", ["feature", "coef", "direction"])


@functools.lru_cache(maxsize=1)
def model_metrics() -> pd.DataFrame:
    return _cache_csv("tab3_model_metrics.csv", ["model", "roc_auc", "pr_auc", "lift_at_10"])


@functools.lru_cache(maxsize=1)
def roc_curve() -> pd.DataFrame:
    return _cache_csv("tab3_roc_curve.csv", ["fpr", "tpr"])


@functools.lru_cache(maxsize=1)
def calibration() -> pd.DataFrame:
    return _cache_csv("tab3_calibration.csv", ["predicted", "observed", "count"])


@functools.lru_cache(maxsize=1)
def forecast() -> pd.DataFrame:
    return _cache_csv("tab3_forecast.csv", ["family", "year", "share", "kind", "lo", "hi"])

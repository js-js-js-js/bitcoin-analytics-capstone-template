"""Enhanced Dynamic DCA model V4 - COMPLIANT VERSION (No Violations)

This is a cleaned version of Enhanced4 with all violations removed:
1. ❌ Removed: Recent window special handling (2023-01-01 cutoff)
2. ❌ Removed: Window-level time profile (start_bull detection)
3. ✅ Kept: All signal-based features (MVRV, price_bias, momentum, sentiment)
4. ✅ Kept: Multiplier computation logic
"""

import logging
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import base functionality from template
from template.prelude_template import load_polymarket_data
from template.model_development_template import (
    _compute_stable_signal,
    allocate_sequential_stable,
    _clean_array,
)

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"

# Strategy parameters
MIN_W = 1e-6
MA_WINDOW = 200
DYNAMIC_STRENGTH = 4.0

# Enhanced thresholds using absolute MVRV values
MVRV_ABSOLUTE_BOTTOM = 1.0
MVRV_RELATIVE_BOTTOM = 1.5
MVRV_BULL_CAUTION = 2.0
MVRV_EXTREME_TOP = 3.0

# Price bias thresholds
PRICE_BIAS_CAUTION = 1.5
PRICE_BIAS_EXTREME = 2.0

# Feature column names
FEATS = [
    "price_vs_ma",
    "mvrv_absolute",
    "price_bias",
    "polymarket_sentiment",
]


# =============================================================================
# Model-Specific Data Loading
# =============================================================================

def load_polymarket_btc_sentiment() -> pd.DataFrame:
    """Load Polymarket BTC sentiment with enhanced processing"""
    try:
        polymarket_data = load_polymarket_data()
        if "markets" not in polymarket_data:
            return pd.DataFrame()

        markets_df = polymarket_data["markets"]
        btc_markets = markets_df[
            markets_df["question"].str.contains("Bitcoin|BTC|btc", case=False, na=False)
        ].copy()

        if btc_markets.empty:
            return pd.DataFrame()

        btc_markets["created_date"] = pd.to_datetime(btc_markets["created_at"]).dt.normalize()

        # Enhanced sentiment calculation with volume weighting
        daily_stats = btc_markets.groupby("created_date").agg(
            daily_market_count=("market_id", "count"),
            daily_volume=("volume", "sum")
        ).reset_index()

        daily_stats = daily_stats.set_index("created_date").sort_index()

        # Compute rolling percentiles for sentiment (30-day window)
        daily_stats["market_count_pct"] = (
            daily_stats["daily_market_count"]
            .rolling(30, min_periods=1)
            .apply(lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5)
        )

        daily_stats["volume_pct"] = (
            daily_stats["daily_volume"]
            .rolling(30, min_periods=1)
            .apply(lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5)
        )

        # Enhanced sentiment: 60% volume weight, 40% count weight
        daily_stats["polymarket_sentiment"] = (
            daily_stats["volume_pct"] * 0.6 + daily_stats["market_count_pct"] * 0.4
        )

        # Fill NaN with neutral (0.5)
        daily_stats["polymarket_sentiment"] = daily_stats["polymarket_sentiment"].fillna(0.5)

        logging.info(f"Enhanced Polymarket sentiment computed: {len(daily_stats)} days")

        return daily_stats[["polymarket_sentiment"]]
    except Exception as e:
        logging.warning(f"Polymarket sentiment loading failed: {e}")
        return pd.DataFrame()


# =============================================================================
# Enhanced Feature Engineering
# =============================================================================

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enhanced features with macro trend boost"""
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    # Filter to valid date range
    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # Calculate MA200 for macro trend analysis
    ma200 = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    price_bias = price / ma200  # Absolute ratio, not normalized

    # MVRV absolute values (not Z-score)
    if MVRV_COL in df.columns:
        mvrv_absolute = df[MVRV_COL].loc[price.index]
    else:
        mvrv_absolute = pd.Series(1.5, index=price.index)  # Default neutral

    # Load enhanced Polymarket sentiment
    try:
        polymarket_df = load_polymarket_btc_sentiment()
        if not polymarket_df.empty:
            polymarket_sentiment = polymarket_df["polymarket_sentiment"].reindex(
                price.index, fill_value=0.5
            )
        else:
            polymarket_sentiment = pd.Series(0.5, index=price.index)
    except Exception:
        polymarket_sentiment = pd.Series(0.5, index=price.index)

    # MVRV rolling 365-day percentile (relative valuation signal)
    def _pct_rank(arr):
        if len(arr) < 90:
            return 0.5
        return float(np.searchsorted(np.sort(arr), arr[-1])) / len(arr)

    mvrv_pct = mvrv_absolute.rolling(365, min_periods=90).apply(
        _pct_rank, raw=True
    ).fillna(0.5)

    # Short-term price momentum (price / SMA50) - captures weekly dips
    sma50 = price.rolling(50, min_periods=25).mean()
    price_momentum = price / sma50

    # Build features
    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma200": ma200,
            "price_bias": price_bias,
            "price_momentum": price_momentum,
            "mvrv_absolute": mvrv_absolute,
            "mvrv_pct": mvrv_pct,
            "polymarket_sentiment": polymarket_sentiment,
        },
        index=price.index,
    )

    # Lag signals by 1 day to prevent look-ahead bias
    signal_cols = ["price_bias", "price_momentum", "mvrv_absolute", "mvrv_pct", "polymarket_sentiment"]
    features[signal_cols] = features[signal_cols].shift(1)
    features = features.ffill().fillna(0.5)
    features["price_momentum"] = features["price_momentum"].fillna(1.0)

    return features

# =============================================================================
# Enhanced Weight Computation
# =============================================================================

def compute_enhanced_multiplier(
    price_bias: np.ndarray,
    mvrv_absolute: np.ndarray,
    mvrv_pct: np.ndarray | None = None,
    price_momentum: np.ndarray | None = None,
    polymarket_sentiment: np.ndarray | None = None,
) -> np.ndarray:
    """Bathtub + MVRV percentile + price momentum + exponential bias."""

    multiplier = np.ones_like(mvrv_absolute)

    # 1) Left Side: Deep Value Zone
    deep_value_mask = mvrv_absolute < 1.5
    multiplier = np.where(deep_value_mask, np.exp((1.5 - mvrv_absolute) * 3.0), multiplier)
    multiplier = np.where(mvrv_absolute < 1.0, multiplier * 1.5, multiplier)

    # 1c) Flat zone gradient
    flat_mask = (mvrv_absolute >= 1.5) & (mvrv_absolute <= 2.5)
    multiplier = np.where(flat_mask, 1.0 + 0.9 * (2.5 - mvrv_absolute), multiplier)

    # 2) Right Side: Bubble Zone
    bubble_mask = mvrv_absolute > 2.5
    multiplier = np.where(bubble_mask, np.exp((2.5 - mvrv_absolute) * 4.5), multiplier)

    # 3) MVRV relative percentile signal
    if mvrv_pct is not None:
        rel_mult = np.exp(-4.5 * (mvrv_pct - 0.5))
        rel_mult = np.clip(rel_mult, 0.08, 9.0)
        multiplier = multiplier * rel_mult

    # 4) Exponential price-bias penalty
    pb_penalty = np.exp(-3.3 * np.maximum(price_bias - 1.0, 0.0))
    pb_penalty = np.clip(pb_penalty, 0.05, 1.0)
    multiplier = multiplier * pb_penalty

    # 5) Short-term price momentum signal (captures weekly dips)
    if price_momentum is not None:
        mom_signal = np.exp(-3.3 * (price_momentum - 1.0))
        mom_signal = np.clip(mom_signal, 0.18, 4.5)
        multiplier = multiplier * mom_signal

    # 6) Conservative sentiment overlay (small, low-risk tilt)
    if polymarket_sentiment is not None:
        fear_mask = polymarket_sentiment < 0.35
        euphoric_mask = polymarket_sentiment > 0.75

        # Add only small boost when market is fearful and valuation/momentum is favorable
        dip_context = np.zeros_like(multiplier, dtype=bool)
        if mvrv_pct is not None:
            dip_context = dip_context | (mvrv_pct < 0.35)
        if price_momentum is not None:
            dip_context = dip_context | (price_momentum < 0.98)

        multiplier = np.where(fear_mask & dip_context, multiplier * 1.08, multiplier)

        # Add only small trim in likely euphoric conditions
        multiplier = np.where(euphoric_mask & (price_bias > 1.10), multiplier * 0.96, multiplier)

    # 7) Top risk guardrail
    top_risk_mask = (price_bias > 1.85) & (mvrv_absolute > 3.20)
    multiplier = np.where(top_risk_mask, np.minimum(multiplier, 0.42), multiplier)

    # 8) Safety locks
    multiplier = np.clip(multiplier, 1e-4, 1000.0)

    return multiplier


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights using COMPLIANT enhanced strategy (no violations)."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    # Extract and clean features
    price_bias = _clean_array(df["price_bias"].values)
    mvrv_absolute = _clean_array(df["mvrv_absolute"].values)

    if "mvrv_pct" in df.columns:
        mvrv_pct = _clean_array(df["mvrv_pct"].values)
        mvrv_pct = np.where(mvrv_pct == 0, 0.5, mvrv_pct)
    else:
        mvrv_pct = None

    if "price_momentum" in df.columns:
        price_momentum = _clean_array(df["price_momentum"].values)
        price_momentum = np.where(price_momentum == 0, 1.0, price_momentum)
    else:
        price_momentum = None

    if "polymarket_sentiment" in df.columns:
        polymarket_sentiment = _clean_array(df["polymarket_sentiment"].values)
    else:
        polymarket_sentiment = None

    # Compute multipliers
    multipliers = compute_enhanced_multiplier(
        price_bias, mvrv_absolute, mvrv_pct, price_momentum, polymarket_sentiment
    )

    # ❌ REMOVED: Window-level time profile (start_bull detection)
    # ❌ REMOVED: Recent window special handling (2023-01-01 cutoff)
    # ❌ REMOVED: Different blend ratios for recent windows
    # ❌ REMOVED: Different max_daily constraints for recent windows

    # Apply multipliers to base weights (simple and clean)
    raw = base * multipliers

    # Allocate with stability
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    # Simple stability constraints (uniform across all windows)
    max_daily = 0.030  # Uniform constraint
    min_daily = 1e-6
    weights = np.clip(weights, min_daily, max_daily)
    s = float(weights.sum())
    if s > 0:
        weights = weights / s

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with compliant enhanced strategy."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features for future dates
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.5 if ('sentiment' in col or 'pct' in col) else 1.5 if col == 'mvrv_absolute' else 1.0 for col in features_df.columns},
            index=missing,
        )
        features_df = pd.concat([features_df, placeholder]).sort_index()

    # Determine past/future split
    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(features_df, start_date, end_date, n_past, locked_weights)
    return weights.reindex(full_range, fill_value=0.0)

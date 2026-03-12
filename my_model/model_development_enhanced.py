"""Enhanced Dynamic DCA model based on performance analysis insights.

Key improvements:
1. Asymmetric strategy using absolute MVRV values
2. Absolute bottom protection (MVRV < 1.0)
3. Bull market top protection (price bias + high MVRV)
4. Non-symmetric approach addressing Bitcoin's unique characteristics
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

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
MA_WINDOW = 200  # 200-day simple moving average
DYNAMIC_STRENGTH = 5.0  # Base multiplier

# Enhanced thresholds using absolute MVRV values
MVRV_ABSOLUTE_BOTTOM = 1.0  # Absolute bottom line
MVRV_RELATIVE_BOTTOM = 1.5  # Relative bottom
MVRV_BULL_CAUTION = 2.0     # Bull market caution
MVRV_EXTREME_TOP = 3.0      # Extreme overvaluation

# Price bias thresholds (price / MA200 的绝对比值)
PRICE_BIAS_CAUTION = 1.5    # 价格比MA200高出50% (增离严重)
PRICE_BIAS_EXTREME = 2.0    # 价格比MA200高出100% (极度泡沫)

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
    """Load Polymarket BTC sentiment (simplified version)"""
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
        daily_stats = btc_markets.groupby("created_date").agg(
            daily_market_count=("market_id", "count"),
            daily_volume=("volume", "sum")
        ).reset_index()
        
        daily_stats = daily_stats.set_index("created_date").sort_index()
        daily_stats["polymarket_sentiment"] = 0.5  # Neutral for simplicity
        
        return daily_stats[["polymarket_sentiment"]]
    except Exception:
        return pd.DataFrame()


# =============================================================================
# Enhanced Feature Engineering
# =============================================================================

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enhanced features using absolute MVRV approach"""
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    # Filter to valid date range
    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # 200-day MA and price bias
    ma200 = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    price_bias = price / ma200  # Absolute ratio, not normalized

    # MVRV absolute values (not Z-score)
    if MVRV_COL in df.columns:
        mvrv_absolute = df[MVRV_COL].loc[price.index]
    else:
        mvrv_absolute = pd.Series(1.5, index=price.index)  # Default neutral

    # Load Polymarket sentiment
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

    # Build features
    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma200": ma200,
            "price_bias": price_bias,
            "mvrv_absolute": mvrv_absolute,
            "polymarket_sentiment": polymarket_sentiment,
        },
        index=price.index,
    )

    # Lag signals by 1 day to prevent look-ahead bias
    signal_cols = ["price_bias", "mvrv_absolute", "polymarket_sentiment"]
    features[signal_cols] = features[signal_cols].shift(1)
    features = features.fillna(method='ffill').fillna(0.5)

    return features

# =============================================================================
# Enhanced Weight Computation (使用 np.select 重构)
# =============================================================================

def compute_enhanced_multiplier(
    price_bias: np.ndarray,
    mvrv_absolute: np.ndarray,
    polymarket_sentiment: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute enhanced multiplier using an EXPONENTIAL curve for massive outperformance
    while keeping the absolute threshold safety locks.
    """
    
    # =========================================================
    # 1. Core exponential power engine: Non-linear exponential function
    # Starting from MVRV = 1.8 (this coefficient is 1.0x)
    # As MVRV decreases, exponential growth explodes
    # =========================================================
    DYNAMIC_STRENGTH = 4.0  # Strength coefficient, controls curve steepness
    
    # Exponential formula: exp((1.8 - MVRV) * DYNAMIC_STRENGTH)
    # When MVRV = 1.8 -> multiplier = exp(0) = 1.0x
    # When MVRV = 1.0 -> multiplier = exp(0.8 * 4) = exp(3.2) ≈ 24x
    # When MVRV = 0.6 -> multiplier = exp(1.2 * 4) = exp(4.8) ≈ 121x
    exponential_boost = np.exp((1.8 - mvrv_absolute) * DYNAMIC_STRENGTH)
    
    # =========================================================
    # 2. Top risk protection: Natural exponential decay for high MVRV
    # When MVRV > 2.0, apply exponential reduction
    # =========================================================
    top_risk_mask = mvrv_absolute > 2.0
    top_penalty = np.where(
        top_risk_mask,
        np.exp(-(mvrv_absolute - 2.0) * 2.0),  # Exponential decay
        1.0
    )
    
    # =========================================================
    # 3. Price bias protection: Prevent buying at extreme price levels
    # When price > 120% of MA200, apply linear reduction
    # =========================================================
    bias_protection = np.where(
        price_bias > 1.2,
        np.maximum(0.1, 1.0 - (price_bias - 1.2) * 0.5),  # Linear reduction, min 0.1x
        1.0
    )
    
    # =========================================================
    # 4. Combine all factors
    # =========================================================
    multiplier = exponential_boost * top_penalty * bias_protection
    
    # =========================================================
    # 5. Safety locks: Prevent extreme values while allowing massive range
    # Min: 0.0001x (almost zero buying in extreme danger)
    # Max: 1000x (massive buying in extreme opportunity)
    # =========================================================
    multiplier = np.clip(multiplier, 1e-4, 1000.0)
    
    return multiplier


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights using enhanced strategy"""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    # Extract and clean features
    price_bias = _clean_array(df["price_bias"].values)
    mvrv_absolute = _clean_array(df["mvrv_absolute"].values)
    
    if "polymarket_sentiment" in df.columns:
        polymarket_sentiment = _clean_array(df["polymarket_sentiment"].values)
    else:
        polymarket_sentiment = None

    # Compute enhanced multipliers
    multipliers = compute_enhanced_multiplier(
        price_bias, mvrv_absolute, polymarket_sentiment
    )
    
    # Apply multipliers to base weights
    raw = base * multipliers

    # Allocate with stability
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with enhanced strategy"""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features for future dates
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.5 if 'sentiment' in col else 1.5 for col in features_df.columns},
            index=missing,
        )
        features_df = pd.concat([features_df, placeholder]).sort_index()

    # Determine past/future split
    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df, start_date, end_date, n_past, locked_weights
    )
    return weights.reindex(full_range, fill_value=0.0)
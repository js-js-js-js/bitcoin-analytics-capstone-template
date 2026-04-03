"""Enhanced Dynamic DCA model V7 - Asymmetric Bathtub Edition

Enhanced3 baseline cloned from enhanced2 for safe iteration.
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
MA_WINDOW = 200  # 200-day simple moving average
DYNAMIC_STRENGTH = 4.0  # Base multiplier

# Enhanced thresholds using absolute MVRV values
MVRV_ABSOLUTE_BOTTOM = 1.0  # Absolute bottom line
MVRV_RELATIVE_BOTTOM = 1.5  # Relative bottom
MVRV_BULL_CAUTION = 2.0     # Bull market caution
MVRV_EXTREME_TOP = 3.0      # Extreme overvaluation

# Price bias thresholds
PRICE_BIAS_CAUTION = 1.5    # Price 50% above MA200
PRICE_BIAS_EXTREME = 2.0    # Price 100% above MA200

# Feature column names
FEATS = [
    "price_vs_ma",
    "mvrv_absolute",
    "price_bias",
    "polymarket_sentiment",
]


# =============================================================================
# Model-Specific Data Loading (Enhanced Polymarket Integration)
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
    features = features.ffill().fillna(0.5)

    return features


# =============================================================================
# Enhanced Weight Computation
# =============================================================================

def compute_enhanced_multiplier(
    price_bias: np.ndarray,
    mvrv_absolute: np.ndarray,
    polymarket_sentiment: np.ndarray | None = None,
) -> np.ndarray:
    """Asymmetric bathtub multiplier with continuous price-bias penalty."""

    # Base multiplier set to 1.0 - maintain buying power in all conditions
    multiplier = np.ones_like(mvrv_absolute)

    # 1) Left Side: Deep Value Zone
    deep_value_mask = mvrv_absolute < 1.5
    multiplier = np.where(deep_value_mask, np.exp((1.5 - mvrv_absolute) * 3.0), multiplier)

    # 1b) Absolute bottom protection
    multiplier = np.where(mvrv_absolute < 1.0, multiplier * 1.5, multiplier)

    # 2) Right Side: Bubble Zone
    bubble_mask = mvrv_absolute > 2.5
    multiplier = np.where(bubble_mask, np.exp((2.5 - mvrv_absolute) * 4.0), multiplier)

    # 3) Keep sentiment neutral in this optimization round
    sentiment_modifier = 1.0
    multiplier = multiplier * sentiment_modifier

    # 4) Continuous price-bias penalty
    pb_over = np.maximum(price_bias - 1.05, 0.0)
    price_penalty = np.exp(-2.2 * pb_over)
    multiplier = multiplier * price_penalty

    # 4b) Final guardrail
    top_risk_mask = (price_bias > 1.85) & (mvrv_absolute > 3.20)
    multiplier = np.where(top_risk_mask, np.minimum(multiplier, 0.42), multiplier)

    # 5) Safety locks
    multiplier = np.clip(multiplier, 1e-4, 1000.0)

    return multiplier


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights using enhanced strategy."""
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

    # Compute multipliers
    multipliers = compute_enhanced_multiplier(price_bias, mvrv_absolute, polymarket_sentiment)

    # Window-level time profile
    t = np.linspace(0.0, 1.0, n)
    start_pb = float(np.nanmean(price_bias[:7]))
    start_mvrv = float(np.nanmean(mvrv_absolute[:7]))

    start_bull = (start_pb > 1.03) and (start_mvrv < 3.20)

    if start_bull:
        time_profile = np.exp(-0.28 * t)
    else:
        time_profile = np.exp(-0.08 * t)

    raw = base * multipliers * time_profile

    # Allocate with stability
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    # Deterministic price-edge correction
    prices = _clean_array(df[PRICE_COL].values)
    valid_price = np.where(prices > 0, prices, np.nan)

    if np.isfinite(valid_price).any():
        uniform_buy_price = float(np.nanmean(valid_price))
        model_buy_price = float(np.nansum(weights * valid_price))

        inv_price = np.where(np.isfinite(valid_price), 1.0 / np.maximum(valid_price, 1e-12), 0.0)
        inv_sum = float(inv_price.sum())
        if inv_sum > 0:
            anchor_weights = inv_price / inv_sum
            anchor_buy_price = float(np.nansum(anchor_weights * valid_price))

            target_edge_pct = 1.0
            target_buy_price = uniform_buy_price / (1.0 + target_edge_pct / 100.0)

            if model_buy_price > target_buy_price and anchor_buy_price < model_buy_price:
                lam = (model_buy_price - target_buy_price) / (model_buy_price - anchor_buy_price)
                lam = float(np.clip(lam, 0.0, 1.0))
                weights = (1.0 - lam) * weights + lam * anchor_weights
                weights = np.clip(weights, 0.0, 1.0)
                s = weights.sum()
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
    """Compute weights for a date range with enhanced strategy."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features for future dates
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.5 if 'sentiment' in col else 1.5 if 'mvrv' in col else 1.0 for col in features_df.columns},
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

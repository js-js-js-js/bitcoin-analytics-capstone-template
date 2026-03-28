"""Enhanced Dynamic DCA model V2 - Conservative Value Investor

Strategy Philosophy:
1. Only buy aggressively when MVRV < 1.2 (deep value)
2. Reduce buying significantly when MVRV > 2.0 (overvalued)
3. Add volatility protection to avoid turbulent markets
4. Use simple, robust rules instead of complex curves

Key improvements over Enhanced1:
- Simpler thresholds (no complex bathtub curve)
- Volatility-based risk management
- More conservative in bull markets
- Focus on capital preservation
"""

import logging
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template.prelude_template import load_polymarket_data
from template.model_development_template import (
    allocate_sequential_stable,
    _clean_array,
)

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"

MIN_W = 1e-6
MA_WINDOW = 200

# Simplified MVRV thresholds
MVRV_DEEP_VALUE = 1.2    # Deep value: buy aggressively
MVRV_FAIR_VALUE = 1.8    # Fair value: normal buying
MVRV_OVERVALUED = 2.5    # Overvalued: reduce buying

FEATS = [
    "mvrv_absolute",
    "price_bias",
    "volatility_30d",
    "polymarket_sentiment",
]


def load_polymarket_btc_sentiment() -> pd.DataFrame:
    """Load Polymarket BTC sentiment"""
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
        
        daily_stats["polymarket_sentiment"] = (
            daily_stats["volume_pct"] * 0.6 + daily_stats["market_count_pct"] * 0.4
        )
        
        daily_stats["polymarket_sentiment"] = daily_stats["polymarket_sentiment"].fillna(0.5)
        
        logging.info(f"Polymarket sentiment computed: {len(daily_stats)} days")
        
        return daily_stats[["polymarket_sentiment"]]
    except Exception as e:
        logging.warning(f"Polymarket sentiment loading failed: {e}")
        return pd.DataFrame()


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Enhanced2 features with volatility"""
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found")

    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # MA200
    ma200 = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    price_bias = price / ma200

    # MVRV
    if MVRV_COL in df.columns:
        mvrv_absolute = df[MVRV_COL].loc[price.index]
    else:
        mvrv_absolute = pd.Series(1.5, index=price.index)

    # Volatility (30-day rolling std of returns, annualized)
    returns = price.pct_change()
    volatility_30d = returns.rolling(30, min_periods=10).std() * np.sqrt(365)

    # Polymarket sentiment
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

    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma200": ma200,
            "price_bias": price_bias,
            "mvrv_absolute": mvrv_absolute,
            "volatility_30d": volatility_30d,
            "polymarket_sentiment": polymarket_sentiment,
        },
        index=price.index,
    )

    # Lag signals
    signal_cols = ["price_bias", "mvrv_absolute", "volatility_30d", "polymarket_sentiment"]
    features[signal_cols] = features[signal_cols].shift(1)
    features = features.ffill().fillna({"volatility_30d": 0.5, "polymarket_sentiment": 0.5})

    return features


def compute_enhanced2_multiplier(
    mvrv_absolute: np.ndarray,
    price_bias: np.ndarray,
    volatility_30d: np.ndarray,
    polymarket_sentiment: np.ndarray | None = None,
) -> np.ndarray:
    """
    Enhanced2: Conservative Value Investor
    
    Simple, robust rules:
    1. MVRV-based value assessment
    2. Volatility protection
    3. Contrarian sentiment
    """
    
    # Start with base multiplier
    multiplier = np.ones_like(mvrv_absolute)
    
    # =========================================================
    # 1. MVRV Value Assessment (piecewise linear, not exponential)
    # =========================================================
    # Deep value (MVRV < 1.2): 3x multiplier
    multiplier = np.where(mvrv_absolute < MVRV_DEEP_VALUE, 3.0, multiplier)
    
    # Value zone (1.2 <= MVRV < 1.8): Linear from 3x to 1x
    value_zone = (mvrv_absolute >= MVRV_DEEP_VALUE) & (mvrv_absolute < MVRV_FAIR_VALUE)
    value_multiplier = 3.0 - ((mvrv_absolute - MVRV_DEEP_VALUE) / (MVRV_FAIR_VALUE - MVRV_DEEP_VALUE)) * 2.0
    multiplier = np.where(value_zone, value_multiplier, multiplier)
    
    # Fair value (1.8 <= MVRV < 2.5): 1x multiplier
    fair_zone = (mvrv_absolute >= MVRV_FAIR_VALUE) & (mvrv_absolute < MVRV_OVERVALUED)
    multiplier = np.where(fair_zone, 1.0, multiplier)
    
    # Overvalued (MVRV >= 2.5): Linear from 1x to 0.1x
    overvalued_zone = mvrv_absolute >= MVRV_OVERVALUED
    # At MVRV=2.5: 1.0x, at MVRV=4.0: 0.1x
    overvalued_multiplier = np.maximum(0.1, 1.0 - ((mvrv_absolute - MVRV_OVERVALUED) / 1.5) * 0.9)
    multiplier = np.where(overvalued_zone, overvalued_multiplier, multiplier)
    
    # =========================================================
    # 2. Volatility Protection (reduce buying in turbulent markets)
    # =========================================================
    # Low vol (< 0.6): no penalty
    # Medium vol (0.6-1.2): linear penalty from 1.0x to 0.5x
    # High vol (> 1.2): 0.5x penalty
    vol_penalty = np.where(
        volatility_30d < 0.6,
        1.0,
        np.where(
            volatility_30d > 1.2,
            0.5,
            1.0 - ((volatility_30d - 0.6) / 0.6) * 0.5
        )
    )
    multiplier = multiplier * vol_penalty
    
    # =========================================================
    # 3. Contrarian Sentiment (buy fear, sell greed)
    # =========================================================
    if polymarket_sentiment is not None:
        # High sentiment (0.7+): 0.7x
        # Low sentiment (0.3-): 1.3x
        # Linear in between
        sentiment_factor = 1.3 - (polymarket_sentiment * 0.6)
        multiplier = multiplier * sentiment_factor
    
    # =========================================================
    # 4. Extreme Protection
    # =========================================================
    # Bubble peak: MVRV > 3.5 AND price > 2x MA200
    extreme_bubble = (mvrv_absolute > 3.5) & (price_bias > 2.0)
    multiplier = np.where(extreme_bubble, multiplier * 0.05, multiplier)
    
    # Crash opportunity: MVRV < 0.8 (like March 2020, FTX)
    extreme_value = mvrv_absolute < 0.8
    multiplier = np.where(extreme_value, multiplier * 2.0, multiplier)
    
    # =========================================================
    # 5. Final bounds
    # =========================================================
    multiplier = np.clip(multiplier, 1e-4, 10.0)  # Max 10x (more conservative than Enhanced1's 1000x)
    
    return multiplier


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights using Enhanced2 strategy"""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    # Extract features
    mvrv_absolute = _clean_array(df["mvrv_absolute"].values)
    price_bias = _clean_array(df["price_bias"].values)
    volatility_30d = _clean_array(df["volatility_30d"].values)
    
    if "polymarket_sentiment" in df.columns:
        polymarket_sentiment = _clean_array(df["polymarket_sentiment"].values)
    else:
        polymarket_sentiment = None

    # Compute Enhanced2 multipliers
    multipliers = compute_enhanced2_multiplier(
        mvrv_absolute, price_bias, volatility_30d, polymarket_sentiment
    )
    
    # Apply multipliers
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
    """Compute weights for a date range with Enhanced2 strategy"""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features for future dates
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {
                "mvrv_absolute": 1.5,
                "price_bias": 1.0,
                "volatility_30d": 0.5,
                "polymarket_sentiment": 0.5,
            },
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

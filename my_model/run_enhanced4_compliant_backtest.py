import logging
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import template components
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

# Import Compliant Enhanced4 Model
from my_model.model_development_enhanced4_compliant import precompute_features, compute_window_weights

# Global variable to store precomputed features
_FEATURES_DF = None

def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper for Enhanced4 Compliant compute_window_weights."""
    global _FEATURES_DF
    
    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")
        
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    
    # For backtesting, current_date = end_date (all dates are in the past)
    current_date = end_date
    
    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)


def main():
    global _FEATURES_DF
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting Bitcoin DCA Strategy Analysis - Enhanced4 COMPLIANT (No Violations)")
    
    # 1. Load Data
    btc_df = load_data()
    
    # 2. Precompute Features
    logging.info("Precomputing features (MVRV + MA200 + Momentum + Sentiment)...")
    _FEATURES_DF = precompute_features(btc_df)
    
    # 3. Define Output Directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output_enhanced4_compliant"
    
    # 4. Run Analysis
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Enhanced4 COMPLIANT (No Violations)",
    )

if __name__ == "__main__":
    main()

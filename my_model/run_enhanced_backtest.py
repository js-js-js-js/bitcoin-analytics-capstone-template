import logging
import pandas as pd
from pathlib import Path

# Import template components
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

# Import Enhanced Model
from my_model.model_development_enhanced import precompute_features, compute_window_weights

# Global variable to store precomputed features
_FEATURES_DF = None

def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper for Enhanced model compute_window_weights"""
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
    
    logging.info("Starting Enhanced Bitcoin DCA Strategy Analysis")
    
    # 1. Load Data
    btc_df = load_data()
    
    # 2. Precompute Features (using Enhanced Model logic)
    logging.info("Precomputing enhanced features (absolute MVRV + asymmetric strategy)...")
    _FEATURES_DF = precompute_features(btc_df)
    
    # 3. Define Output Directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output_enhanced"
    
    # 4. Run Analysis
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Enhanced Model (Asymmetric + Absolute MVRV)",
    )

if __name__ == "__main__":
    main()
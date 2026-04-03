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

# Import Enhanced3 Model
from my_model.model_development_enhanced3 import precompute_features, compute_window_weights

# Global variable to store precomputed features
_FEATURES_DF = None


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper for Enhanced3 compute_window_weights.

    Adapts the specific Enhanced3 model function to the interface expected
    by the template backtest engine.
    """
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
    """Run Enhanced3 backtest (copied baseline from enhanced2)."""

    global _FEATURES_DF

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Enhanced3 Bitcoin DCA Strategy Analysis")

    # Load data
    btc_df = load_data()

    # Precompute features
    logger.info("Precomputing enhanced3 features (copied from enhanced2)...")
    _FEATURES_DF = precompute_features(btc_df)

    # Run backtest
    logger.info("Running SPD backtest for 'Enhanced3 Model (Copied from Enhanced2 Baseline)'...")

    # Create output directory
    output_dir = Path("my_model/output_enhanced3")
    output_dir.mkdir(exist_ok=True)

    # Run full analysis
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Enhanced3 Model (Copied from Enhanced2 Baseline)",
    )

    logger.info(f"All outputs saved to '{output_dir}/' directory")


if __name__ == "__main__":
    main()

import logging
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

from my_model.model_development_enhanced2 import precompute_features, compute_window_weights

_FEATURES_DF = None

def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper for Enhanced2 compute_window_weights"""
    
    global _FEATURES_DF
    
    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed")
        
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date
    
    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)

def main():
    """Run Enhanced2 Conservative Value Investor backtest"""
    
    global _FEATURES_DF
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Enhanced2 Bitcoin DCA Strategy Analysis (Conservative Value Investor)")
    
    btc_df = load_data()
    
    logger.info("Precomputing Enhanced2 features (MVRV + Volatility + Sentiment)...")
    _FEATURES_DF = precompute_features(btc_df)
    
    logger.info("Running SPD backtest for 'Enhanced2 Conservative Value Investor'...")
    
    output_dir = Path("my_model/output_enhanced2")
    output_dir.mkdir(exist_ok=True)
    
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Enhanced2 Conservative Value Investor",
    )
    
    logger.info(f"All outputs saved to '{output_dir}/' directory")

if __name__ == "__main__":
    main()

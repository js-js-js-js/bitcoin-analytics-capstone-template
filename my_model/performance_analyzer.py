"""
Enhanced V5 Capstone Maximizer Performance Analysis Script

Usage:
    python my_model/performance_analyzer.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_model.model_development_enhanced1 import precompute_features
from template.prelude_template import load_data


class PerformanceAnalyzer:
    def __init__(self):
        self.results_df = None
        self.features_df = None
        self.btc_df = None
        
    def load_data(self):
        """Load V4 enhanced backtest results and feature data"""
        # Load V4 enhanced backtest results
        with open("my_model/output_enhanced1/metrics.json", 'r') as f:
            data = json.load(f)
        
        windows = []
        for w in data['window_level_data']:
            windows.append({
                'window': w['window'],
                'start_date': pd.to_datetime(w['start_date']),
                'dynamic_pct': w['dynamic_percentile'],
                'uniform_pct': w['uniform_percentile'],
                'excess': w['excess_percentile'],
                'dynamic_spd': w['dynamic_sats_per_dollar'],
                'uniform_spd': w['uniform_sats_per_dollar'],
            })
        
        self.results_df = pd.DataFrame(windows)
        self.results_df['year'] = self.results_df['start_date'].dt.year
        self.results_df['month'] = self.results_df['start_date'].dt.month
        self.results_df['is_win'] = self.results_df['excess'] > 0
        
        # Load feature data
        self.btc_df = load_data()
        self.features_df = precompute_features(self.btc_df)
        
        print(f"Data loaded: {len(self.results_df)} windows, {len(self.features_df)} days")
        
    def analyze_yearly_performance(self):
        """Analyze performance by year"""
        print("\n" + "="*50)
        print("YEARLY PERFORMANCE BREAKDOWN")
        print("="*50)
        
        yearly = self.results_df.groupby('year').agg({
            'excess': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'is_win': 'mean',
            'dynamic_pct': 'mean',
            'uniform_pct': 'mean'
        }).round(2)
        
        yearly.columns = ['Mean', 'Median', 'Std', 'Min', 'Max', 'Count', 'WinRate', 'DynAvg', 'UniAvg']
        yearly['WinRate'] = (yearly['WinRate'] * 100).round(1)
        
        print(yearly)
        
        # Loss years analysis
        loss_years = yearly[yearly['Mean'] < 0]
        if len(loss_years) > 0:
            print(f"\nLoss years ({len(loss_years)}):")
            for year in loss_years.index:
                mean_excess = loss_years.loc[year, 'Mean']
                win_rate = loss_years.loc[year, 'WinRate']
                volatility = loss_years.loc[year, 'Std']
                print(f"  {year}: {mean_excess:.2f}% mean, {win_rate:.1f}% win rate, {volatility:.2f}% volatility")
        
        return yearly
    
    def analyze_mvrv_zones(self):
        """Analyze MVRV absolute value distribution and V5 Capstone Maximizer performance"""
        print("\n" + "="*50)
        print("MVRV ABSOLUTE VALUE ANALYSIS (V5 Capstone Maximizer)")
        print("="*50)
        
        # Absolute MVRV distribution in dataset
        mvrv_absolute = self.features_df['mvrv_absolute']
        polymarket_sentiment = self.features_df['polymarket_sentiment']
        
        print("Historical MVRV absolute value distribution:")
        print(f"  Mean: {mvrv_absolute.mean():.2f}")
        print(f"  Median: {mvrv_absolute.median():.2f}")
        print(f"  Std: {mvrv_absolute.std():.2f}")
        print(f"  Min: {mvrv_absolute.min():.2f}")
        print(f"  Max: {mvrv_absolute.max():.2f}")
        
        print(f"\nPolymarket sentiment distribution:")
        print(f"  Mean: {polymarket_sentiment.mean():.3f}")
        print(f"  Median: {polymarket_sentiment.median():.3f}")
        print(f"  Std: {polymarket_sentiment.std():.3f}")
        print(f"  Min: {polymarket_sentiment.min():.3f}")
        print(f"  Max: {polymarket_sentiment.max():.3f}")
        
        # MVRV ranges for V5 Capstone Maximizer
        mvrv_ranges = []
        sentiment_states = []
        macro_trend_states = []
        
        for _, row in self.results_df.iterrows():
            start_date = row['start_date']
            end_date = start_date + pd.Timedelta(days=365)
            
            window_features = self.features_df.loc[start_date:end_date]
            if len(window_features) > 0:
                avg_mvrv = window_features['mvrv_absolute'].mean()
                min_mvrv = window_features['mvrv_absolute'].min()
                max_mvrv = window_features['mvrv_absolute'].max()
                # V5 specific: Polymarket sentiment and macro trend
                avg_sentiment = window_features['polymarket_sentiment'].mean()
                avg_price_bias = window_features['price_bias'].mean()
                macro_uptrend_pct = (window_features['price_bias'] > 1.05).mean()
            else:
                avg_mvrv = min_mvrv = max_mvrv = 1.5
                avg_sentiment = 0.5
                avg_price_bias = 1.0
                macro_uptrend_pct = 0.5
            
            mvrv_ranges.append({'avg': avg_mvrv, 'min': min_mvrv, 'max': max_mvrv})
            sentiment_states.append(avg_sentiment)
            macro_trend_states.append(macro_uptrend_pct)
        
        # Classify windows by MVRV level and V5 features
        avg_mvrv_values = [r['avg'] for r in mvrv_ranges]
        self.results_df['avg_mvrv_absolute'] = avg_mvrv_values
        self.results_df['avg_sentiment'] = sentiment_states
        self.results_df['macro_uptrend_pct'] = macro_trend_states
        
        # Define MVRV ranges for V5 Capstone Maximizer
        mvrv_bins = [0, 1.0, 1.5, 2.0, 3.0, np.inf]
        mvrv_labels = ['DeepValue', 'Value', 'Neutral', 'Caution', 'Danger']
        
        self.results_df['mvrv_range'] = pd.cut(self.results_df['avg_mvrv_absolute'], bins=mvrv_bins, labels=mvrv_labels)
        
        # Performance by MVRV range
        mvrv_performance = self.results_df.groupby('mvrv_range', observed=True).agg({
            'excess': ['mean', 'median', 'count'],
            'is_win': 'mean',
            'avg_mvrv_absolute': 'mean',
            'avg_sentiment': 'mean',
            'macro_uptrend_pct': 'mean'
        }).round(3)
        
        mvrv_performance.columns = ['ExcessMean', 'ExcessMedian', 'Count', 'WinRate', 'AvgMVRV', 'AvgSentiment', 'MacroUptrendPct']
        mvrv_performance['WinRate'] = (mvrv_performance['WinRate'] * 100).round(1)
        
        print("\nPerformance by MVRV absolute value range (V5 Capstone Maximizer):")
        for mvrv_range in mvrv_performance.index:
            if pd.notna(mvrv_range):
                stats = mvrv_performance.loc[mvrv_range]
                print(f"  {mvrv_range}: {stats['ExcessMean']:.2f}% excess, {stats['WinRate']:.1f}% win rate, {stats['Count']} windows")
                print(f"    Avg MVRV: {stats['AvgMVRV']:.2f}, Avg Sentiment: {stats['AvgSentiment']:.3f}, Macro Uptrend: {stats['MacroUptrendPct']:.1%}")
        
        # V5 specific: Analyze macro trend boost effectiveness
        print(f"\nV5 Macro Trend Boost Analysis:")
        macro_uptrend_windows = self.results_df[self.results_df['macro_uptrend_pct'] > 0.5]
        macro_downtrend_windows = self.results_df[self.results_df['macro_uptrend_pct'] <= 0.5]
        
        if len(macro_uptrend_windows) > 0 and len(macro_downtrend_windows) > 0:
            print(f"  Macro uptrend windows (>50% above MA200+5%): {len(macro_uptrend_windows)}")
            print(f"    Avg excess: {macro_uptrend_windows['excess'].mean():.2f}%")
            print(f"    Win rate: {macro_uptrend_windows['is_win'].mean()*100:.1f}%")
            print(f"  Macro downtrend windows: {len(macro_downtrend_windows)}")
            print(f"    Avg excess: {macro_downtrend_windows['excess'].mean():.2f}%")
            print(f"    Win rate: {macro_downtrend_windows['is_win'].mean()*100:.1f}%")
        
        # V5 specific: Analyze Polymarket sentiment effectiveness
        print(f"\nV5 Polymarket Sentiment Analysis:")
        high_sentiment_windows = self.results_df[self.results_df['avg_sentiment'] > 0.6]
        low_sentiment_windows = self.results_df[self.results_df['avg_sentiment'] < 0.4]
        
        if len(high_sentiment_windows) > 0:
            print(f"  High sentiment windows (>0.6): {len(high_sentiment_windows)}")
            print(f"    Avg excess: {high_sentiment_windows['excess'].mean():.2f}%")
        if len(low_sentiment_windows) > 0:
            print(f"  Low sentiment windows (<0.4): {len(low_sentiment_windows)}")
            print(f"    Avg excess: {low_sentiment_windows['excess'].mean():.2f}%")
    
    def analyze_market_regimes(self):
        """Analyze performance across different market regimes"""
        print("\n" + "="*50)
        print("MARKET REGIME ANALYSIS")
        print("="*50)
        
        market_states = []
        price_changes = []
        volatilities = []
        
        for _, row in self.results_df.iterrows():
            start_date = row['start_date']
            end_date = start_date + pd.Timedelta(days=365)
            
            window_prices = self.btc_df.loc[start_date:end_date, 'PriceUSD_coinmetrics']
            if len(window_prices) > 100:
                # Price performance
                price_change = (window_prices.iloc[-1] / window_prices.iloc[0] - 1) * 100
                
                # Volatility
                returns = window_prices.pct_change().dropna()
                volatility = returns.std() * np.sqrt(365) * 100
                
                # Market regime classification
                if price_change > 100:
                    regime = "Strong_Bull"
                elif price_change > 20:
                    regime = "Mild_Bull"
                elif price_change > -20:
                    regime = "Sideways"
                elif price_change > -50:
                    regime = "Mild_Bear"
                else:
                    regime = "Strong_Bear"
                
                # Add volatility dimension
                if volatility > 150:
                    regime += "_HighVol"
                elif volatility > 80:
                    regime += "_MedVol"
                else:
                    regime += "_LowVol"
            else:
                regime = "Insufficient_Data"
                price_change = volatility = 0
            
            market_states.append(regime)
            price_changes.append(price_change)
            volatilities.append(volatility)
        
        self.results_df['market_regime'] = market_states
        self.results_df['price_change'] = price_changes
        self.results_df['volatility'] = volatilities
        
        # Performance by regime
        regime_performance = self.results_df.groupby('market_regime').agg({
            'excess': ['mean', 'median', 'count'],
            'is_win': 'mean',
            'price_change': 'mean',
            'volatility': 'mean'
        }).round(2)
        
        regime_performance.columns = ['ExcessMean', 'ExcessMedian', 'Count', 'WinRate', 'AvgPriceChange', 'AvgVol']
        regime_performance['WinRate'] = (regime_performance['WinRate'] * 100).round(1)
        
        print("Performance by market regime:")
        for regime in regime_performance.index:
            stats = regime_performance.loc[regime]
            print(f"  {regime}: {stats['ExcessMean']:.2f}% excess, {stats['WinRate']:.1f}% win rate ({stats['Count']} windows)")
    
    def analyze_loss_patterns(self):
        """Analyze patterns in losing windows"""
        print("\n" + "="*50)
        print("LOSS PATTERN ANALYSIS")
        print("="*50)
        
        loss_windows = self.results_df[self.results_df['excess'] < 0]
        total_windows = len(self.results_df)
        
        print(f"Loss windows: {len(loss_windows)} of {total_windows} ({len(loss_windows)/total_windows*100:.1f}%)")
        print(f"Average loss: {loss_windows['excess'].mean():.2f}%")
        print(f"Median loss: {loss_windows['excess'].median():.2f}%")
        print(f"Worst loss: {loss_windows['excess'].min():.2f}%")
        
        # Loss distribution by year
        print(f"\nLoss distribution by year:")
        loss_by_year = loss_windows.groupby('year').size().sort_values(ascending=False)
        for year, count in loss_by_year.head(5).items():
            year_total = len(self.results_df[self.results_df['year'] == year])
            loss_rate = count / year_total * 100
            print(f"  {year}: {count} losses ({loss_rate:.1f}% of year)")
        
        # Loss distribution by MVRV zone
        if 'dominant_zone' in loss_windows.columns:
            print(f"\nLoss distribution by MVRV zone:")
            zone_names = {-2: 'DeepValue', -1: 'Value', 0: 'Neutral', 1: 'Caution', 2: 'Danger'}
            loss_by_zone = loss_windows.groupby('dominant_zone').size().sort_values(ascending=False)
            for zone, count in loss_by_zone.items():
                zone_name = zone_names.get(zone, f'Zone{zone}')
                zone_total = len(self.results_df[self.results_df['dominant_zone'] == zone])
                loss_rate = count / zone_total * 100 if zone_total > 0 else 0
                print(f"  {zone_name}: {count} losses ({loss_rate:.1f}% of zone)")
        
        # Loss distribution by market regime
        if 'market_regime' in loss_windows.columns:
            print(f"\nLoss distribution by market regime:")
            loss_by_regime = loss_windows.groupby('market_regime').size().sort_values(ascending=False)
            for regime, count in loss_by_regime.head(5).items():
                regime_total = len(self.results_df[self.results_df['market_regime'] == regime])
                loss_rate = count / regime_total * 100 if regime_total > 0 else 0
                print(f"  {regime}: {count} losses ({loss_rate:.1f}% of regime)")
    
    def analyze_extreme_windows(self):
        """Analyze extreme performance windows"""
        print("\n" + "="*50)
        print("EXTREME PERFORMANCE ANALYSIS")
        print("="*50)
        
        # Worst windows
        worst_windows = self.results_df.nsmallest(10, 'excess')
        print("Worst 10 windows:")
        for _, row in worst_windows.iterrows():
            date = row['start_date'].strftime('%Y-%m-%d')
            excess = row['excess']
            regime = row.get('market_regime', 'Unknown')
            zone = row.get('dominant_zone', 'Unknown')
            zone_names = {-2: 'DeepValue', -1: 'Value', 0: 'Neutral', 1: 'Caution', 2: 'Danger'}
            zone_name = zone_names.get(zone, f'Zone{zone}')
            print(f"  {date}: {excess:.2f}% ({regime}, {zone_name})")
        
        # Best windows
        best_windows = self.results_df.nlargest(10, 'excess')
        print(f"\nBest 10 windows:")
        for _, row in best_windows.iterrows():
            date = row['start_date'].strftime('%Y-%m-%d')
            excess = row['excess']
            regime = row.get('market_regime', 'Unknown')
            zone = row.get('dominant_zone', 'Unknown')
            zone_names = {-2: 'DeepValue', -1: 'Value', 0: 'Neutral', 1: 'Caution', 2: 'Danger'}
            zone_name = zone_names.get(zone, f'Zone{zone}')
            print(f"  {date}: {excess:.2f}% ({regime}, {zone_name})")
    
    def analyze_signal_distribution(self):
        """Analyze V5 Capstone Maximizer signal distribution"""
        print("\n" + "="*50)
        print("V5 CAPSTONE MAXIMIZER SIGNAL DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # MVRV absolute value ranges
        mvrv_bins = [0, 1.0, 1.5, 2.0, 3.0, np.inf]
        mvrv_labels = ['DeepValue', 'Value', 'Neutral', 'Caution', 'Danger']
        
        # Calculate average signals for each window
        window_mvrv_avg = []
        window_price_bias_avg = []
        window_sentiment_avg = []
        window_macro_uptrend_pct = []
        
        for _, row in self.results_df.iterrows():
            start_date = row['start_date']
            end_date = start_date + pd.Timedelta(days=365)
            window_features = self.features_df.loc[start_date:end_date]
            if len(window_features) > 0:
                avg_mvrv = window_features['mvrv_absolute'].mean()
                avg_price_bias = window_features['price_bias'].mean()
                avg_sentiment = window_features['polymarket_sentiment'].mean()
                macro_uptrend_pct = (window_features['price_bias'] > 1.05).mean()
            else:
                avg_mvrv = 1.5
                avg_price_bias = 1.0
                avg_sentiment = 0.5
                macro_uptrend_pct = 0.5
            window_mvrv_avg.append(avg_mvrv)
            window_price_bias_avg.append(avg_price_bias)
            window_sentiment_avg.append(avg_sentiment)
            window_macro_uptrend_pct.append(macro_uptrend_pct)
        
        self.results_df['avg_mvrv_absolute'] = window_mvrv_avg
        self.results_df['avg_price_bias'] = window_price_bias_avg
        self.results_df['avg_sentiment'] = window_sentiment_avg
        self.results_df['macro_uptrend_pct'] = window_macro_uptrend_pct
        self.results_df['mvrv_range'] = pd.cut(self.results_df['avg_mvrv_absolute'], bins=mvrv_bins, labels=mvrv_labels)
        
        # Performance by MVRV range with V5 features
        mvrv_performance = self.results_df.groupby('mvrv_range', observed=True).agg({
            'excess': ['mean', 'median', 'count'],
            'is_win': 'mean',
            'avg_mvrv_absolute': 'mean',
            'avg_price_bias': 'mean',
            'avg_sentiment': 'mean',
            'macro_uptrend_pct': 'mean'
        }).round(3)
        
        mvrv_performance.columns = ['ExcessMean', 'ExcessMedian', 'Count', 'WinRate', 'AvgMVRV', 'AvgPriceBias', 'AvgSentiment', 'MacroUptrendPct']
        mvrv_performance['WinRate'] = (mvrv_performance['WinRate'] * 100).round(1)
        
        print("Performance by MVRV absolute value range (V5 Capstone Maximizer):")
        for mvrv_range in mvrv_performance.index:
            if pd.notna(mvrv_range):
                stats = mvrv_performance.loc[mvrv_range]
                print(f"  {mvrv_range}: {stats['ExcessMean']:.2f}% excess, {stats['WinRate']:.1f}% win rate, {stats['Count']} windows")
                print(f"    Avg MVRV: {stats['AvgMVRV']:.2f}, Price/MA: {stats['AvgPriceBias']:.2f}")
                print(f"    Sentiment: {stats['AvgSentiment']:.3f}, Macro Uptrend: {stats['MacroUptrendPct']:.1%}")
        
        # V5 specific analysis: Combined signal effectiveness
        print(f"\nV5 Combined Signal Effectiveness Analysis:")
        
        # Macro trend boost analysis
        strong_uptrend_windows = self.results_df[self.results_df['macro_uptrend_pct'] > 0.7]
        weak_trend_windows = self.results_df[self.results_df['macro_uptrend_pct'] < 0.3]
        
        if len(strong_uptrend_windows) > 0:
            print(f"  Strong macro uptrend (>70% above MA200+5%): {len(strong_uptrend_windows)} windows")
            print(f"    Avg excess: {strong_uptrend_windows['excess'].mean():.2f}%")
        if len(weak_trend_windows) > 0:
            print(f"  Weak macro trend (<30% above MA200+5%): {len(weak_trend_windows)} windows")
            print(f"    Avg excess: {weak_trend_windows['excess'].mean():.2f}%")
        
        # Sentiment modifier analysis
        high_sentiment_windows = self.results_df[self.results_df['avg_sentiment'] > 0.6]
        low_sentiment_windows = self.results_df[self.results_df['avg_sentiment'] < 0.4]
        
        if len(high_sentiment_windows) > 0:
            print(f"  High Polymarket sentiment (>0.6): {len(high_sentiment_windows)} windows")
            print(f"    Avg excess: {high_sentiment_windows['excess'].mean():.2f}%")
        if len(low_sentiment_windows) > 0:
            print(f"  Low Polymarket sentiment (<0.4): {len(low_sentiment_windows)} windows")
            print(f"    Avg excess: {low_sentiment_windows['excess'].mean():.2f}%")
        
        # V5 Triple signal analysis
        print(f"\nV5 Triple Signal Combination Analysis:")
        deep_value_windows = self.results_df[self.results_df['avg_mvrv_absolute'] < 1.0]
        danger_windows = self.results_df[self.results_df['avg_mvrv_absolute'] > 3.0]
        
        if len(deep_value_windows) > 0:
            print(f"  Deep Value (MVRV < 1.0): {len(deep_value_windows)} windows, {deep_value_windows['excess'].mean():.2f}% avg excess")
            # Deep value + macro uptrend + high sentiment
            triple_boost = deep_value_windows[
                (deep_value_windows['macro_uptrend_pct'] > 0.5) & 
                (deep_value_windows['avg_sentiment'] > 0.5)
            ]
            if len(triple_boost) > 0:
                print(f"    Deep Value + Macro Uptrend + High Sentiment: {len(triple_boost)} windows, {triple_boost['excess'].mean():.2f}% avg excess")
        
        if len(danger_windows) > 0:
            print(f"  Danger (MVRV > 3.0): {len(danger_windows)} windows, {danger_windows['excess'].mean():.2f}% avg excess")
            # Danger + weak trend + low sentiment (maximum protection)
            triple_protection = danger_windows[
                (danger_windows['macro_uptrend_pct'] < 0.3) & 
                (danger_windows['avg_sentiment'] < 0.4)
            ]
            if len(triple_protection) > 0:
                print(f"    Danger + Weak Trend + Low Sentiment: {len(triple_protection)} windows, {triple_protection['excess'].mean():.2f}% avg excess")
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in performance"""
        print("\n" + "="*50)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*50)
        
        # Performance by month
        monthly_performance = self.results_df.groupby('month').agg({
            'excess': 'mean',
            'is_win': 'mean'
        }).round(2)
        
        monthly_performance['WinRate'] = (monthly_performance['is_win'] * 100).round(1)
        
        print("Performance by start month:")
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        for month in monthly_performance.index:
            month_name = month_names.get(month, str(month))
            excess = monthly_performance.loc[month, 'excess']
            win_rate = monthly_performance.loc[month, 'WinRate']
            print(f"  {month_name}: {excess:.2f}% excess, {win_rate:.1f}% win rate")
        
        # Performance trend over time
        print(f"\nPerformance trend (3-year rolling average):")
        self.results_df_sorted = self.results_df.sort_values('start_date')
        rolling_performance = self.results_df_sorted['excess'].rolling(window=1095, min_periods=365).mean()  # ~3 years
        
        # Sample every 365 windows for annual trend
        for i in range(0, len(rolling_performance), 365):
            if pd.notna(rolling_performance.iloc[i]):
                date = self.results_df_sorted.iloc[i]['start_date'].strftime('%Y')
                performance = rolling_performance.iloc[i]
                print(f"  {date}: {performance:.2f}% (3-year rolling avg)")
    
    def run_analysis(self):
        """Run complete performance analysis"""
        print("Loading data...")
        self.load_data()
        
        # Run all analyses
        self.analyze_yearly_performance()
        self.analyze_mvrv_zones()
        self.analyze_market_regimes()
        self.analyze_loss_patterns()
        self.analyze_extreme_windows()
        self.analyze_signal_distribution()
        self.analyze_temporal_patterns()
        
        print(f"\n" + "="*50)
        print("V5 CAPSTONE MAXIMIZER ANALYSIS COMPLETE")
        print("="*50)


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.run_analysis()
"""
Performance Analysis Script - Objective analysis of model performance

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

from my_model.model_development import precompute_features
from template.prelude_template import load_data


class PerformanceAnalyzer:
    def __init__(self):
        self.results_df = None
        self.features_df = None
        self.btc_df = None
        
    def load_data(self):
        """Load backtest results and feature data"""
        # Load backtest results
        with open("my_model/output/metrics.json", 'r') as f:
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
        """Analyze MVRV zone distribution and performance"""
        print("\n" + "="*50)
        print("MVRV ZONE ANALYSIS")
        print("="*50)
        
        # Zone distribution in dataset
        zones = self.features_df['mvrv_zone'].value_counts().sort_index()
        zone_names = {-2: 'DeepValue', -1: 'Value', 0: 'Neutral', 1: 'Caution', 2: 'Danger'}
        
        print("Historical zone distribution:")
        for zone, count in zones.items():
            name = zone_names.get(zone, f'Zone{zone}')
            pct = count / len(self.features_df) * 100
            print(f"  {name}: {count} days ({pct:.1f}%)")
        
        # Window performance by dominant MVRV zone
        window_zones = []
        window_mvrv_stats = []
        
        for _, row in self.results_df.iterrows():
            start_date = row['start_date']
            end_date = start_date + pd.Timedelta(days=365)
            
            window_features = self.features_df.loc[start_date:end_date]
            if len(window_features) > 0:
                # Dominant zone
                dominant_zone = window_features['mvrv_zone'].mode().iloc[0] if len(window_features['mvrv_zone'].mode()) > 0 else 0
                # MVRV statistics for this window
                avg_mvrv = window_features['mvrv_zscore'].mean()
                max_mvrv = window_features['mvrv_zscore'].max()
                min_mvrv = window_features['mvrv_zscore'].min()
            else:
                dominant_zone = 0
                avg_mvrv = max_mvrv = min_mvrv = 0
            
            window_zones.append(dominant_zone)
            window_mvrv_stats.append({'avg': avg_mvrv, 'max': max_mvrv, 'min': min_mvrv})
        
        self.results_df['dominant_zone'] = window_zones
        
        # Performance by zone
        zone_performance = self.results_df.groupby('dominant_zone').agg({
            'excess': ['mean', 'median', 'std', 'count'],
            'is_win': 'mean'
        }).round(2)
        
        zone_performance.columns = ['Mean', 'Median', 'Std', 'Count', 'WinRate']
        zone_performance['WinRate'] = (zone_performance['WinRate'] * 100).round(1)
        
        print("\nWindow performance by dominant zone:")
        for zone in zone_performance.index:
            name = zone_names.get(zone, f'Zone{zone}')
            stats = zone_performance.loc[zone]
            print(f"  {name}: {stats['Mean']:.2f}% mean, {stats['WinRate']:.1f}% win rate, {stats['Count']} windows")
    
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
        """Analyze MVRV signal distribution across windows"""
        print("\n" + "="*50)
        print("SIGNAL DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # MVRV Z-score ranges
        mvrv_bins = [-np.inf, -2, -1, 0, 1.5, 2.5, np.inf]
        mvrv_labels = ['DeepValue', 'Value', 'Neutral', 'Caution', 'Danger', 'Extreme']
        
        # Calculate average MVRV for each window
        window_mvrv_avg = []
        for _, row in self.results_df.iterrows():
            start_date = row['start_date']
            end_date = start_date + pd.Timedelta(days=365)
            window_features = self.features_df.loc[start_date:end_date]
            if len(window_features) > 0:
                avg_mvrv = window_features['mvrv_zscore'].mean()
            else:
                avg_mvrv = 0
            window_mvrv_avg.append(avg_mvrv)
        
        self.results_df['avg_mvrv_zscore'] = window_mvrv_avg
        self.results_df['mvrv_range'] = pd.cut(self.results_df['avg_mvrv_zscore'], bins=mvrv_bins, labels=mvrv_labels)
        
        # Performance by MVRV range
        mvrv_performance = self.results_df.groupby('mvrv_range', observed=True).agg({
            'excess': ['mean', 'median', 'count'],
            'is_win': 'mean',
            'avg_mvrv_zscore': 'mean'
        }).round(2)
        
        mvrv_performance.columns = ['ExcessMean', 'ExcessMedian', 'Count', 'WinRate', 'AvgMVRV']
        mvrv_performance['WinRate'] = (mvrv_performance['WinRate'] * 100).round(1)
        
        print("Performance by MVRV Z-score range:")
        for mvrv_range in mvrv_performance.index:
            if pd.notna(mvrv_range):
                stats = mvrv_performance.loc[mvrv_range]
                print(f"  {mvrv_range}: {stats['ExcessMean']:.2f}% excess, {stats['WinRate']:.1f}% win rate, {stats['Count']} windows (avg Z: {stats['AvgMVRV']:.2f})")
    
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
        print("ANALYSIS COMPLETE")
        print("="*50)


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.run_analysis()
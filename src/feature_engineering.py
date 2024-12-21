# src/feature_engineering.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features efficiently"""
        logger.info("Starting feature engineering...")
        print("Creating features...")
        print(f"Initial shape: {df.shape}")
        
        # Pre-sort the dataframe once
        df = df.sort_values(['symbol_id', 'date_id', 'time_id'])
        
        # Initialize dictionary to store all features
        feature_dicts = {
            'time': self._create_time_features(df),
            'rolling': self._create_rolling_features_efficient(df),
            'lag': self._create_lag_features_efficient(df),
            'cross': self._create_cross_features_efficient(df)
        }
        
        # Combine all features efficiently
        all_features = pd.concat([df] + [features for features in feature_dicts.values()], axis=1)
        print(f"Final shape: {all_features.shape}")
        return all_features
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        print("Creating time features...")
        
        features_dict = {
            'time_diff': df.groupby('symbol_id')['time_id'].diff(),
            'date_diff': df.groupby('symbol_id')['date_id'].diff(),
            'time_pct': df.groupby('date_id')['time_id'].rank(pct=True)
        }
        
        return pd.DataFrame(features_dict)
    
    def _create_rolling_features_efficient(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features efficiently"""
        print("Creating rolling features...")
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        windows = [5, 10, 20]
        
        # Initialize dictionary to store all features
        features_dict = {}
        
        for window in tqdm(windows, desc="Creating rolling features"):
            # Calculate rolling statistics for each window size
            grouped = df[feature_cols].groupby(df['symbol_id'])
            
            # Calculate means and stds in one pass
            rolling = grouped.rolling(window=window, min_periods=1)
            means = rolling.mean()
            stds = rolling.std()
            
            # Add features to dictionary
            for feat in feature_cols:
                features_dict[f'{feat}_rmean_{window}'] = means[feat]
                features_dict[f'{feat}_rstd_{window}'] = stds[feat]
        
        return pd.DataFrame(features_dict)
    
    def _create_lag_features_efficient(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features efficiently"""
        print("Creating lag features...")
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        lags = [1, 2, 3]
        
        # Initialize dictionary to store all features
        features_dict = {}
        
        # Calculate all lags at once
        for lag in tqdm(lags, desc="Creating lag features"):
            lagged_values = df.groupby('symbol_id')[feature_cols].shift(lag)
            for feat in feature_cols:
                features_dict[f'{feat}_lag_{lag}'] = lagged_values[feat]
        
        return pd.DataFrame(features_dict)
    
    def _create_cross_features_efficient(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features efficiently"""
        print("Creating cross features...")
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        
        features_dict = {
            'mean_features': df[feature_cols].mean(axis=1),
            'std_features': df[feature_cols].std(axis=1)
        }
        
        # Feature interactions (first few features only for efficiency)
        for i in range(5):
            for j in range(i+1, 5):
                feat_i = f'feature_{i:02d}'
                feat_j = f'feature_{j:02d}'
                features_dict[f'interact_{i}_{j}'] = df[feat_i] * df[feat_j]
        
        return pd.DataFrame(features_dict)

    def _memory_status(self):
        """Print current memory usage"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"\nMemory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
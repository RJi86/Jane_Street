# src/feature_engineering.py
import polars as pl
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create all features efficiently using Polars"""
        logger.info("Starting feature engineering...")
        print("Creating features...")
        print(f"Initial shape: {df.shape}")
        
        # Sort the dataframe
        df = df.sort(["symbol_id", "date_id", "time_id"])
        
        # Create features
        df_with_features = (df
            .pipe(self._create_time_features)
            .pipe(self._create_rolling_features)
            .pipe(self._create_lag_features)
            .pipe(self._create_cross_features)
        )
        
        print(f"Final shape: {df_with_features.shape}")
        return df_with_features
    
    def _create_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create time-based features using Polars"""
        print("Creating time features...")
        
        return df.with_columns([
            pl.col("time_id").diff().over("symbol_id").alias("time_diff"),
            pl.col("date_id").diff().over("symbol_id").alias("date_diff"),
            pl.col("time_id").rank().over("date_id").alias("time_pct")
        ])
    
    def _create_rolling_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create rolling window features using Polars"""
        print("Creating rolling features...")
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        windows = [5, 10, 20]
        
        expressions = []
        for window in tqdm(windows, desc="Creating rolling features"):
            for feat in feature_cols:
                expressions.extend([
                    pl.col(feat)
                        .rolling_mean(window_size=window)
                        .over("symbol_id")
                        .alias(f'{feat}_rmean_{window}'),
                    pl.col(feat)
                        .rolling_std(window_size=window)
                        .over("symbol_id")
                        .alias(f'{feat}_rstd_{window}')
                ])
        
        return df.with_columns(expressions)
    
    def _create_lag_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create lagged features using Polars"""
        print("Creating lag features...")
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        lags = [1, 2, 3]
        
        expressions = []
        for lag in tqdm(lags, desc="Creating lag features"):
            for feat in feature_cols:
                expressions.append(
                    pl.col(feat)
                        .shift(lag)
                        .over("symbol_id")
                        .alias(f'{feat}_lag_{lag}')
                )
        
        return df.with_columns(expressions)
    
    def _create_cross_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create interaction features using Polars"""
        print("Creating cross features...")
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        
        # Basic statistical features - Modified to have unique names
        expressions = [
            pl.col(feature_cols).mean().alias("all_features_mean"),  # Changed name to be unique
            pl.col(feature_cols).std().alias("all_features_std")     # Changed name to be unique
        ]
        
        # Feature interactions (first few features only for efficiency)
        for i in range(5):
            for j in range(i+1, 5):
                feat_i = f'feature_{i:02d}'
                feat_j = f'feature_{j:02d}'
                expressions.append(
                    (pl.col(feat_i) * pl.col(feat_j)).alias(f'interact_{i}_{j}')
                )
        
        return df.with_columns(expressions)
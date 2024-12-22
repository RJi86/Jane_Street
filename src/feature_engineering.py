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
        print("Initial columns:", df.columns)
        
        # Sort the dataframe
        df = df.sort(["symbol_id", "date_id", "time_id"])
        
        # Create features with logging after each step
        df = self._create_time_features(df)
        
        df = self._create_rolling_features(df)
        
        df = self._create_lag_features(df)
        
        df = self._create_cross_features(df)
        
        return df
    
    def _create_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create time-based features using Polars"""
        print("Creating time features...")
        
        # Create time_idx column
        df = df.with_row_count("time_idx")
        
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
        print("Available columns before cross features:", df.columns)
        
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        
        # Exclude debugging features
        exclude_features = [f"feature_{i:02d}_lag_{j}" for i in range(60, 80) for j in range(1, 4)] + \
                           [f"feature_59_rmean_20", f"feature_59_rstd_20"]
        feature_cols = [col for col in feature_cols if col not in exclude_features]

        # First, calculate the mean and std in separate operations
        df = df.with_columns([
            pl.fold(
                acc=pl.lit(0.0),
                function=lambda acc, x: acc + x,
                exprs=[pl.col(col) for col in feature_cols]
            ).truediv(len(feature_cols)).alias("all_features_mean")
        ])
        
        print("Columns after mean calculation:", df.columns)
        
        # Now calculate std using the already calculated mean
        df = df.with_columns([
            pl.fold(
                acc=pl.lit(0.0),
                function=lambda acc, x: acc + x.pow(2),
                exprs=[pl.col(col) for col in feature_cols]
            ).truediv(len(feature_cols))
            .sub(pl.col("all_features_mean").pow(2))
            .sqrt()
            .alias("all_features_std")
        ])
        
        print("Columns after std calculation:", df.columns)
        
        # Feature interactions (first few features only for efficiency)
        interaction_expressions = []
        for i in range(5):
            for j in range(i+1, 5):
                feat_i = f'feature_{i:02d}'
                feat_j = f'feature_{j:02d}'
                interaction_expressions.append(
                    (pl.col(feat_i) * pl.col(feat_j)).alias(f'interact_{i}_{j}')
                )
        
        # Add interaction features
        df = df.with_columns(interaction_expressions)
        
        print("Final columns after cross features:", df.columns)
        
        return df
import lightgbm as lgb
import numpy as np
import polars as pl
import psutil
import torch
import os

class Trainer:
    def __init__(self, config, checkpoint_manager):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.setup_gpu()

    def setup_gpu(self):
        """Setup GPU if available and requested"""
        if self.config.use_gpu:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.gpu_device)
                print(f"\nGPU Setup:")
                print(f"- Using GPU: {torch.cuda.get_device_name(self.config.gpu_device)}")
                print(f"- Device Index: {self.config.gpu_device}")
                print(f"- CUDA Version: {torch.version.cuda}")
            else:
                print("\nWARNING: GPU was requested but no CUDA device is available.")
                print("Falling back to CPU training.")
                self.config.use_gpu = False

    def weighted_r2(self, preds, train_data):
        """Calculate Weighted R²"""
        labels = train_data.get_label()
        weights = train_data.get_weight()
        
        # Compute weighted mean of the true values
        weighted_mean = np.sum(weights * labels) / np.sum(weights)
        
        # Compute weighted sum of squared errors and total sum of squared errors
        weighted_sq_error = np.sum(weights * (labels - preds) ** 2)
        weighted_total_sq = np.sum(weights * (labels - weighted_mean) ** 2)
        
        # Calculate R²
        r2 = 1 - (weighted_sq_error / weighted_total_sq)
        return 'weighted_r2', r2, False

    def train(self, train_data: pl.DataFrame, val_data: pl.DataFrame, start_epoch=0, model=None):
        """Train the model with Polars DataFrames"""
        self.print_memory_usage()

        # Get feature columns
        feature_cols = [col for col in train_data.columns 
                       if col not in ['date_id', 'time_id', 'symbol_id', 'weight'] + 
                       [f'responder_{i}' for i in range(9)]]

        # Setup LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': min(self.config.num_leaves, 31),
            'learning_rate': self.config.learning_rate,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_bin': 255,
            'min_data_in_leaf': 20,
            'max_depth': 8
        }

        if self.config.use_gpu:
            params.update({
                'device': 'gpu',
                'gpu_device_id': self.config.gpu_device,
                'max_bin': 63
            })

        # Convert Polars DataFrame to numpy arrays efficiently
        print("\nPreparing training data...")
        X_train = train_data.select(feature_cols).to_numpy()
        y_train = train_data.select(['responder_6']).to_numpy().ravel()
        w_train = train_data.select(['weight']).to_numpy().ravel()

        X_val = val_data.select(feature_cols).to_numpy()
        y_val = val_data.select(['responder_6']).to_numpy().ravel()
        w_val = val_data.select(['weight']).to_numpy().ravel()

        # Create datasets
        print("\nCreating LightGBM datasets...")
        train_dataset = lgb.Dataset(
            X_train.astype(np.float32),
            label=y_train.astype(np.float32),
            weight=w_train.astype(np.float32),
            free_raw_data=True
        )

        val_dataset = lgb.Dataset(
            X_val.astype(np.float32),
            label=y_val.astype(np.float32),
            weight=w_val.astype(np.float32),
            reference=train_dataset,
            free_raw_data=True
        )

        # Training with callbacks
        print("\nStarting training...")
        callbacks = [
            lgb.callback.log_evaluation(period=100),
            lgb.callback.record_evaluation({}),
            # lgb.callback.early_stopping(self.config.early_stopping_rounds)
        ]

        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            callbacks=callbacks,
            num_boost_round=self.config.num_boost_round,
            init_model=model,
            valid_names=['train', 'valid'],
            feval=self.weighted_r2  # Add custom evaluation metric
        )

        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            state={
                'model': model,
                'feature_columns': feature_cols,
                'model_params': params
            },
            epoch=start_epoch + self.config.num_boost_round,
            metrics=model.best_score
        )

        return model

    @staticmethod
    def print_memory_usage():
        """Print current memory usage"""
        process = psutil.Process(os.getpid())
        print(f"\nMemory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

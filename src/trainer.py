import lightgbm as lgb
import torch
from tqdm import tqdm
import psutil
import os
import numpy as np
import polars as pl
import warnings

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

    def r_squared(self, y_true, y_pred, weights=None):
        """Calculate R² score, optionally with weights"""
        if weights is None:
            weights = np.ones_like(y_true)
        weighted_mean = np.average(y_true, weights=weights)
        total_ss = np.sum(weights * (y_true - weighted_mean) ** 2)
        residual_ss = np.sum(weights * (y_true - y_pred) ** 2)
        r2 = 1 - (residual_ss / total_ss)
        return r2
    
    def train(self, train_data: pl.DataFrame, val_data: pl.DataFrame, start_epoch=0, model=None):
        """Train the model with Polars DataFrames"""
        self.print_memory_usage()
        
        # Get feature columns
        feature_cols = [col for col in train_data.columns 
                       if col not in ['date_id', 'time_id', 'symbol_id', 'weight'] + 
                       [f'responder_{i}' for i in range(9)]]
        
        # Setup LightGBM parameters with additional warning suppressions
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': min(self.config.num_leaves, 31),
            'learning_rate': self.config.learning_rate,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 2,
            'max_bin': 255,
            'min_data_in_leaf': 20,
            'max_depth': 8,
            'force_col_wise': True,
            'deterministic': False,
            'use_missing': False,
            'zero_as_missing': False,
            'first_metric_only': True
        }
        
        if self.config.use_gpu:
            params.update({
                'device_type': 'cuda',
                'gpu_device_id': self.config.gpu_device,
                'max_bin': 63,
                'gpu_use_dp': False,
                'gpu_platform_id': 0,
            })
        
        # Prepare data
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
        
        # Custom callback for more formatted logging
        train_scores = []
        valid_scores = []
        total_epochs = self.config.num_boost_round

        def callback(env):
            if env.iteration % 1 == 0:  # Log every epoch
                # Get predictions for R² calculation
                y_train_pred = env.model.predict(X_train)
                y_val_pred = env.model.predict(X_val)
                
                # Calculate R² scores
                train_r2 = self.r_squared(y_train, y_train_pred, w_train)
                valid_r2 = self.r_squared(y_val, y_val_pred, w_val)
                
                # Clear the line before printing new output
                print('\r', end='')
                
                # Format the output with epoch counter and metrics
                print(f'Epoch {env.iteration+1}/{total_epochs} - '
                      f"Train RMSE: {env.evaluation_result_list[0][2]:.6f} "
                      f"R²: {train_r2:.6f} | "
                      f"Valid RMSE: {env.evaluation_result_list[1][2]:.6f} "
                      f"R²: {valid_r2:.6f}", end='\n')
                
                train_scores.append((env.evaluation_result_list[0][2], train_r2))
                valid_scores.append((env.evaluation_result_list[1][2], valid_r2))
        
        # Training with callbacks and warning suppression
        print("\nStarting training...")
        print("=" * 80)  # Add separator line
        
        # Suppress LightGBM warnings at the Python level
        warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
        
        callbacks = [
            callback,
            lgb.callback.record_evaluation({})
        ]
        
        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            callbacks=callbacks,
            num_boost_round=self.config.num_boost_round,
            init_model=model,
            valid_names=['train', 'valid']
        )
        
        print("\n" + "=" * 80)  # Add separator line
        print("Training completed!")
        
        # Save final checkpoint with additional metrics
        final_metrics = {
            'train_rmse': train_scores[-1][0],
            'train_r2': train_scores[-1][1],
            'valid_rmse': valid_scores[-1][0],
            'valid_r2': valid_scores[-1][1]
        }
        
        self.checkpoint_manager.save_checkpoint(
            state={
                'model': model,
                'feature_columns': feature_cols,
                'model_params': params,
                'train_history': train_scores,
                'valid_history': valid_scores
            },
            epoch=start_epoch + self.config.num_boost_round,
            metrics=final_metrics
        )
        
        return model
    
    @staticmethod
    def print_memory_usage():
        """Print current memory usage"""
        process = psutil.Process(os.getpid())
        print(f"\nMemory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
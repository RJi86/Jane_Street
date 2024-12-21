import lightgbm as lgb
import torch
from tqdm import tqdm
import psutil
import os
import numpy as np

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
    
    def train(self, train_data, val_data, start_epoch=0, model=None):
        """Train the model"""
        # Print memory usage
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
            'num_leaves': self.config.num_leaves,
            'learning_rate': self.config.learning_rate,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'verbose': -1
        }
        
        if self.config.use_gpu:
            params.update({
                'device': 'gpu',
                'gpu_device_id': self.config.gpu_device
            })
            
        # Create datasets
        print("\nCreating LightGBM datasets...")
        train_dataset = lgb.Dataset(
            train_data[feature_cols],
            label=train_data['responder_6'],
            weight=train_data['weight']
        )
        
        val_dataset = lgb.Dataset(
            val_data[feature_cols],
            label=val_data['responder_6'],
            weight=val_data['weight'],
            reference=train_dataset
        )
        
        # Training with callbacks
        print("\nStarting training...")
        callbacks = [
            lgb.callback.log_evaluation(period=100),
            lgb.callback.record_evaluation({}),
        ]
        
        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            callbacks=callbacks,
            num_boost_round=self.config.num_boost_round,
            init_model=model,  # Continue from existing model if provided
            valid_names=['train', 'valid']
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
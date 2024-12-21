import argparse
import datetime
import os
import sys
from pathlib import Path
import torch
import logging
import psutil
from tqdm import tqdm
import numpy as np
import json
import polars as pl

from config.model_config import ModelConfig
from src.trainer import Trainer
from src.checkpoint_manager import CheckpointManager
from src.feature_engineering import FeatureEngineer
from src.utils import load_training_data, setup_logging

class TrainingManager:
    def __init__(self, args):
        self.args = args
        self.config = self._initialize_config()
        # Remove config parameter from setup_logging call
        self.logger = setup_logging()  # Changed this line
        self.checkpoint_manager = CheckpointManager(self.config)
        self.start_time = datetime.datetime.utcnow()
        
    def _initialize_config(self):
        config = ModelConfig()
        config.model_name = self.args.model_name
        config.use_gpu = self.args.use_gpu
        config.gpu_device = self.args.gpu_device
        return config
        
    def print_system_info(self):
        """Print system information"""
        print("\nSystem Information:")
        print(f"- Python version: {sys.version.split()[0]}")
        print(f"- Current directory: {Path.cwd()}")
        print(f"- CPU count: {psutil.cpu_count()}")
        print(f"- Memory available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        
        if torch.cuda.is_available():
            print(f"- CUDA version: {torch.version.cuda}")
            print(f"- GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                
    def print_training_config(self):
        """Print training configuration"""
        print("\nTraining Configuration:")
        print(f"- Model name: {self.config.model_name}")
        print(f"- GPU enabled: {self.config.use_gpu}")
        print(f"- GPU device: {self.config.gpu_device}")
        print(f"- Data partitions: {self.args.partitions}")
        print(f"- Learning rate: {self.config.learning_rate}")
        print(f"- Number of leaves: {self.config.num_leaves}")
        print(f"- Feature fraction: {self.config.feature_fraction}")
        print(f"- Bagging fraction: {self.config.bagging_fraction}")
        print(f"- Number of rounds: {self.config.num_boost_round}")
        print(f"\nTime (UTC): {self.start_time}")
        print(f"User: {os.getenv('USERNAME', 'unknown')}")
        
    def save_run_metadata(self):
        """Save metadata about the training run"""
        metadata = {
            'start_time': self.start_time.isoformat(),
            'model_name': self.config.model_name,
            'gpu_enabled': self.config.use_gpu,
            'partitions': self.args.partitions,
            'user': os.getenv('USERNAME', 'unknown'),
            'python_version': sys.version.split()[0],
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_info': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        }
        
        metadata_path = self.config.checkpoint_dir / f"run_metadata_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    def run(self):
        """Main training process"""
        self.print_system_info()
        self.print_training_config()
        self.save_run_metadata()
        
        if self.args.list_checkpoints:
            self.checkpoint_manager.list_checkpoints()
            return
            
        trainer = None
        model = None
        start_epoch = 0
        
        try:
            # Initialize trainer
            trainer = Trainer(self.config, self.checkpoint_manager)
            
            # Load checkpoint if resuming
            if self.args.resume or self.args.checkpoint:
                print("\nLoading checkpoint...")
                checkpoint = self.checkpoint_manager.load_checkpoint(self.args.checkpoint)
                if checkpoint:
                    model = checkpoint['model']
                    start_epoch = checkpoint['metadata']['epoch']
                    print(f"Successfully loaded checkpoint from epoch {start_epoch}")
                else:
                    print("No checkpoint found, starting from scratch")
            
            # Load and process data
            print("\nLoading data...")
            df = load_training_data(self.config.data_dir, partitions=self.args.partitions)
            
            if self.args.debug:
                print("\nDebug mode: Using subset of data")
                df = df.sample(n=min(10000, len(df)), random_state=42)
            
            # Feature engineering
            print("\nPerforming feature engineering...")
            feature_engineer = FeatureEngineer(self.config)
            df = feature_engineer.create_features(df)
            
            # Train/validation split
            print("\nSplitting data...")
            try:
                split_date = df['date_id'].max() * 0.8
                print(f"Split date: {split_date}")
                
                # Use filter instead of boolean indexing
                train_data = df.filter(pl.col('date_id') < split_date)
                val_data = df.filter(pl.col('date_id') >= split_date)
                
                print(f"\nDataset shapes:")
                print(f"- Training data: {train_data.shape}")
                print(f"- Validation data: {val_data.shape}")
                
            except Exception as e:
                print(f"Error during data splitting: {e}")
                print(f"DataFrame info:")
                print(f"- Shape: {df.shape}")
                print(f"- Columns: {df.columns}")
                print(f"- date_id range: {df['date_id'].min()} to {df['date_id'].max()}")
                raise
            
            print(f"\nDataset shapes:")
            print(f"- Training data: {train_data.shape}")
            print(f"- Validation data: {val_data.shape}")
            
            # Train model
            print("\nStarting training...")
            model = trainer.train(
                train_data=train_data,
                val_data=val_data,
                start_epoch=start_epoch,
                model=model
            )
            
            print("\nTraining completed successfully!")
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            if model is not None:
                print("Saving emergency checkpoint...")
                try:
                    emergency_checkpoint_name = f"{self.config.model_name}_emergency_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                    self.checkpoint_manager.save_checkpoint(
                        state={'model': model},
                        epoch=start_epoch,
                        metrics={},
                        user_info={'emergency': True, 'interrupted': True}
                    )
                    print(f"Emergency checkpoint saved: {emergency_checkpoint_name}")
                except Exception as e:
                    print(f"Failed to save emergency checkpoint: {e}")
            return 1
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            self.logger.exception("Training failed with exception")
            raise
            
        finally:
            # Print final memory usage
            process = psutil.Process(os.getpid())
            print(f"\nFinal memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
        return 0

def main():
    parser = argparse.ArgumentParser(description='Jane Street Market Prediction Training')
    
    # Model and training arguments
    parser.add_argument('--model_name', 
                       default='lgb_baseline', 
                       help='Name of the model (used for checkpoints)')
    parser.add_argument('--use_gpu', 
                       action='store_true', 
                       help='Use GPU for training if available')
    parser.add_argument('--gpu_device', 
                       type=int, 
                       default=0,
                       help='GPU device index to use')
    
    # Checkpoint arguments
    parser.add_argument('--resume', 
                       action='store_true', 
                       help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', 
                       type=str, 
                       help='Resume from specific checkpoint name')
    parser.add_argument('--list_checkpoints', 
                       action='store_true', 
                       help='List all available checkpoints')
    
    # Data arguments
    parser.add_argument('--partitions', 
                       type=int, 
                       nargs='+', 
                       default=[0, 1],
                       help='Data partitions to use (e.g., --partitions 0 1 2)')
    
    # Additional arguments
    parser.add_argument('--debug', 
                       action='store_true', 
                       help='Enable debug mode with smaller dataset')
    
    args = parser.parse_args()
    
    # Print start time and user info
    print(f"\nCurrent Date and Time (UTC): {datetime.datetime.utcnow()}")
    print(f"Current User's Login: {os.getenv('USERNAME', 'unknown')}\n")
    
    # Run training
    training_manager = TrainingManager(args)
    return training_manager.run()

if __name__ == "__main__":
    sys.exit(main())
import argparse
import datetime
import os
import sys  # Add this line
from pathlib import Path
import torch
import logging
import psutil
from tqdm import tqdm
import numpy as np
import json
import polars as pl

from config.model_config import ModelConfig
from src.tft_trainer import TFTTrainer
from src.checkpoint_manager import CheckpointManager
from src.feature_engineering import FeatureEngineer
from src.utils import load_training_data, setup_logging

class TrainingManager:
    def __init__(self, args):
        self.args = args
        self.config = self._initialize_config()
        self.logger = setup_logging()
        self.checkpoint_manager = CheckpointManager(self.config)
        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        
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
        print(f"- Max encoder length: {self.config.max_encoder_length}")
        print(f"- Max prediction length: {self.config.max_prediction_length}")
        print(f"- Hidden size: {self.config.hidden_size}")
        print(f"- Attention head size: {self.config.attention_head_size}")
        print(f"- Dropout: {self.config.dropout}")
        print(f"- Hidden continuous size: {self.config.hidden_continuous_size}")
        print(f"- Output size: {self.config.output_size}")
        print(f"- Max epochs: {self.config.max_epochs}")
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
        
        try:
            # Initialize TFTTrainer
            trainer = TFTTrainer(self.config)

            # Load and process data
            print("\nLoading data...")
            df = load_training_data(self.config.data_dir, partitions=self.args.partitions)
            
            if self.args.debug:
                print("\nDebug mode: Using subset of data")
                df = df.head(10000)
            
            # Feature engineering
            print("\nPerforming feature engineering...")
            feature_engineer = FeatureEngineer(self.config)
            df = feature_engineer.create_features(df).collect(streaming=True)
            
            # Train/validation split
            print("\nSplitting data...")
            try:
                split_date = df.select('date_id').max() * 0.8
                print(f"Split date: {split_date}")
                
                train_data = df.filter(pl.col('date_id') < split_date)
                val_data = df.filter(pl.col('date_id') >= split_date)
                
            except Exception as e:
                print(f"Error during data splitting: {e}")
                print(f"DataFrame info:")
                print(f"- Columns: {df.columns}")
                print(f"- date_id range: {df.select('date_id').min()} to {df.select('date_id').max()}")
                raise
            # Train model
            print("\nStarting training...")
            model = trainer.train(
                data=train_data  # Assuming train_data is properly preprocessed
            )
            
            print("\nTraining completed successfully!")
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            return 1
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            self.logger.exception("Training failed with exception")
            raise
            
        finally:
            process = psutil.Process(os.getpid())
            print(f"\nFinal memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
        return 0

def main():
    parser = argparse.ArgumentParser(description='Jane Street Market Prediction Training')
    
    # Model and training arguments
    parser.add_argument('--model_name', 
                       default='tft', 
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
    print(f"\nCurrent Date and Time (UTC): {datetime.datetime.now(datetime.timezone.utc)}")
    print(f"Current User's Login: {os.getenv('USERNAME', 'unknown')}\n")
    
    # Run training
    training_manager = TrainingManager(args)
    return training_manager.run()

if __name__ == "__main__":
    sys.exit(main())
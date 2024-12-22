from config.model_config import ModelConfig
from src.utils import load_training_data
import sys

def test_data_loading():
    config = ModelConfig()
    print("\nTesting data loading...")
    print(f"Data directory: {config.data_dir}")
    
    try:
        # Try loading just one partition
        df = load_training_data(config.data_dir, partitions=[0])
        print("\nSuccessfully loaded test partition!")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
    except Exception as e:
        print(f"\nError during test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_data_loading()
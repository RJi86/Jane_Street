import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = "/shared/Dataset"
    TRAIN_DIR = DATA_DIR / "train.parquet"
    MODEL_DIR = BASE_DIR / "models"
    
    # Data parameters
    N_FEATURES = 79
    N_RESPONDERS = 9
    TARGET_RESPONDER = 6
    
    # Model parameters
    RANDOM_STATE = 42
    NUM_SPLITS = 5
    BATCH_SIZE = 10000
    
    # Feature engineering parameters
    ROLLING_WINDOWS = [5, 10, 20]
    LAG_WINDOWS = [1, 2, 3]
    
    @staticmethod
    def get_feature_names():
        return [f'feature_{i:02d}' for i in range(Config.N_FEATURES)]
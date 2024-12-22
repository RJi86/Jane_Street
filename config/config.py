import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "Dataset"
    TRAIN_DIR = DATA_DIR / "train.parquet"
    MODEL_DIR = BASE_DIR / "models"
    
    # Data parameters
    N_FEATURES = 79
    N_RESPONDERS = 9
    TARGET_RESPONDER = 6
    
    # Model parameters
    RANDOM_STATE = 42
    NUM_SPLITS = 5
    BATCH_SIZE = 64  # Adjusted for TFT
    
    # Feature engineering parameters
    ROLLING_WINDOWS = [5, 10, 20]
    LAG_WINDOWS = [1, 2, 3]
    
    # TFT-specific parameters
    MAX_ENCODER_LENGTH = 60
    MAX_PREDICTION_LENGTH = 30
    HIDDEN_SIZE = 16
    ATTENTION_HEAD_SIZE = 4
    DROPOUT = 0.3
    HIDDEN_CONTINUOUS_SIZE = 8
    OUTPUT_SIZE = 7
    LEARNING_RATE = 0.03
    MAX_EPOCHS = 30
    
    @staticmethod
    def get_feature_names():
        return [f'feature_{i:02d}' for i in range(Config.N_FEATURES)]
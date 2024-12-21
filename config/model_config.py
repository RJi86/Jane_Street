from pathlib import Path
from dataclasses import dataclass
import datetime
import os

@dataclass
class ModelConfig:
    # Model parameters
    model_name: str = "lgb_baseline"
    model_type: str = "lightgbm"
    
    # GPU settings
    use_gpu: bool = False
    gpu_device: int = 0
    
    # Training parameters
    learning_rate: float = 0.05
    num_leaves: int = 31
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    early_stopping_rounds: int = -1
    num_boost_round: int = 1000
    
    def __post_init__(self):
        # Paths
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = os.getenv("DATA_DIR") or self.base_dir / "Dataset" / "train.parquet"
        self.checkpoint_dir = self.base_dir / "checkpoints"
        
        # Create necessary directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Runtime info
        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        self.user = "RJi86"
from pathlib import Path
from dataclasses import dataclass
import datetime
import os

@dataclass
class ModelConfig:
    # Model parameters
    model_name: str = "tft"
    model_type: str = "temporal_fusion_transformer"
    
    # GPU settings
    use_gpu: bool = False
    gpu_device: int = 0
    
    # Training parameters
    learning_rate: float = 0.03
    max_encoder_length: int = 60
    max_prediction_length: int = 30
    hidden_size: int = 16
    attention_head_size: int = 4
    dropout: float = 0.3
    hidden_continuous_size: int = 8
    output_size: int = 7
    max_epochs: int = 30
    batch_size: int = 64

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
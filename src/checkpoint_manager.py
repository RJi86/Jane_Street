import os
import json
import datetime
from pathlib import Path
import lightgbm as lgb

class CheckpointManager:
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, state, epoch, metrics, user_info=None):
        """
        Save a checkpoint with model state and metadata
        Args:
            state: dict containing model and other states
            epoch: current training epoch
            metrics: dictionary of training metrics
            user_info: optional user information
        """
        # Create timestamp and checkpoint name
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"{self.config.model_name}_epoch{epoch}_{timestamp}"
        
        # Save model
        model_path = self.checkpoint_dir / f"{checkpoint_name}.txt"
        state['model'].save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics,
            'feature_columns': state.get('feature_columns', []),
            'model_params': state.get('model_params', {}),
            'last_partition_processed': state.get('last_partition_processed', 0),
            'user_info': user_info or {
                'username': os.getenv('USERNAME', 'unknown'),
                'save_time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        }
        
        metadata_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"\nCheckpoint saved:")
        print(f"- Model: {model_path}")
        print(f"- Metadata: {metadata_path}")
        print(f"- Epoch: {epoch}")
        print(f"- Timestamp: {timestamp}")
        
    def get_latest_checkpoint(self):
        """Find and return the most recent checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob(f"{self.config.model_name}*.json"))
        if not checkpoints:
            return None
            
        latest = max(checkpoints, key=os.path.getmtime)
        checkpoint_base = latest.stem
        
        return {
            'model_path': self.checkpoint_dir / f"{checkpoint_base}.txt",
            'metadata_path': latest
        }
        
    def load_checkpoint(self, checkpoint_name=None):
        """
        Load a specific checkpoint or the latest one
        Args:
            checkpoint_name: optional specific checkpoint to load
        """
        if checkpoint_name:
            model_path = self.checkpoint_dir / f"{checkpoint_name}.txt"
            metadata_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        else:
            checkpoint = self.get_latest_checkpoint()
            if not checkpoint:
                print("No checkpoint found.")
                return None
            model_path = checkpoint['model_path']
            metadata_path = checkpoint['metadata_path']
            
        if not model_path.exists() or not metadata_path.exists():
            print(f"Checkpoint files not found: {checkpoint_name}")
            return None
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Load model
        model = lgb.Booster(model_file=str(model_path))
        
        print(f"\nLoaded checkpoint:")
        print(f"- Model: {model_path}")
        print(f"- Metadata: {metadata_path}")
        print(f"- Epoch: {metadata['epoch']}")
        print(f"- Original timestamp: {metadata['timestamp']}")
        
        return {
            'model': model,
            'metadata': metadata
        }
        
    def list_checkpoints(self):
        """List all available checkpoints with details"""
        checkpoints = list(self.checkpoint_dir.glob(f"{self.config.model_name}*.json"))
        if not checkpoints:
            print("No checkpoints found")
            return
        
        print("\nAvailable checkpoints:")
        for cp in sorted(checkpoints, key=os.path.getmtime):
            with open(cp, 'r') as f:
                metadata = json.load(f)
            print(f"\n- Checkpoint: {cp.stem}")
            print(f"  Epoch: {metadata['epoch']}")
            print(f"  Time: {metadata['timestamp']}")
            print(f"  Metrics: {metadata['metrics']}")
            print(f"  User: {metadata['user_info']['username']}")
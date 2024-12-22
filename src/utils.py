import polars as pl
import numpy as np
from pathlib import Path
import logging
import sys
import datetime

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)

def load_partition(partition_id, base_path):
    """Load a single partition of the training data using Polars"""
    partition_path = Path(base_path) / f"partition_id={partition_id}" / "part-0.parquet"
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading partition from: {partition_path}")
    
    if not partition_path.exists():
        logger.error(f"Partition file not found: {partition_path}")
        raise FileNotFoundError(f"Partition file not found: {partition_path}")
    
    # Use Polars to read parquet file
    df = pl.read_parquet(partition_path)
    logger.info(f"Successfully loaded partition {partition_id}")
    return df

def load_training_data(base_path, partitions=None):
    """Load training data from specified partitions using Polars"""
    if partitions is None:
        partitions = range(10)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from: {base_path}")
    logger.info(f"Partitions to load: {partitions}")
    
    dfs = []
    
    for partition_id in partitions:
        try:
            df = load_partition(partition_id, base_path)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading partition {partition_id}: {e}")
            raise
    
    # Concatenate all dataframes using Polars
    final_df = pl.concat(dfs)
    return final_df
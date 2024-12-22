import polars as pl
import sys

# Function to load data
def load_data(filepath):
    df = pl.read_parquet(filepath)
    print("Loaded DataFrame Schema:", df.schema)
    return df

# Function to create features
def create_features(df):
    # Print initial columns
    print("Initial columns:", df.collect_schema().names())
    
    # Check if 'time_idx' or 'time_id' column exists
    if 'time_idx' not in df.columns and 'time_id' not in df.columns:
        raise ValueError("Column 'time_idx' or 'time_id' not found in the DataFrame")
    
    # Rename 'time_id' to 'time_idx' if necessary
    if 'time_id' in df.columns and 'time_idx' not in df.columns:
        df = df.rename({'time_id': 'time_idx'})
        print("Renamed 'time_id' to 'time_idx'")
    
    # Perform additional feature engineering steps here
    # ...
    return df

def main():
    # Path to the data
    data_path = "/shared/Dataset/train.parquet/partition_id=0/part-0.parquet"
    
    try:
        # Load data
        df = load_data(data_path)
        
        # Perform feature engineering
        df = create_features(df)
        
        print("Feature engineering completed successfully")
    
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
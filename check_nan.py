import pandas as pd
import os

def check_nan_in_feature_00(directory):
    # Traverse the directory and subdirectories to find all Parquet files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                print(f"Checking file: {file_path}")
                
                # Load the Parquet file into a DataFrame
                df = pd.read_parquet(file_path)
                
                # Check if 'feature_00' contains any NaN values
                if df['feature_00'].isnull().any():
                    print(f"NaN values found in 'feature_00' in file: {file_path}")
                else:
                    print(f"No NaN values in 'feature_00' in file: {file_path}")

# Specify the directory containing the Parquet files
directory = '/shared/Dataset/train.parquet'

# Run the NaN check
check_nan_in_feature_00(directory)
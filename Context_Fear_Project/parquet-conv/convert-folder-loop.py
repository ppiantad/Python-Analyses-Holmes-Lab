import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Define the metafolder containing all the subfolders
META_FOLDER = r"D:\Context Data\PFC Last\Raw Data\PFC alone\Raw Data"

# Convert to Path object
META_FOLDER = Path(META_FOLDER)

# Recursively find all .parquet files in the metafolder
parquet_files = META_FOLDER.rglob("*.parquet*")

# Process each .parquet file
for parquet_file in tqdm(parquet_files):
    # Read the .parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)
    
    # Define the output path (same as the input folder)
    out_path = parquet_file.with_suffix('.csv')
    
    # Save the DataFrame to a .csv file
    df.to_csv(out_path, index=False)
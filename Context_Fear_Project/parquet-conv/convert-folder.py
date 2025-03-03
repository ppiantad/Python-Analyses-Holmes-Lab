import pandas as pd
from pathlib import Path
from tqdm import tqdm

# in top right of VScode window, hit "Run Python File in Dedicated Terminal"

IN_FOLDER = r"D:\Context Data\PFC Last\Raw Data\PFC alone\Raw Data"


SAME_FOLDER = True

if not SAME_FOLDER:
    OUT_FOLDER = r""
else:
    OUT_FOLDER = IN_FOLDER

OUT_FOLDER = Path(OUT_FOLDER)

parquet_files = Path(IN_FOLDER).glob("*.parquet*")

for parquet_file in tqdm(parquet_files):
    df = pd.read_parquet(parquet_file)
    
    out_path = OUT_FOLDER / (parquet_file.name.split(".")[0] + ".csv")
    
    df.to_csv(out_path, index=False)
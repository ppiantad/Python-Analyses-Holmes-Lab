import pandas as pd

IN_FILE = r""

OUT_FILE =r""


df = pd.read_parquet(IN_FILE)

df.to_csv(OUT_FILE)

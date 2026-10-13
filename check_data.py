
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
data_folder = root / "data" / "raw"

for csv_file in data_folder.rglob("*.csv"):
    try:
        df = pd.read_csv(csv_file, nrows=5)
        print("\nFile:", csv_file.relative_to(root))
        print("Columns:", list(df.columns))
        print("Sample rows:", len(df))
    except Exception as e:
        print("Error reading", csv_file.name, ":", e)

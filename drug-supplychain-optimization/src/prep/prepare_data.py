# src/prep/prepare_data.py
import pandas as pd
import os

# define data directory relative to project root
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def load_orders():
    fn = os.path.join(DATA_DIR, "orders.csv")
    print(f"Loading orders from: {fn}")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"File not found: {fn}")

    df = pd.read_csv(fn, parse_dates=['timestamp'])
    print("✅ Loaded", len(df), "orders")
    print("Columns:", df.columns.tolist())
    print(df.head())
    return df

if __name__ == "__main__":
    df = load_orders()

    # create sample episode for dev
    sample = df.head(20)
    out_fn = os.path.join(DATA_DIR, "episode_sample.csv")
    sample.to_csv(out_fn, index=False)
    print(f"✅ Wrote {out_fn}")

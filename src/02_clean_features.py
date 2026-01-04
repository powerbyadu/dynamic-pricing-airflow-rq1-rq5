import argparse
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

def main(split: str):
    inp = PROCESSED / f"model_{split}_joined.csv"
    if not inp.exists():
        raise FileNotFoundError(f"Missing {inp}. Run 01_ingest_join.py first.")

    df = pd.read_csv(inp)

    # Datetime features (if present)
    for c in ["order_purchase_timestamp", "purchase_timestamp", "order_date"]:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce")
            df["purchase_hour"] = ts.dt.hour.fillna(-1).astype(int)
            df["purchase_dow"] = ts.dt.dayofweek.fillna(-1).astype(int)
            df["purchase_month"] = ts.dt.month.fillna(-1).astype(int)
            break

    # Fill missing values
    for col in df.columns:
        if df[col].dtype.kind in "ifc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown").astype(str)

    # Simple ratio feature
    if "price" in df.columns and "shipping_charges" in df.columns:
        df["shipping_perc_of_price"] = (df["shipping_charges"] / (df["price"] + 1e-6)).clip(0, 10)

    out = PROCESSED / f"model_{split}_features.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()
    main(args.split)

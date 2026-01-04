import argparse
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TRAIN_DIR = BASE / "data" / "train"
TEST_DIR = BASE / "data" / "test"
PROCESSED = BASE / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

FILES = {
    "customers": "df_Customers_{split}.csv",
    "orders": "df_Orders_{split}.csv",
    "orderitems": "df_OrderItems_{split}.csv",
    "payments": "df_Payments_{split}.csv",
    "products": "df_Products_{split}.csv",
}

def read_csv(folder: Path, key: str, split: str) -> pd.DataFrame:
    p = folder / FILES[key].format(split=split)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)

def main(split: str):
    folder = TRAIN_DIR if split == "train" else TEST_DIR

    customers = read_csv(folder, "customers", split)
    orders    = read_csv(folder, "orders", split)
    items     = read_csv(folder, "orderitems", split)
    payments  = read_csv(folder, "payments", split)
    products  = read_csv(folder, "products", split)

    df = orders.copy()
    if "order_id" in df.columns and "order_id" in items.columns:
        df = df.merge(items, on="order_id", how="inner")

    if "order_id" in df.columns and "order_id" in payments.columns:
        df = df.merge(payments, on="order_id", how="left")

    if "customer_id" in df.columns and "customer_id" in customers.columns:
        df = df.merge(customers, on="customer_id", how="left")

    if "product_id" in df.columns and "product_id" in products.columns:
        df = df.merge(products, on="product_id", how="left")

    out = PROCESSED / f"model_{split}_joined.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out} rows={len(df)} cols={df.shape[1]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()
    main(args.split)

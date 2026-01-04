import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"
MODEL_DIR = PROCESSED / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "payment_value"  # if this errors, we will change the target name

def main():
    df = pd.read_csv(PROCESSED / "model_train_features.csv")

    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found. Available columns: {list(df.columns)[:50]} ...")

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET], errors="ignore")

    # Drop id-like columns
    for c in ["order_id", "customer_id", "product_id", "seller_id"]:
        if c in X.columns:
            X = X.drop(columns=[c])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xva)
    metrics = {
        "MAE": float(mean_absolute_error(yva, pred)),
        "R2": float(r2_score(yva, pred)),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
    }

    pd.DataFrame([metrics]).to_csv(MODEL_DIR / "train_metrics.csv", index=False)
    joblib.dump(pipe, MODEL_DIR / "model.joblib")
    print("Saved model and metrics to data/processed/models/")

if __name__ == "__main__":
    main()

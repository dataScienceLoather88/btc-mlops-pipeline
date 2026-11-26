"""ETL steps 2-3: preprocess + feature engineering, then train and evaluate."""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.model_utils import _build_lag_features

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/btc_raw.csv")
PROCESSED_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed/btc_features.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/btc_forecaster.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2.0.0-ridge-lag")
MAX_MAPE = float(os.getenv("MAX_MAPE", "15.0"))  # promotion threshold (%)
LAGS = 7
TEST_DAYS = 7


def preprocess(raw_path: str = RAW_DATA_PATH, out_path: str = PROCESSED_PATH) -> str:
    """Clean the raw data and add technical indicators."""
    df = pd.read_csv(raw_path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Clean: drop impossible rows, fill small gaps
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)]
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx).ffill()
    df.index.name = "Date"

    # Technical indicators
    df["return_1d"] = df["Close"].pct_change()
    df["ma_7"] = df["Close"].rolling(7).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["volatility_7"] = df["return_1d"].rolling(7).std()
    df["volatility_30"] = df["return_1d"].rolling(30).std()

    df = df.dropna()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path)
    print(f"[preprocess] rows={len(df)} cols={list(df.columns)} -> {out_path}")
    return out_path


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


def train_and_evaluate(
    processed_path: str = PROCESSED_PATH,
    model_path: str = MODEL_PATH,
    max_mape: float = MAX_MAPE,
) -> dict:
    """Train on everything except the last 7 days, evaluate on those 7 days.

    The model is only written to model_path if MAPE <= max_mape.
    Returns a summary dict with 'promoted' = True/False.
    """
    df = pd.read_csv(processed_path, parse_dates=["Date"], index_col="Date")
    close = df["Close"].to_numpy(dtype=float)

    X, y = _build_lag_features(close, lags=LAGS)
    X_train, y_train = X[:-TEST_DAYS], y[:-TEST_DAYS]
    X_test, y_test = X[-TEST_DAYS:], y[-TEST_DAYS:]

    # 1) Evaluate on the held-out latest 7-day window
    candidate = Ridge(alpha=1.0).fit(X_train, y_train)
    scores = _metrics(y_test, candidate.predict(X_test))

    # 2) Refit on ALL data so the served model has seen the latest days
    final_model = Ridge(alpha=1.0).fit(X, y)

    metrics = {
        **scores,
        "samples_trained": int(len(X)),
        "test_window_days": TEST_DAYS,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    promoted = scores["mape"] <= max_mape

    if promoted:
        bundle = {
            "model": final_model,
            "lags": LAGS,
            "last_window": close[-LAGS:].tolist(),
            "metrics": metrics,
            "version": MODEL_VERSION,
        }
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        if os.path.exists(model_path):  # keep one backup for rollback
            shutil.copy(model_path, model_path + ".bak")
        tmp_path = model_path + ".tmp"
        joblib.dump(bundle, tmp_path)
        os.replace(tmp_path, model_path)  # atomic: API never reads a partial file
        print(f"[train] PROMOTED model {MODEL_VERSION} metrics={scores}")
    else:
        print(
            f"[train] REJECTED: MAPE {scores['mape']} > {max_mape}; "
            "keeping old model"
        )

    summary = {"promoted": promoted, "version": MODEL_VERSION, **metrics}
    metrics_path = os.path.join(os.path.dirname(model_path) or ".", "latest_run.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and train BTC model")
    parser.add_argument("--raw", default=RAW_DATA_PATH)
    parser.add_argument("--processed", default=PROCESSED_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--max-mape", type=float, default=MAX_MAPE)
    args = parser.parse_args()
    preprocess(args.raw, args.processed)
    train_and_evaluate(args.processed, args.model, args.max_mape)

import os
from datetime import datetime, timedelta, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_PATH = os.getenv("MODEL_PATH", "models/btc_forecaster.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0-ridge-lag")


def fetch_btc_close_series(days: int = 120) -> pd.Series:
    """Fetch daily BTC-USD closing prices via yfinance with offline fallback."""
    try:
        import yfinance as yf

        df = yf.download(
            "BTC-USD",
            period=f"{max(days, 60)}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df is not None and not df.empty:
            close = df["Close"].squeeze().dropna()
            if len(close) >= 20:
                return close.astype(float)
    except Exception:
        pass

    # Fallback deterministic daily series if network/yfinance is unavailable
    base = np.linspace(60000.0, 90000.0, days)
    wave = 1500.0 * np.sin(np.linspace(0, 8 * np.pi, days))
    idx = pd.date_range(end=datetime.now(timezone.utc).date(), periods=days, freq="D")
    return pd.Series(base + wave, index=idx, name="Close", dtype=float)


def _build_lag_features(series: np.ndarray, lags: int = 7):
    X, y = [], []
    for i in range(lags, len(series)):
        window = series[i - lags : i]
        feats = np.concatenate(
            [
                window,
                [np.mean(window), np.std(window), window[-1] - window[0]],
            ]
        )
        X.append(feats)
        y.append(series[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def train_and_save_model(model_path: str = MODEL_PATH, lags: int = 7) -> dict:
    close_series = fetch_btc_close_series(days=150)
    values = close_series.to_numpy(dtype=float)

    X, y = _build_lag_features(values, lags=lags)
    split_idx = max(1, int(len(X) * 0.85))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    if len(X_val) > 0:
        preds = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mape = float(
            np.mean(np.abs((y_val - preds) / np.clip(np.abs(y_val), 1e-8, None)))
            * 100.0
        )
    else:
        mae, rmse, mape = 0.0, 0.0, 0.0

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "samples_trained": int(len(X_train)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    bundle = {
        "model": model,
        "lags": lags,
        "last_window": values[-lags:].tolist(),
        "metrics": metrics,
        "version": MODEL_VERSION,
    }
    joblib.dump(bundle, model_path)
    return bundle


def load_or_train_bundle(model_path: str = MODEL_PATH) -> dict:
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return train_and_save_model(model_path=model_path)


def forecast_next_days(
    bundle: dict, forecast_days: int = 7, history_override=None
) -> list[dict]:
    model = bundle["model"]
    lags = int(bundle.get("lags", 7))

    if history_override is not None:
        arr = np.asarray(history_override, dtype=float)
        if arr.ndim != 1 or len(arr) < lags or not np.all(np.isfinite(arr)):
            raise ValueError(
                f"'history' must be a 1D list of at least {lags} finite numbers."
            )
        window = arr[-lags:].copy()
    else:
        window = np.asarray(bundle["last_window"], dtype=float).copy()

    start_date = datetime.now(timezone.utc).date() + timedelta(days=1)
    predictions = []

    for step in range(forecast_days):
        feats = np.concatenate(
            [
                window,
                [np.mean(window), np.std(window), window[-1] - window[0]],
            ]
        ).reshape(1, -1)
        next_price = float(model.predict(feats)[0])
        pred_date = (start_date + timedelta(days=step)).isoformat()
        predictions.append(
            {
                "date": pred_date,
                "predicted_close": round(next_price, 2),
            }
        )
        window = np.append(window[1:], next_price)

    return predictions

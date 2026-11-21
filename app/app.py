import os
import time
from flask import Flask, jsonify, request
from src.model_utils import (
    MODEL_PATH,
    MODEL_VERSION,
    forecast_next_days,
    load_or_train_bundle,
    train_and_save_model,
)

app = Flask(__name__)
START_TIME = time.time()


def _get_bundle():
    return load_or_train_bundle(MODEL_PATH)


@app.route("/health", methods=["GET"])
def health():
    try:
        bundle = _get_bundle()
        uptime_seconds = round(time.time() - START_TIME, 2)
        return (
            jsonify(
                {
                    "status": "ok",
                    "uptime_seconds": uptime_seconds,
                    "model_version": bundle.get("version", MODEL_VERSION),
                }
            ),
            200,
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        bundle = _get_bundle()
        return (
            jsonify(
                {
                    "model_version": bundle.get("version", MODEL_VERSION),
                    "metrics": bundle.get("metrics", {}),
                }
            ),
            200,
        )
    except Exception as exc:
        return (
            jsonify({"error": "Failed to retrieve model metrics", "details": str(exc)}),
            500,
        )


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request Content-Type must be application/json"}), 400

    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return jsonify({"error": "Invalid or empty JSON payload"}), 400

    forecast_days = payload.get("forecast_days", 7)
    if (
        isinstance(forecast_days, bool)
        or not isinstance(forecast_days, int)
        or not (1 <= forecast_days <= 30)
    ):
        return (
            jsonify({"error": "'forecast_days' must be an integer between 1 and 30"}),
            400,
        )

    history = payload.get("history", None)
    if history is not None and not isinstance(history, list):
        return (
            jsonify({"error": "'history' must be a list of numeric closing prices"}),
            400,
        )

    try:
        bundle = _get_bundle()
        preds = forecast_next_days(
            bundle, forecast_days=forecast_days, history_override=history
        )
        return (
            jsonify(
                {
                    "symbol": "BTC-USD",
                    "model_version": bundle.get("version", MODEL_VERSION),
                    "forecast_days": forecast_days,
                    "predictions": preds,
                }
            ),
            200,
        )
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        return jsonify({"error": "Prediction failed", "details": str(exc)}), 500


@app.route("/reload-model", methods=["POST"])
def reload_model():
    try:
        bundle = train_and_save_model(MODEL_PATH)
        return (
            jsonify(
                {
                    "status": "reloaded",
                    "model_version": bundle.get("version", MODEL_VERSION),
                    "metrics": bundle.get("metrics", {}),
                }
            ),
            200,
        )
    except Exception as exc:
        return jsonify({"error": "Model reload failed", "details": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

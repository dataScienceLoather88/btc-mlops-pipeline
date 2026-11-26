"""Daily BTC ETL: fetch -> preprocess -> train/evaluate -> reload API."""

import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

RAW_PATH = "/opt/airflow/data/raw/btc_raw.csv"
PROCESSED_PATH = "/opt/airflow/data/processed/btc_features.csv"
MODEL_PATH = "/opt/airflow/models/btc_forecaster.pkl"
MAX_MAPE = float(os.getenv("MAX_MAPE", "5.0"))
FLASK_API_URL = os.getenv("FLASK_API_URL", "http://flask-api:5000").rstrip("/")


def fetch_data():
    from src.etl_fetch import run_fetch

    run_fetch(RAW_PATH, days=730)


def preprocess_features():
    from src.train_forecast import preprocess

    preprocess(RAW_PATH, PROCESSED_PATH)


def train_evaluate():
    # Airflow loads dill, which taints pickles that Flask cannot read
    try:
        import dill

        dill.extend(False)
    except ImportError:
        pass
    from src.train_forecast import train_and_evaluate

    return train_and_evaluate(PROCESSED_PATH, MODEL_PATH, MAX_MAPE)


def reload_api(ti):
    summary = ti.xcom_pull(task_ids="train_evaluate")
    if not summary or not summary.get("promoted"):
        raise AirflowSkipException("Model not promoted; skipping reload")
    resp = requests.post(f"{FLASK_API_URL}/reload-model", timeout=60)
    resp.raise_for_status()
    print(f"[reload_api] {resp.status_code} {resp.text}")


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USD ETL, retraining and API reload",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["btc", "mlops"],
) as dag:
    t_fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    t_prep = PythonOperator(
        task_id="preprocess_features", python_callable=preprocess_features
    )
    t_train = PythonOperator(task_id="train_evaluate", python_callable=train_evaluate)
    t_reload = PythonOperator(task_id="reload_api", python_callable=reload_api)

    t_fetch >> t_prep >> t_train >> t_reload

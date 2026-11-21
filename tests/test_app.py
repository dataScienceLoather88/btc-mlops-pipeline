import pytest
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "model_version" in data


def test_metrics(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert "metrics" in data
    for key in ("mae", "rmse", "mape"):
        assert key in data["metrics"]


def test_predict_default_7_days(client):
    res = client.post("/predict", json={})
    assert res.status_code == 200
    data = res.get_json()
    assert data["forecast_days"] == 7
    assert len(data["predictions"]) == 7
    assert "date" in data["predictions"][0]
    assert "predicted_close" in data["predictions"][0]


def test_predict_custom_history(client):
    payload = {
        "forecast_days": 5,
        "history": [64000, 64500, 65100, 64800, 65500, 66000, 66250, 66500],
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["forecast_days"] == 5
    assert len(data["predictions"]) == 5


def test_predict_invalid_payload(client):
    res = client.post("/predict", json={"forecast_days": 0})
    assert res.status_code == 400

    res_short_hist = client.post("/predict", json={"history": [100, 200]})
    assert res_short_hist.status_code == 400

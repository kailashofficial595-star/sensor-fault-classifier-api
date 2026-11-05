from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

VALID = {"air_temp": 298.1, "process_temp": 308.6, "rpm": 1551, "torque": 42.8, "tool_wear": 0}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_input():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fault_probability"] <= 1.0
    assert isinstance(body["fault"], bool)


def test_predict_is_deterministic():
    assert client.post("/predict", json=VALID).json() == client.post("/predict", json=VALID).json()


def test_missing_field_is_rejected():
    bad = {k: v for k, v in VALID.items() if k != "torque"}
    assert client.post("/predict", json=bad).status_code == 422


def test_out_of_range_value_is_rejected():
    assert client.post("/predict", json={**VALID, "rpm": -5}).status_code == 422
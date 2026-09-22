from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"


def test_predict():
    payload = {
        "order": {
            "order_id": "test_order",
            "customer_id": "test_customer",
            "customer_unique_id": "test_unique_customer",
            "order_status": "delivered",
            "order_purchase_timestamp": "2018-01-15 10:30:00",
            "order_approved_at": "2018-01-15 10:40:00",
            "order_delivered_carrier_date": "2018-01-17 10:00:00",
            "order_delivered_customer_date": "2018-01-25 12:00:00",
            "order_estimated_delivery_date": "2018-01-30 00:00:00",
            "customer_zip_code_prefix": 14409,
            "customer_city": "Franca",
            "customer_state": "SP",
            "total_items": 1,
            "total_price": 100.0,
            "total_freight": 20.0,
            "unique_products": 1,
            "unique_sellers": 1,
            "total_payments": 120.0,
            "payment_count": 1,
            "payment_types": 1
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "late_probability" in data
    assert "prediction" in data
    assert "prediction_label" in data

    assert 0 <= data["late_probability"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["On-time", "Late"]


def test_predict_rejects_invalid_input():
    payload = {
        "order": {
            "customer_state": "SP"
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400

import pytest
from app.app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_home(client):
    res = client.get("/")
    assert res.status_code == 200

def test_predict(client):
    response = client.post("/predict", data={
        "delivery_count": 400,
        "inventory_level": 15000,
        "lead_time_avg": 2.5,
        "warehouse_utilization": 0.65,
        "vehicle_availability": 30,
        "fuel_price_index": 85.0,
        "economic_index": 65.5,
        "temperature": 22.0,
        "rainfall_mm": 3.0,
        "holiday_flag": 0,
        "promo_flag": 1,
        "lag_1_order": 420,
        "lag_7_order": 410,
        "lag_30_order": 395,
        "day": 15,
        "month": 6,
        "year": 2025,
        "day_of_week": 2,
        "is_weekend": 0
    })

    assert response.status_code == 200
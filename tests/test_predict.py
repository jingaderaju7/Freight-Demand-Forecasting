import pandas as pd
import pickle

def test_prediction():
    with open("model/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("model/feature_names.pkl", "rb") as f:
        features = pickle.load(f)

    # Create DataFrame instead of numpy array
    data = pd.DataFrame([[ 
        400, 15000, 2.5, 0.65, 30,
        85.0, 65.5, 22.0, 3.0,
        0, 1,
        420, 410, 395,
        15, 6, 2025, 2, 0
    ]], columns=features)

    scaled = scaler.transform(data)
    pred = model.predict(scaled)

    assert pred is not None
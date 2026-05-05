import os
import pickle

def test_model_files_exist():
    assert os.path.exists("model/random_forest_model.pkl")
    assert os.path.exists("model/scaler.pkl")

def test_model_loading():
    with open("model/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None
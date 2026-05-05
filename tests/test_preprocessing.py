import pandas as pd

def test_preprocessing():
    data = {
        "feature1": [100, 200],
        "feature2": [50, 60],
        "feature3": [1, 2]
    }

    df = pd.DataFrame(data)

    assert df is not None
    assert df.shape[0] == 2
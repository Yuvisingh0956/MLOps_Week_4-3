# tests/test_model.py
import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

@pytest.fixture(scope="module")
def model():
    return joblib.load("models/model.pkl")

@pytest.fixture(scope="module")
def data():
    return pd.read_csv("data/data.csv")

def test_model_exists(model):
    assert model is not None

def test_prediction_shape(model, data):
    # Match train.py column prep
    X = data.iloc[:, :-1]
    X.columns = [f"feature_{i}" for i in range(X.shape[1])]
    preds = model.predict(X)
    assert preds.shape[0] == len(data)

def test_accuracy_threshold(model, data):
    X = data.iloc[:, :-1]
    X.columns = [f"feature_{i}" for i in range(X.shape[1])]
    y = data.iloc[:, -1]
    preds = model.predict(X)
    # Binary/Multiclass check; accuracy threshold can be set as needed
    from sklearn.metrics import accuracy_score
    assert accuracy_score(y, preds) > 0.5

def test_no_missing_values(data):
    assert data.isnull().sum().sum() == 0

def test_target_values(data):
    # Label values as per CSV
    y = data.iloc[:, -1]
    assert set(y.unique()) <= set(["setosa", "versicolor", "virginica"])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import json
import os

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Define paths
DATA_PATH = 'data/iris.csv'
MODEL_PATH = 'models/model.pkl'
METRICS_PATH = 'metrics/metrics.json'

def train_model():
    print(f"--- Starting training using {DATA_PATH} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Have you run 'dvc checkout'?")
        return

    # Assuming the Iris dataset structure for simplicity: 
    # Last column is target, others are features (sepal length, sepal width, petal length, petal width)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Handle missing column names from raw data, assign simple names
    X.columns = [f'feature_{i}' for i in range(X.shape[1])]
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Train Model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 4. Evaluate Model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy:.4f}")

    # 5. Save Artifacts
    
    # Save Model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save Metrics
    metrics = {"accuracy": accuracy, "data_version": "V1/V2 Placeholder"}
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    train_model()

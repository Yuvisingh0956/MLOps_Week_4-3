# train_and_log_mlflow.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def train_eval(path, run_name=None, random_state=42):
    df = pd.read_csv(path)

    # Encode species label
    le = LabelEncoder()
    y = le.fit_transform(df["species"])
    X = df.drop(columns=["species", "poisoned"], errors="ignore")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("data_path", str(path))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))

        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1))

        cm = confusion_matrix(y_val, preds)
        cr = classification_report(y_val, preds, target_names=le.classes_, output_dict=False)

        with open("report.txt", "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 (macro): {f1:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\nClassification Report:\n")
            f.write(cr)

        mlflow.log_artifact("report.txt")
        mlflow.sklearn.log_model(clf, "model")
        print(f"✅ Trained on {path} | Accuracy={acc:.4f} | F1={f1:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args()

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    for ds in args.datasets:
        run_name = ds.split("/")[-1].replace(".csv", "")
        train_eval(ds, run_name=run_name)

if __name__ == "__main__":
    main()

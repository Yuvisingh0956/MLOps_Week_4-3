# plot_poison_effects.py
import mlflow
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import re

def extract_poison_level(run_name):
    """
    Extract poison percentage from run name.
    Expected names:
        iris_poison_5pct_random
        iris_poison_10pct_random
        iris_poison_50pct_random
    """
    match = re.search(r"(\d+)pct", run_name)
    if match:
        return int(match.group(1))
    return 0   # 0% for clean dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--out", default="poison_results.png")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()

    # Get experiment
    exp = client.get_experiment_by_name(args.experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{args.experiment_name}' not found!")

    runs = client.search_runs(exp.experiment_id)

    records = []

    for r in runs:
        run_name = r.data.tags.get("mlflow.runName", "unknown")
        poison_pct = extract_poison_level(run_name)
        acc = r.data.metrics.get("accuracy", None)
        f1 = r.data.metrics.get("f1_macro", None)

        if acc is not None and f1 is not None:
            records.append({
                "poison_pct": poison_pct,
                "accuracy": acc,
                "f1_macro": f1
            })

    df = pd.DataFrame(records)
    df = df.sort_values("poison_pct")

    print("\n=== Extracted Results ===")
    print(df)

    # ---- PLOT ----
    plt.figure(figsize=(10, 6))
    plt.plot(df["poison_pct"], df["accuracy"], "-o", label="Accuracy")
    plt.plot(df["poison_pct"], df["f1_macro"], "-o", label="F1 Macro")
    plt.xlabel("Poison Level (%)")
    plt.ylabel("Score")
    plt.title("Impact of Poisoning on IRIS Model Performance")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.savefig(args.out, dpi=300)
    print(f"\n📁 Plot saved as: {args.out}")

if __name__ == "__main__":
    main()

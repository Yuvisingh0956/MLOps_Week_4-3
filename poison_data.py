# poison_data.py
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def poison_df(df, fraction, strategy="random"):
    n = len(df)
    k = int(n * fraction)
    dfp = df.copy().reset_index(drop=True)

    if k == 0:
        return dfp

    idx = np.random.choice(dfp.index, size=k, replace=False)

    # Poison strategies
    if strategy == "random":
        for col in dfp.columns:
            if col == "species":
                continue
            col_min, col_max = dfp[col].min(), dfp[col].max()
            dfp.loc[idx, col] = np.random.uniform(col_min, col_max, size=k)
    elif strategy == "gaussian":
        for col in dfp.columns:
            if col == "species":
                continue
            mu, sigma = dfp[col].mean(), dfp[col].std() * 5.0
            dfp.loc[idx, col] = np.random.normal(mu, sigma, size=k)
    elif strategy == "label_flip":
        classes = dfp["species"].unique()
        dfp.loc[idx, "species"] = np.random.choice(classes, size=k)
    else:
        raise ValueError("Unknown strategy")

    dfp["poisoned"] = False
    dfp.loc[idx, "poisoned"] = True
    return dfp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to clean CSV file")
    parser.add_argument("--out-dir", default="data/poisoned")
    parser.add_argument("--fractions", default="0.05,0.1,0.5")
    parser.add_argument("--strategy", default="random")
    args = parser.parse_args()

    inp = Path(args.input)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    fracs = [float(x) for x in args.fractions.split(",")]

    for f in fracs:
        dfp = poison_df(df, f, strategy=args.strategy)
        out_path = outdir / f"iris_poison_{int(f*100)}pct_{args.strategy}.csv"
        dfp.to_csv(out_path, index=False)
        print(f"✅ Wrote: {out_path}")

if __name__ == "__main__":
    main()

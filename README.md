Here’s a properly formatted `README.md` for your assignment, using clear Markdown sections, tables, and command formatting to ensure excellent readability and reproducibility.

***

# IRIS Dataset Data Poisoning Impact Analysis

## 🎯 Objective

This assignment evaluates the impact of data poisoning on an IRIS classification model and demonstrates how to:
- Analyze model validation outcomes as poisoning percentage changes (5%, 10%, 50%)
- Use MLflow to log and compare all experiments
- Detect and mitigate poisoning attacks
- Assess how required dataset quantity changes when quality decreases

**Key tasks:** Dataset poisoning, model training, MLflow experiment logging, and results visualization.

***

## 🧬 Dataset Description

This project uses the IRIS dataset, which contains:

| sepal_length | sepal_width | petal_length | petal_width | species                         |
| ------------ | ----------- | ------------ | ----------- | ------------------------------- |
| numeric      | numeric     | numeric      | numeric     | setosa / versicolor / virginica |

- **Features:** 4 numeric columns
- **Label:** `species`

***

## 🧪 Data Poisoning Method

Three poisoned versions of the IRIS dataset are generated:
- **5% poisoning**
- **10% poisoning**
- **50% poisoning**

Each version randomly corrupts feature values for a subset of rows (5%, 10%, or 50%), using random values within each feature’s min–max range.

**Additional:** Each dataset has a boolean column `poisoned`, indicating manipulated rows.

### Generate Datasets

```bash
python poison_data.py --input data/iris.csv --out-dir data/poisoned --fractions 0.05,0.1,0.5
```

**Output files:**
- `iris_poison_5pct_random.csv`
- `iris_poison_10pct_random.csv`
- `iris_poison_50pct_random.csv`

***

## 🤖 Model Training & MLflow Logging

A Random Forest Classifier is trained on:
- Clean dataset
- 5% poisoned
- 10% poisoned
- 50% poisoned

**For every run, MLflow tracks:**
- ⚙️ Hyperparameters
- 📊 Metrics (Accuracy, F1-Macro)
- 🧩 Confusion Matrix (as artifact)
- 📝 Classification Report
- 🧠 Saved Model

### Run Training & Logging

```bash
python train_and_log_mlflow.py \
    --datasets data/iris.csv \
               data/poisoned/iris_poison_5pct_random.csv \
               data/poisoned/iris_poison_10pct_random.csv \
               data/poisoned/iris_poison_50pct_random.csv \
    --mlflow-tracking-uri http://127.0.0.1:5000
```

**Start MLflow UI:**

```bash
mlflow ui --port 5000
```

***

## 📊 Visualization of Poisoning Effects

The script extracts MLflow results and plots the following:
- Accuracy vs Poison %
- F1 Score vs Poison %

### Generate Performance Plots

```bash
python plot_poison_effects.py \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --experiment-name Default \
  --out poison_plot.png
```

**Plot:** Shows classification performance drops as poisoning increases.

***

## 🔎 Key Observations

- 5–10% poisoning → mild accuracy drop
- 50% poisoning → model performs close to randomly
- Confusion matrices reveal increased mixing among species with more poisoning

***

## 🔐 Mitigation Strategies

- **Data Validation:** Validate feature ranges, missing values, and unusual patterns
- **Outlier Detection:** Use Isolation Forest or Z-score filtering for anomaly removal
- **Data Provenance Tracking:** Restrict data ingestion from untrusted sources
- **Label Verification:** Manually or semi-automatically confirm labels
- **Model-Level Defenses:** Use robust or ensemble models
- **Adversarial Poison Detection:** Detect poisoned instances before model training

***

## 📦 Data Quantity vs Quality

As quality drops, effective clean data $$N_{clean}$$ reduces:  
$$
N_{clean} = (1 - p) \times N
$$  
To maintain performance:  
$$
N_{required} = \frac{N}{1 - p}
$$  

| Poison % | Effective Clean Data | Required Data Multiplier        |
| -------- | -------------------- | ------------------------------ |
| **5%**   | 95%                  | Need ~1.05× more data          |
| **10%**  | 90%                  | Need ~1.11× more data          |
| **50%**  | 50%                  | Need **2×** more clean data    |

- More data helps against random poisoning.
- Targeted attacks demand advanced detection, not just more data.

***

## ✅ Conclusion

- Poisoning degrades ML model accuracy
- Multiple poisoning levels are easy to generate and track
- MLflow enables experiment comparison and tracking
- Visualizations highlight performance decline and tradeoffs
- Mitigation requires both better validation/detection and consideration of data quantity vs quality

**This pipeline is fully reproducible and scalable.**

***

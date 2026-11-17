# Customer Churn Analysis — Telco

End-to-end exploratory data analysis and a logistic-regression model that predicts
customer churn for a telecom provider and surfaces the **main drivers of churn**.

Built in Python (pandas, scikit-learn, matplotlib, seaborn) on the public
[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset — 7,043 customers with 20 attributes covering demographics, account
details and the services each customer subscribes to.

## Problem

Retaining an existing customer is far cheaper than acquiring a new one. The goal
is to (1) understand **who** churns and **why**, and (2) build a model that flags
at-risk customers so retention effort can be targeted.

## Results

Logistic regression on a held-out 20% test set (1,409 customers), with class
weighting to handle the 26.5% / 73.5% churn imbalance:

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.842** |
| Recall (churners) | 0.783 |
| Precision (churners) | 0.504 |
| F1 (churners) | 0.614 |
| Accuracy | 0.738 |

The model is tuned to **catch churners** — it correctly identifies ~78% of
customers who actually leave, which is the metric that matters for a retention
campaign (missing a churner costs more than a false alarm).

### Key churn drivers

Ranked from the model's coefficients (odds ratios in brackets):

**Increase churn risk**
- Fiber-optic internet service (×3.4 odds)
- High total charges
- Streaming TV / movies add-ons
- Electronic-check payment method

**Reduce churn risk**
- Two-year contract (×0.24 odds) and one-year contract
- Longer tenure
- Having phone service

The headline takeaway: **month-to-month, fiber-optic customers paying by
electronic check are the highest-risk segment**, while locking customers into
longer contracts is the single strongest retention lever.

## Figures

All generated to `outputs/figures/`:

| | |
|---|---|
| Churn distribution | Churn rate by contract |
| Tenure by churn | Monthly charges by churn |
| Churn by key services | Confusion matrix |
| ROC curve | Top churn drivers |

![Top churn drivers](outputs/figures/08_churn_drivers.png)

## Project structure

```
telco-customer-churn/
├── data/
│   └── Telco-Customer-Churn.csv     # raw public dataset (7,043 rows)
├── src/
│   └── churn_analysis.py            # full pipeline: clean -> EDA -> model
├── notebooks/
│   └── churn_analysis.ipynb         # same analysis, annotated step by step
├── outputs/
│   ├── figures/                     # EDA + model charts (.png)
│   ├── metrics.json                 # model performance
│   └── churn_drivers.csv            # ranked feature coefficients
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. run the full pipeline
python src/churn_analysis.py
```

This cleans the data, writes all EDA/model figures to `outputs/figures/`, saves
metrics to `outputs/metrics.json`, and ranks every feature in
`outputs/churn_drivers.csv`.

## Method

1. **Clean** — coerce `TotalCharges` to numeric (blank for new customers at
   tenure 0), drop the non-predictive `customerID`, encode the target.
2. **EDA** — churn balance, and churn rate broken down by contract, tenure,
   monthly charges and the main service features.
3. **Feature engineering** — one-hot encode categoricals (`drop_first` to avoid
   collinearity); standard-scale the three continuous features.
4. **Model** — logistic regression (`class_weight="balanced"`, `max_iter=1000`),
   stratified 80/20 split, `random_state=42` for reproducibility.
5. **Evaluate** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
   and coefficient/odds-ratio analysis to interpret the drivers.

## Data source

IBM sample "Telco Customer Churn" dataset, widely mirrored (e.g. on
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)). Each
row is one customer; the `Churn` column marks whether they left in the last month.

---

**Hitesh Bharambe** · [GitHub](https://github.com/HiteshBharambe99) · [LinkedIn](https://linkedin.com/in/hitesh-bharambe)

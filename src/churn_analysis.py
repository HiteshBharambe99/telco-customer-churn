"""
Telco customer churn analysis.

Cleans the data, does some EDA, then fits a logistic regression to predict
churn and read off which factors drive it. Writes charts, metrics and the
ranked drivers to outputs/.

    python src/churn_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
PALETTE = ["#4C72B0", "#DD8452"]


# --------------------------------------------------------------------------- #
# 1. Load & clean
# --------------------------------------------------------------------------- #
def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # TotalCharges loads as text because new customers (tenure 0) have it blank.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # 0/1 -> Yes/No just so it reads nicely in the EDA plots.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])  # just an ID, no signal

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Overall churn rate: {df['Churn'].mean():.1%}")
    return df


# --------------------------------------------------------------------------- #
# 2. Exploratory data analysis
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame) -> None:
    # Churn balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["Churn"].map({0: "Retained", 1: "Churned"}).value_counts()
    ax.bar(counts.index, counts.values, color=PALETTE)
    ax.set_title("Churn distribution")
    ax.set_ylabel("Customers")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, f"{v:,}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_churn_distribution.png", dpi=120)
    plt.close(fig)

    # Churn rate by contract type
    fig, ax = plt.subplots(figsize=(6, 4))
    rate = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    ax.bar(rate.index, rate.values * 100, color="#4C72B0")
    ax.set_title("Churn rate by contract type")
    ax.set_ylabel("Churn rate (%)")
    for i, v in enumerate(rate.values):
        ax.text(i, v * 100 + 1, f"{v:.0%}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_churn_by_contract.png", dpi=120)
    plt.close(fig)

    # Tenure distribution split by churn
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, grp in df.groupby("Churn"):
        ax.hist(
            grp["tenure"],
            bins=30,
            alpha=0.6,
            label="Churned" if label else "Retained",
            color=PALETTE[label],
        )
    ax.set_title("Tenure distribution by churn")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_tenure_by_churn.png", dpi=120)
    plt.close(fig)

    # Monthly charges split by churn
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, grp in df.groupby("Churn"):
        sns.kdeplot(
            grp["MonthlyCharges"],
            ax=ax,
            fill=True,
            alpha=0.4,
            label="Churned" if label else "Retained",
            color=PALETTE[label],
        )
    ax.set_title("Monthly charges by churn")
    ax.set_xlabel("Monthly charges ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_monthly_charges_by_churn.png", dpi=120)
    plt.close(fig)

    # Churn rate across a few key categorical services
    cat_cols = ["InternetService", "TechSupport", "OnlineSecurity", "PaymentMethod"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), cat_cols):
        rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        ax.bar(rate.index, rate.values * 100, color="#4C72B0")
        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("Churn rate (%)")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_churn_by_services.png", dpi=120)
    plt.close(fig)

    print(f"Saved EDA figures to {FIG_DIR}")


# --------------------------------------------------------------------------- #
# 3. Feature engineering & preprocessing
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame):
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # one-hot the categoricals, drop_first to avoid collinearity
    X = pd.get_dummies(X, drop_first=True)

    numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    return X, y, numeric


# --------------------------------------------------------------------------- #
# 4 & 5. Train, model, evaluate
# --------------------------------------------------------------------------- #
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, numeric: list[str]) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # scale the numeric columns only (fit on train, apply to test)
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric] = scaler.fit_transform(X_train[numeric])
    X_test[numeric] = scaler.transform(X_test[numeric])

    # balanced class weight because only ~27% of customers churn
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print("\nModel performance (held-out test set):")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"], ax=ax,
    )
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#4C72B0", label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title("ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_roc_curve.png", dpi=120)
    plt.close(fig)

    # Coefficient analysis -> churn drivers
    coefs = (
        pd.DataFrame({"feature": X.columns, "coefficient": model.coef_[0]})
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )
    coefs["odds_ratio"] = np.exp(coefs["coefficient"]).round(3)
    coefs["coefficient"] = coefs["coefficient"].round(4)
    coefs.to_csv(OUT_DIR / "churn_drivers.csv", index=False)

    top_up = coefs.head(10)
    top_down = coefs.tail(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    drivers = pd.concat([top_up, top_down])
    colors = ["#DD8452" if c > 0 else "#4C72B0" for c in drivers["coefficient"]]
    ax.barh(drivers["feature"], drivers["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top churn drivers (logistic-regression coefficients)")
    ax.set_xlabel("Coefficient  (positive = more churn)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_churn_drivers.png", dpi=120)
    plt.close(fig)

    print("\nTop 5 features increasing churn risk:")
    for _, r in top_up.head(5).iterrows():
        print(f"  {r['feature']:35s} coef={r['coefficient']:+.3f}  OR={r['odds_ratio']}")
    print("\nTop 5 features reducing churn risk:")
    for _, r in top_down.head(5).iterrows():
        print(f"  {r['feature']:35s} coef={r['coefficient']:+.3f}  OR={r['odds_ratio']}")

    return metrics


def main() -> None:
    df = load_and_clean()
    run_eda(df)
    X, y, numeric = build_features(df)
    metrics = train_and_evaluate(X, y, numeric)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()

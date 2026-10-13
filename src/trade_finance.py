
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "raw" / "trade_finance"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATA_FILE = DATA_DIR / "trade_finance.csv"
MODEL_FILE = MODEL_DIR / "trade_finance_model.joblib"
REPORT_FILE = REPORT_DIR / "trade_finance_evaluation.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "invoice_amount",
    "payment_terms_days",
    "buyer_credit_score",
    "shipment_delay_days",
    "country_risk_score",
    "previous_defaults",
]

TARGET = "high_risk"


def create_sample_dataset(rows=3000):
    data = pd.DataFrame({
        "invoice_amount": pd.Series(
            [1000 + (i * 7919) % 99000 for i in range(rows)],
            dtype=float,
        ),
        "payment_terms_days": [
            [15, 30, 45, 60, 90][i % 5]
            for i in range(rows)
        ],
        "buyer_credit_score": [
            300 + (i * 37) % 551
            for i in range(rows)
        ],
        "shipment_delay_days": [
            (i * 7) % 31
            for i in range(rows)
        ],
        "country_risk_score": [
            (i * 13) % 101
            for i in range(rows)
        ],
        "previous_defaults": [
            int((i * 11) % 7 == 0)
            for i in range(rows)
        ],
    })

    risk_score = (
        (data["buyer_credit_score"] < 580).astype(int) * 2
        + (data["shipment_delay_days"] > 10).astype(int)
        + (data["country_risk_score"] > 65).astype(int)
        + data["previous_defaults"] * 3
        + (data["payment_terms_days"] >= 60).astype(int)
        + (data["invoice_amount"] > 75000).astype(int)
    )

    data[TARGET] = (risk_score >= 3).astype(int)
    data.to_csv(DATA_FILE, index=False)

    print("Synthetic demonstration dataset created.")


print("=" * 55)
print("LOGISCHAIN AI - TRADE FINANCE RISK")
print("=" * 55)

if not DATA_FILE.exists():
    create_sample_dataset()

df = pd.read_csv(DATA_FILE)

required = set(FEATURES + [TARGET])
missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns in {DATA_FILE.name}: {sorted(missing)}"
    )

df = df[FEATURES + [TARGET]].copy()

for column in FEATURES + [TARGET]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna()

if df[TARGET].nunique() != 2:
    raise ValueError(
        "The high_risk target must contain both 0 and 1."
    )

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("The high_risk target must contain only 0 and 1.")

X = df[FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=80,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=2,
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    zero_division=0,
)

joblib.dump(
    {
        "model": model,
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "dataset_type": (
            "synthetic demonstration data"
            if not DATA_FILE.exists()
            else "local dataset; verify its provenance"
        ),
    },
    MODEL_FILE,
)

REPORT_FILE.write_text(
    f"Dataset rows: {len(df)}\n"
    f"Accuracy: {accuracy:.4f}\n\n"
    f"{report}",
    encoding="utf-8",
)

print("\nTraining completed.")
print(f"Demonstration accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(report)

print(f"Model saved: {MODEL_FILE}")
print(f"Evaluation saved: {REPORT_FILE}")
print("\nTrade finance module ready.")

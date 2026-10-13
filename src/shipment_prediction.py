
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT / "data" / "raw" / "logistics"
    / "DataCoSupplyChainDataset.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, encoding="latin1", low_memory=False)

target = "Late_delivery_risk"

if target not in df.columns:
    raise ValueError(
        f"Target '{target}' not found. Available columns: {list(df.columns)}"
    )

features = [
    "Type",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Category Name",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Region",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Shipping Mode",
]

features = [col for col in features if col in df.columns]

df = df.dropna(subset=[target])
X = df[features].copy()
y = pd.to_numeric(df[target], errors="coerce")

valid = y.isin([0, 1])
X = X.loc[valid]
y = y.loc[valid].astype(int)

if y.nunique() != 2:
    raise ValueError("The target must contain both classes: 0 and 1.")

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = [
    col for col in X.columns if col not in numeric_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training shipment-risk model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Evaluation")
print("----------------")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Features used: {len(features)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

model_path = MODEL_DIR / "shipment_risk_model.joblib"
joblib.dump({
    "model": model,
    "feature_columns": features,
    "target_column": target,
}, model_path)

print(f"\nModel saved to: {model_path}")

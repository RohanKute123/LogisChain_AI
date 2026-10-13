
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT
    / "data"
    / "raw"
    / "demand_forecasting"
    / "train.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "demand_forecast_model.joblib"
EVALUATION_FILE = REPORT_DIR / "demand_forecast_evaluation.csv"
FORECAST_FILE = REPORT_DIR / "next_7_day_demand_forecast.csv"

print("=" * 55)
print("LOGISCHAIN AI - DEMAND FORECASTING")
print("=" * 55)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}\n"
        "Check that train.csv is inside "
        "data/raw/demand_forecasting/"
    )

print("\nLoading dataset...")
df = pd.read_csv(DATA_FILE)

required_columns = {"date", "store", "item", "sales"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}\n"
        f"Available columns: {list(df.columns)}"
    )

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

df = df.dropna(
    subset=["date", "store", "item", "sales"]
).copy()

df["store"] = df["store"].astype(str)
df["item"] = df["item"].astype(str)

df = df.sort_values("date").reset_index(drop=True)

if len(df) < 20:
    raise ValueError(
        "The dataset has too few valid rows for forecasting."
    )

print(f"Valid dataset rows: {len(df):,}")
print(f"First date: {df['date'].min().date()}")
print(f"Last date:  {df['date'].max().date()}")


def create_features(data):
    result = data.copy()

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    result["dayofweek"] = result["date"].dt.dayofweek
    result["weekofyear"] = (
        result["date"].dt.isocalendar().week.astype(int)
    )
    result["is_weekend"] = (
        result["date"].dt.dayofweek >= 5
    ).astype(int)

    return result


df = create_features(df)

features = [
    "store",
    "item",
    "year",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "is_weekend",
]

categorical_features = ["store", "item"]

numeric_features = [
    "year",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "is_weekend",
]

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "Unable to create training and testing datasets."
    )

print(f"\nTraining rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

MAX_TRAIN_ROWS = 200000

if len(train_df) > MAX_TRAIN_ROWS:
    print(
        f"Limiting training to {MAX_TRAIN_ROWS:,} "
        "randomly selected rows..."
    )

    train_df = train_df.sample(
        n=MAX_TRAIN_ROWS,
        random_state=42,
    ).sort_values("date")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_features,
        ),
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0)),
])

print("\nTraining Ridge Regression model...")

model.fit(
    train_df[features],
    train_df["sales"],
)

print("Training completed successfully.")

print("\nEvaluating model...")

predictions = model.predict(test_df[features])
predictions = predictions.clip(min=0)

mae = mean_absolute_error(
    test_df["sales"],
    predictions,
)

rmse = mean_squared_error(
    test_df["sales"],
    predictions,
) ** 0.5

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

evaluation = pd.DataFrame({
    "date": test_df["date"].values,
    "store": test_df["store"].values,
    "item": test_df["item"].values,
    "actual_sales": test_df["sales"].values,
    "predicted_sales": predictions,
})

evaluation.to_csv(
    EVALUATION_FILE,
    index=False,
)

print(f"\nEvaluation saved: {EVALUATION_FILE}")

print("\nSaving trained model...")

joblib.dump(
    {
        "model": model,
        "feature_columns": features,
        "target_column": "sales",
        "model_name": "Ridge Regression",
    },
    MODEL_FILE,
)

print(f"Model saved: {MODEL_FILE}")

print("\nGenerating next seven days of forecasts...")

last_date = df["date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7,
    freq="D",
)

store_items = df[
    ["store", "item"]
].drop_duplicates()

future = store_items.merge(
    pd.DataFrame({"date": future_dates}),
    how="cross",
)

future = create_features(future)

future_predictions = model.predict(
    future[features]
)

future["predicted_sales"] = (
    future_predictions.clip(min=0).round(2)
)

future = future[
    [
        "date",
        "store",
        "item",
        "predicted_sales",
    ]
]

future.to_csv(
    FORECAST_FILE,
    index=False,
)

print(f"Forecast saved: {FORECAST_FILE}")

print("\n" + "=" * 55)
print("DEMAND FORECASTING COMPLETED")
print("=" * 55)

print("\nGenerated files:")
print("1. models/demand_forecast_model.joblib")
print("2. reports/demand_forecast_evaluation.csv")
print("3. reports/next_7_day_demand_forecast.csv")

print("\nForecast preview:")
print(future.head(10).to_string(index=False))

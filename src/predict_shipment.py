
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "shipment_risk_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run src/shipment_prediction.py first."
    )

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["feature_columns"]

shipment = {
    "Type": "DEBIT",
    "Days for shipment (scheduled)": 4,
    "Benefit per order": 25.0,
    "Sales per customer": 150.0,
    "Category Name": "Electronics",
    "Customer Segment": "Consumer",
    "Department Name": "Technology",
    "Market": "Pacific Asia",
    "Order Region": "Southeast Asia",
    "Order Item Discount": 10.0,
    "Order Item Discount Rate": 0.05,
    "Order Item Product Price": 200.0,
    "Order Item Quantity": 2,
    "Sales": 400.0,
    "Order Item Total": 390.0,
    "Order Profit Per Order": 35.0,
    "Shipping Mode": "Standard Class",
}

input_df = pd.DataFrame([shipment])[features]

prediction = int(model.predict(input_df)[0])

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    risk_probability = float(probabilities[classes.index(1)])
else:
    risk_probability = float(prediction)

print("\n===== LogisChain AI =====")
print("Shipment Delivery Risk")
print("=======================")
print("Prediction:", "At Risk" if prediction == 1 else "Lower Risk")
print(f"Estimated risk probability: {risk_probability:.1%}")
print("=======================")
print("Note: This is a prototype prediction, not a guarantee.")

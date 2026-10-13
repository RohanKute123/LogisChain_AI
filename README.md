# LogisChain AI
### Predictive Trade Finance & Logistics Valuation

LogisChain AI is an AI-powered logistics and trade-finance analytics platform designed to predict shipment delivery risks, forecast product demand, and assess trade-finance risk through machine learning.

## Key Features

- **Shipment Risk Prediction:** Predicts the likelihood of late shipment delivery.
- **Demand Forecasting:** Forecasts product demand using historical sales data.
- **Trade Finance Risk Analysis:** Estimates financial risk using invoice, payment, buyer-credit, and shipment-delay information.
- **Interactive Dashboard:** Displays model status, analytics, and forecasting results.
- **Prediction APIs:** Provides endpoints for shipment and trade-finance predictions.
- **Report Downloads:** Supports downloading available forecast and evaluation reports.

## Technology Stack

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- Plotly
- HTML and CSS
- Machine Learning

## Project Structure

```text
LogisChain_AI/
├── app.py
├── data/
│   └── raw/
│       ├── demand_forecasting/
│       ├── logistics/
│       └── trade_finance/
├── models/
│   ├── shipment_risk_model.joblib
│   ├── demand_forecast_model.joblib
│   └── trade_finance_model.joblib
├── reports/
│   ├── demand_forecast_evaluation.csv
│   ├── next_7_day_demand_forecast.csv
│   └── trade_finance_evaluation.txt
└── src/
    ├── shipment_prediction.py
    ├── predict_shipment.py
    ├── demand_forecasting.py
    └── trade_finance.py
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd LogisChain_AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment on Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install flask pandas scikit-learn joblib plotly
```

### 5. Run the application

```bash
python app.py
```

Open the local dashboard at:

http://127.0.0.1:5000

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Main dashboard |
| `/api/health` | GET | Check application and model status |
| `/api/forecast-data` | GET | Retrieve forecast data |
| `/api/shipment-predict` | POST | Predict shipment risk |
| `/api/trade-finance-predict` | POST | Predict trade-finance risk |

## Machine Learning Modules

### Shipment Risk Prediction
Uses historical logistics data to classify shipment delivery risk.

### Demand Forecasting
Uses historical store-item sales information to forecast demand.

### Trade Finance Risk
Estimates risk using financial and shipment-related input features.

## Data and Limitations

Model performance depends on data quality, feature availability, and evaluation methodology. Synthetic trade-finance data is intended for demonstration and development only. It does not establish real-world financial predictive accuracy.

This project is an educational prototype and should not be used as the sole basis for operational logistics or financial decisions.

## Future Enhancements

- Real-time shipment tracking integration
- Live logistics and demand data pipelines
- Explainable AI for prediction results
- Interactive geographic risk visualization
- Automated model monitoring
- User authentication and role-based access

## Author

**Rohan Kute**

B.Sc. Data Science

**Project:** LogisChain AI — Predictive Trade Finance & Logistics Valuation
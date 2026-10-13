
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, Response

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

MODEL_FILES = {
    "shipment": ROOT / "models" / "shipment_risk_model.joblib",
    "demand": ROOT / "models" / "demand_forecast_model.joblib",
    "trade": ROOT / "models" / "trade_finance_model.joblib",
}

FILES = {
    "logistics": ROOT / "data" / "raw" / "logistics" / "DataCoSupplyChainDataset.csv",
    "forecast": ROOT / "reports" / "next_7_day_demand_forecast.csv",
    "demand_evaluation": ROOT / "reports" / "demand_forecast_evaluation.csv",
    "trade_evaluation": ROOT / "reports" / "trade_finance_evaluation.txt",
}

models = {}

for name, path in MODEL_FILES.items():
    try:
        if path.exists():
            models[name] = joblib.load(path)
            print(f"{name.title()} model loaded.")
        else:
            print(f"{name.title()} model not found: {path}")
    except Exception as exc:
        print(f"Could not load {name} model: {exc}")


def read_csv_safe(path):
    try:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
    return pd.DataFrame()


def logistics_path():
    path = FILES["logistics"]

    if path.exists():
        return path

    candidates = list(
        (ROOT / "data").rglob("DataCoSupplyChainDataset.csv")
    )

    return candidates[0] if candidates else path


def get_logistics_rows():
    path = logistics_path()

    if not path.exists():
        return 0

    try:
        with path.open("rb") as file:
            return max(sum(1 for _ in file) - 1, 0)
    except OSError:
        return 0


def get_forecast_data():
    df = read_csv_safe(FILES["forecast"])

    if df.empty:
        return df, None, None

    date_col = next(
        (c for c in ["date", "forecast_date", "ds"] if c in df.columns),
        None,
    )

    value_col = next(
        (
            c for c in [
                "forecast_sales",
                "predicted_sales",
                "sales",
                "predicted_demand",
                "forecast",
            ]
            if c in df.columns
        ),
        None,
    )

    if value_col is None:
        candidates = [
            c for c in df.columns
            if c.lower() not in {
                "date", "forecast_date", "ds", "store", "item"
            }
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        value_col = candidates[-1] if candidates else None

    return df, date_col, value_col


def get_estimator(bundle):
    if isinstance(bundle, dict):
        return bundle.get("model", bundle)
    return bundle


def get_features(bundle):
    if isinstance(bundle, dict):
        return bundle.get("feature_columns", [])
    return []


def get_positive_probability(estimator, df):
    if not hasattr(estimator, "predict_proba"):
        return None

    probabilities = estimator.predict_proba(df)[0]
    classes = list(estimator.classes_)

    if 1 not in classes:
        return None

    return float(probabilities[classes.index(1)])


def predict_shipment(form):
    bundle = models.get("shipment")

    if bundle is None:
        return None, "Shipment model is not loaded."

    try:
        estimator = get_estimator(bundle)
        features = get_features(bundle)

        if not features:
            return None, "Shipment model feature columns are unavailable."

        defaults = {
            "Type": "DEBIT",
            "Days for shipment (scheduled)": 3,
            "Benefit per order": 0,
            "Sales per customer": 100,
            "Category Name": "Sporting Goods",
            "Customer Segment": "Consumer",
            "Department Name": "Fitness",
            "Market": "USCA",
            "Order Region": "West of USA",
            "Order Item Discount": 0,
            "Order Item Discount Rate": 0,
            "Order Item Product Price": 100,
            "Order Item Quantity": 1,
            "Sales": 100,
            "Order Item Total": 100,
            "Order Profit Per Order": 10,
            "Shipping Mode": "Standard Class",
        }

        numeric_fields = {
            "Days for shipment (scheduled)",
            "Benefit per order",
            "Sales per customer",
            "Order Item Discount",
            "Order Item Discount Rate",
            "Order Item Product Price",
            "Order Item Quantity",
            "Sales",
            "Order Item Total",
            "Order Profit Per Order",
        }

        row = {}

        for feature in features:
            default = defaults.get(feature, 0)
            raw_value = form.get(feature, default)

            if feature in numeric_fields:
                value = default if raw_value == "" else float(raw_value)

                if value < 0:
                    return None, f"{feature} cannot be negative."

                row[feature] = value
            else:
                row[feature] = raw_value if raw_value not in (None, "") else default

        df = pd.DataFrame([row], columns=features)
        prediction = int(estimator.predict(df)[0])
        probability = get_positive_probability(estimator, df)

        return {
            "prediction": prediction,
            "label": "High delivery risk" if prediction == 1 else "Low delivery risk",
            "probability": probability,
            "inputs": row,
        }, None

    except Exception as exc:
        return None, f"Shipment prediction failed: {exc}"


def predict_trade_finance(form):
    bundle = models.get("trade")

    if bundle is None:
        return None, "Trade finance model is not loaded."

    try:
        estimator = get_estimator(bundle)
        features = get_features(bundle)

        if not features:
            return None, "Trade finance model feature columns are unavailable."

        defaults = {
            "invoice_amount": 50000,
            "payment_terms_days": 30,
            "buyer_credit_score": 700,
            "shipment_delay_days": 0,
            "country_risk_score": 30,
            "previous_defaults": 0,
        }

        row = {}

        for feature in features:
            default = defaults.get(feature, 0)
            raw_value = form.get(feature, default)
            value = default if raw_value == "" else float(raw_value)

            if value < 0:
                return None, f"{feature} cannot be negative."

            row[feature] = value

        if row.get("buyer_credit_score", 0) > 900:
            return None, "Buyer credit score must be between 0 and 900."

        if row.get("country_risk_score", 0) > 100:
            return None, "Country risk score must be between 0 and 100."

        previous_defaults = row.get("previous_defaults", 0)

        if not previous_defaults.is_integer():
            return None, "Previous defaults must be a whole number."

        df = pd.DataFrame([row], columns=features)
        prediction = int(estimator.predict(df)[0])
        probability = get_positive_probability(estimator, df)

        return {
            "prediction": prediction,
            "label": "High risk" if prediction == 1 else "Lower risk",
            "probability": probability,
            "inputs": row,
        }, None

    except Exception as exc:
        return None, f"Trade finance prediction failed: {exc}"


BASE_STYLE = """
<style>
*{box-sizing:border-box}
body{margin:0;background:#0b1120;color:#e5e7eb;font-family:Arial,sans-serif}
.layout{display:flex;min-height:100vh}
.sidebar{width:235px;flex-shrink:0;background:#111a2e;padding:26px 18px;border-right:1px solid #26334b}
.brand{font-size:21px;font-weight:bold;color:#60a5fa;margin-bottom:8px}
.subtitle{font-size:12px;color:#94a3b8;margin-bottom:30px}
.nav a{display:block;padding:12px;margin:7px 0;border-radius:8px;text-decoration:none;color:#cbd5e1}
.nav a:hover,.nav a.active{background:#1e3a5f;color:#93c5fd}
main{flex:1;min-width:0;padding:30px}
h1{font-size:27px;margin:0 0 8px}
h2{font-size:18px;margin-top:0}
.muted{font-size:13px;color:#94a3b8}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin:24px 0}
.card,.panel{background:#111a2e;border:1px solid #26334b;border-radius:12px;padding:20px;margin-bottom:18px}
.label{font-size:13px;color:#94a3b8;margin-bottom:12px}
.value{font-size:26px;font-weight:bold;overflow-wrap:anywhere}
.blue{color:#60a5fa}.green{color:#6ee7b7}.orange{color:#fdba74}.purple{color:#c4b5fd}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.chart{width:100%;min-height:300px}
.grid-form{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.field{display:flex;flex-direction:column;gap:8px}
input,select{width:100%;padding:11px;border-radius:7px;border:1px solid #334155;background:#0b1120;color:#e5e7eb}
button,.btn{display:inline-block;padding:11px 16px;border:0;border-radius:7px;background:#2563eb;color:white;cursor:pointer;text-decoration:none;font-weight:bold}
button:hover,.btn:hover{background:#1d4ed8}
.result{padding:15px;border-radius:8px;margin-top:18px;background:#142b3c;border:1px solid #31516b;overflow-wrap:anywhere}
.error{background:#451a1a;border-color:#7f1d1d}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:11px;border-bottom:1px solid #26334b}
th{color:#93c5fd}
.table-wrap{overflow-x:auto}
pre{white-space:pre-wrap;overflow-wrap:anywhere}
.footer{color:#64748b;font-size:12px;margin-top:25px;line-height:1.7}
@media(max-width:1000px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.charts{grid-template-columns:1fr}}
@media(max-width:700px){.layout{display:block}.sidebar{width:100%}.nav{display:flex;flex-wrap:wrap;gap:4px}.nav a{font-size:12px;padding:9px;margin:0}main{padding:17px}.cards,.grid-form{grid-template-columns:1fr}h1{font-size:23px}}
</style>
"""

NAV = """
<aside class="sidebar">
    <div class="brand">LogisChain AI</div>
    <div class="subtitle">Trade Finance & Logistics Intelligence</div>
    <nav class="nav">
        <a href="/">Dashboard</a>
        <a href="/shipment">Shipment Risk</a>
        <a href="/demand">Demand Forecast</a>
        <a href="/trade-finance">Trade Finance</a>
        <a href="/analytics">Analytics & Reports</a>
    </nav>
</aside>
"""


def page(content, title="LogisChain AI", **context):
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>{{ title }}</title>
    """ + BASE_STYLE + """
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    </head>
    <body>
        <div class="layout">
    """ + NAV + """
            <main>
    """ + content + """
            </main>
        </div>
    </body>
    </html>
    """

    return render_template_string(template, title=title, **context)


@app.route("/")
def dashboard():
    forecast, date_col, value_col = get_forecast_data()

    dates = []
    values = []

    if date_col and value_col:
        chart_data = forecast[[date_col, value_col]].copy()
        chart_data[value_col] = pd.to_numeric(
            chart_data[value_col], errors="coerce"
        )
        chart_data = chart_data.dropna(subset=[value_col]).head(500)
        dates = chart_data[date_col].astype(str).tolist()
        values = [float(v) for v in chart_data[value_col].tolist()]

    shipment_status = "Loaded" if "shipment" in models else "Missing"
    demand_status = "Loaded" if "demand" in models else "Missing"
    trade_status = "Loaded" if "trade" in models else "Missing"

    content = """
    <h1>Supply Chain Intelligence</h1>
    <p class="muted">Monitor logistics, demand forecasts and trade finance risk.</p>

    <section class="cards">
        <div class="card"><div class="label">Logistics Records</div>
            <div class="value blue">{{ rows }}</div></div>
        <div class="card"><div class="label">Shipment Model</div>
            <div class="value green">{{ shipment }}</div></div>
        <div class="card"><div class="label">Demand Model</div>
            <div class="value orange">{{ demand }}</div></div>
        <div class="card"><div class="label">Trade Finance Model</div>
            <div class="value purple">{{ trade }}</div></div>
    </section>

    <section class="charts">
        <div class="panel">
            <h2>Model Availability</h2>
            <p class="muted">Status of trained prediction models</p>
            <div id="modelChart" class="chart"></div>
        </div>
        <div class="panel">
            <h2>Demand Forecast</h2>
            <p class="muted">Generated forecast records</p>
            <div id="demandChart" class="chart"></div>
        </div>
    </section>

    <section class="panel">
        <h2>Project Modules</h2>
        <p>Shipment Risk Prediction — estimates delivery risk from shipment features.</p>
        <p>Demand Forecasting — forecasts future sales using historical demand.</p>
        <p>Trade Finance Risk — demonstrates invoice and buyer risk classification.</p>
        <p style="margin-top:16px"><a class="btn" href="/analytics">View Analytics</a></p>
    </section>

    <p class="footer">LogisChain AI · Data Science Project<br>
    Predictions are estimates and must be validated before real-world use.</p>

    <script>
    const baseLayout = {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {color: "#cbd5e1"},
        margin: {l: 45, r: 15, t: 20, b: 45}
    };

    Plotly.newPlot("modelChart", [{
        type: "bar",
        x: ["Shipment", "Demand", "Trade Finance"],
        y: [{{ s_num }}, {{ d_num }}, {{ t_num }}],
        marker: {color: ["#60a5fa", "#6ee7b7", "#c4b5fd"]},
        text: [{{ shipment|tojson }}, {{ demand|tojson }}, {{ trade|tojson }}],
        textposition: "outside"
    }], {
        ...baseLayout,
        yaxis: {visible: false, range: [0, 1.3]}
    }, {responsive: true, displayModeBar: false});

    const dates = {{ dates|tojson }};
    const values = {{ values|tojson }};

    if (dates.length > 0 && values.length > 0) {
        Plotly.newPlot("demandChart", [{
            type: "scatter",
            mode: "lines+markers",
            x: dates,
            y: values,
            line: {color: "#60a5fa", width: 3},
            marker: {size: 7}
        }], {
            ...baseLayout,
            xaxis: {title: "Date"},
            yaxis: {title: "Forecast demand"}
        }, {responsive: true, displayModeBar: false});
    } else {
        document.getElementById("demandChart").innerHTML =
            "<p class='muted'>No forecast records or recognized forecast columns found.</p>";
    }
    </script>
    """

    return page(
        content,
        "Dashboard | LogisChain AI",
        rows=f"{get_logistics_rows():,}",
        shipment=shipment_status,
        demand=demand_status,
        trade=trade_status,
        s_num=1 if "shipment" in models else 0,
        d_num=1 if "demand" in models else 0,
        t_num=1 if "trade" in models else 0,
        dates=dates,
        values=values,
    )


@app.route("/api/health")
def api_health():
    forecast = read_csv_safe(FILES["forecast"])

    return jsonify({
        "status": "running",
        "logistics_rows": get_logistics_rows(),
        "shipment_model": "loaded" if "shipment" in models else "missing",
        "demand_model": "loaded" if "demand" in models else "missing",
        "trade_finance_model": "loaded" if "trade" in models else "missing",
        "forecast_rows": len(forecast),
    })


@app.route("/api/forecast-data")
def api_forecast_data():
    df, date_col, value_col = get_forecast_data()

    if df.empty or not date_col or not value_col:
        return jsonify({"dates": [], "values": []})

    valid = df[[date_col, value_col]].copy()
    valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
    valid = valid.dropna(subset=[value_col]).head(500)

    return jsonify({
        "dates": valid[date_col].astype(str).tolist(),
        "values": [float(v) for v in valid[value_col].tolist()],
    })


@app.route("/shipment", methods=["GET", "POST"])
def shipment():
    result, error = None, None

    if request.method == "POST":
        result, error = predict_shipment(request.form)

    content = """
    <h1>Shipment Risk Prediction</h1>
    <p class="muted">Enter shipment details to estimate delivery risk.</p>

    <div class="panel">
        <form method="post" class="grid-form">
            <div class="field"><label>Scheduled shipment days</label>
                <input type="number" min="0" name="Days for shipment (scheduled)" value="3" required></div>
            <div class="field"><label>Shipping mode</label>
                <select name="Shipping Mode">
                    <option>Standard Class</option><option>Second Class</option>
                    <option>First Class</option><option>Same Day</option>
                </select></div>
            <div class="field"><label>Customer segment</label>
                <select name="Customer Segment">
                    <option>Consumer</option><option>Corporate</option><option>Home Office</option>
                </select></div>
            <div class="field"><label>Market</label>
                <input name="Market" value="USCA" required></div>
            <div class="field"><label>Product price</label>
                <input type="number" min="0" step="0.01" name="Order Item Product Price" value="100" required></div>
            <div class="field"><label>Quantity</label>
                <input type="number" min="1" name="Order Item Quantity" value="1" required></div>
            <div class="field"><label>Discount</label>
                <input type="number" min="0" step="0.01" name="Order Item Discount" value="0" required></div>
            <div class="field"><label>Sales per customer</label>
                <input type="number" min="0" step="0.01" name="Sales per customer" value="100" required></div>
            <div class="field"><label>Order type</label>
                <select name="Type">
                    <option>DEBIT</option><option>TRANSFER</option><option>PAYMENT</option><option>CASH</option>
                </select></div>
            <div class="field"><label>Order region</label>
                <input name="Order Region" value="West of USA" required></div>
            <div><button type="submit">Predict Shipment Risk</button></div>
        </form>

        {% if error %}<div class="result error">{{ error }}</div>{% endif %}
        {% if result %}
        <div class="result">
            <h2>{{ result.label }}</h2>
            <p>Prediction class: {{ result.prediction }}</p>
            <p>Estimated model probability:
            {% if result.probability is not none %}
                {{ "%.1f"|format(result.probability * 100) }}%
            {% else %}Unavailable{% endif %}</p>
            <p class="muted">This is a model estimate, not a guaranteed delivery outcome.</p>
        </div>
        {% endif %}
    </div>
    """

    return page(
        content,
        "Shipment Risk | LogisChain AI",
        result=result,
        error=error,
    )


@app.route("/trade-finance", methods=["GET", "POST"])
def trade_finance():
    result, error = None, None

    if request.method == "POST":
        result, error = predict_trade_finance(request.form)

    content = """
    <h1>Trade Finance Risk</h1>
    <p class="muted">Estimate invoice and buyer risk using the trained model.</p>

    <div class="panel">
        <form method="post" class="grid-form">
            <div class="field"><label>Invoice amount</label>
                <input type="number" min="0" step="0.01" name="invoice_amount" value="50000" required></div>
            <div class="field"><label>Payment terms (days)</label>
                <input type="number" min="0" name="payment_terms_days" value="30" required></div>
            <div class="field"><label>Buyer credit score (0–900)</label>
                <input type="number" min="0" max="900" name="buyer_credit_score" value="700" required></div>
            <div class="field"><label>Shipment delay (days)</label>
                <input type="number" min="0" name="shipment_delay_days" value="0" required></div>
            <div class="field"><label>Country risk score (0–100)</label>
                <input type="number" min="0" max="100" name="country_risk_score" value="30" required></div>
            <div class="field"><label>Previous defaults</label>
                <input type="number" min="0" step="1" name="previous_defaults" value="0" required></div>
            <div><button type="submit">Analyze Finance Risk</button></div>
        </form>

        {% if error %}<div class="result error">{{ error }}</div>{% endif %}
        {% if result %}
        <div class="result">
            <h2>{{ result.label }}</h2>
            <p>Prediction class: {{ result.prediction }}</p>
            <p>Estimated model probability:
            {% if result.probability is not none %}
                {{ "%.1f"|format(result.probability * 100) }}%
            {% else %}Unavailable{% endif %}</p>
            <p class="muted">Prototype estimate only. Synthetic training data cannot validate real financial risk.</p>
        </div>
        {% endif %}
    </div>
    """

    return page(
        content,
        "Trade Finance | LogisChain AI",
        result=result,
        error=error,
    )


@app.route("/demand")
def demand():
    df, _, _ = get_forecast_data()

    if df.empty:
        table = "<p class='muted'>Forecast data is unavailable. Run src/demand_forecasting.py first.</p>"
    else:
        table = df.head(100).to_html(index=False, border=0)

    content = """
    <h1>Demand Forecast</h1>
    <p class="muted">Forecast output generated by your demand model.</p>
    <div class="panel">
        <p>Forecast records: <strong>{{ count }}</strong></p>
        <p class="muted">Showing up to 100 rows.</p>
        <div class="table-wrap">{{ table|safe }}</div>
        <p style="margin-top:18px">
            <a class="btn" href="/download/forecast">Download Forecast CSV</a>
        </p>
    </div>
    """

    return page(
        content,
        "Demand Forecast | LogisChain AI",
        count=len(df),
        table=table,
    )


@app.route("/analytics")
def analytics():
    evaluation = read_csv_safe(FILES["demand_evaluation"])
    trade_text = "Report not available."

    if FILES["trade_evaluation"].exists():
        try:
            trade_text = FILES["trade_evaluation"].read_text(encoding="utf-8")
        except OSError:
            trade_text = "Could not read trade finance evaluation report."

    if evaluation.empty:
        evaluation_html = "<p class='muted'>Demand evaluation report is not available.</p>"
    else:
        evaluation_html = evaluation.to_html(index=False, border=0)

    content = """
    <h1>Analytics & Reports</h1>
    <p class="muted">Review available model evaluation outputs.</p>

    <div class="panel">
        <h2>Demand Forecast Evaluation</h2>
        <div class="table-wrap">{{ evaluation_html|safe }}</div>
        <p style="margin-top:18px">
            <a class="btn" href="/download/evaluation">Download Evaluation CSV</a>
        </p>
    </div>

    <div class="panel">
        <h2>Trade Finance Evaluation</h2>
        <pre>{{ trade_text }}</pre>
    </div>

    <div class="panel">
        <h2>Model Status</h2>
        <p>Shipment: {{ shipment }}</p>
        <p>Demand: {{ demand }}</p>
        <p>Trade Finance: {{ trade }}</p>
        <p style="margin-top:15px"><a class="btn" href="/api/health">Open Health API</a></p>
    </div>
    """

    return page(
        content,
        "Analytics | LogisChain AI",
        evaluation_html=evaluation_html,
        trade_text=trade_text,
        shipment="Loaded" if "shipment" in models else "Missing",
        demand="Loaded" if "demand" in models else "Missing",
        trade="Loaded" if "trade" in models else "Missing",
    )


@app.route("/download/<report_name>")
def download_report(report_name):
    allowed = {
        "forecast": FILES["forecast"],
        "evaluation": FILES["demand_evaluation"],
    }

    path = allowed.get(report_name)

    if path is None:
        return jsonify({"error": "Unknown report"}), 404

    if not path.exists():
        return jsonify({"error": "Report file not found"}), 404

    try:
        content = path.read_bytes()
    except OSError:
        return jsonify({"error": "Could not read report file"}), 500

    return Response(
        content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{path.name}"'
        },
    )


@app.route("/api/shipment-predict", methods=["POST"])
def api_shipment_predict():
    data = request.get_json(silent=True) or {}
    result, error = predict_shipment(data)

    if error:
        return jsonify({"error": error}), 400

    return jsonify(result)


@app.route("/api/trade-finance-predict", methods=["POST"])
def api_trade_finance_predict():
    data = request.get_json(silent=True) or {}
    result, error = predict_trade_finance(data)

    if error:
        return jsonify({"error": error}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)

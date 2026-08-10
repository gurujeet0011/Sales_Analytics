"""
Flask app for the Sales Data Visualisation + ML Prediction project.

Routes:
  GET  /                 dashboard (charts built from the cleaned dataset)
  GET  /predict           order-level Sales/Profit prediction form
  POST /predict           runs the ML models and shows the prediction
  GET  /forecast           monthly sales forecast page (chart + table)
  GET  /api/predict        JSON API: predict Sales & Profit for given order features
  GET  /api/forecast       JSON API: monthly sales forecast
  GET  /api/dashboard-data JSON API: dashboard aggregates
  GET  /health              simple health check for deployment platforms
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "model", "artifacts")

app = Flask(__name__)

# ---------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------
sales_model = joblib.load(os.path.join(ARTIFACT_DIR, "sales_model.pkl"))
profit_model = joblib.load(os.path.join(ARTIFACT_DIR, "profit_model.pkl"))
forecast_bundle = joblib.load(os.path.join(ARTIFACT_DIR, "forecast_model.pkl"))

with open(os.path.join(ARTIFACT_DIR, "form_options.json")) as f:
    FORM_OPTIONS = json.load(f)

with open(os.path.join(ARTIFACT_DIR, "dashboard_data.json")) as f:
    DASHBOARD_DATA = json.load(f)

with open(os.path.join(ARTIFACT_DIR, "monthly_history.json")) as f:
    MONTHLY_HISTORY = json.load(f)

with open(os.path.join(ARTIFACT_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

FEATURE_ORDER = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode", "Quantity", "Discount"]


# ---------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------
def predict_sales_profit(category, sub_category, region, segment, ship_mode, quantity, discount):
    row = pd.DataFrame([{
        "Category": category,
        "Sub-Category": sub_category,
        "Region": region,
        "Segment": segment,
        "Ship Mode": ship_mode,
        "Quantity": quantity,
        "Discount": discount,
    }])[FEATURE_ORDER]

    sales_log_pred = sales_model.predict(row)[0]
    sales_pred = float(np.expm1(sales_log_pred))

    profit_slog_pred = profit_model.predict(row)[0]
    profit_pred = float(np.sign(profit_slog_pred) * np.expm1(np.abs(profit_slog_pred)))

    margin_pct = (profit_pred / sales_pred * 100) if sales_pred else 0.0

    return {
        "predicted_sales": round(sales_pred, 2),
        "predicted_profit": round(profit_pred, 2),
        "predicted_margin_pct": round(margin_pct, 1),
    }


def forecast_future_months(n_months=6):
    model = forecast_bundle["model"]
    month_dummy_cols = forecast_bundle["month_dummy_cols"]
    last_t = forecast_bundle["last_t"]
    last_date = pd.Timestamp(forecast_bundle["last_date"])
    resid_std = forecast_bundle["resid_std"]

    future_rows = []
    for i in range(1, n_months + 1):
        t = last_t + i
        date = (last_date + pd.DateOffset(months=i))
        month_num = date.month
        row = {"t": t}
        for col in month_dummy_cols:
            row[col] = 1 if col == f"m_{month_num}" else 0
        future_rows.append((date, row))

    X_future = pd.DataFrame([r for _, r in future_rows])[["t"] + month_dummy_cols]
    preds = model.predict(X_future)

    results = []
    for (date, _), pred in zip(future_rows, preds):
        results.append({
            "date": date.strftime("%Y-%m"),
            "predicted_sales": round(float(pred), 2),
            "lower_ci": round(float(pred - 1.96 * resid_std), 2),
            "upper_ci": round(float(pred + 1.96 * resid_std), 2),
        })
    return results


# ---------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html", data=DASHBOARD_DATA)


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    result = None
    form_values = {
        "category": FORM_OPTIONS["categories"][0],
        "sub_category": FORM_OPTIONS["sub_categories_by_category"][FORM_OPTIONS["categories"][0]][0],
        "region": FORM_OPTIONS["regions"][0],
        "segment": FORM_OPTIONS["segments"][0],
        "ship_mode": FORM_OPTIONS["ship_modes"][0],
        "quantity": 3,
        "discount": 0.2,
    }

    if request.method == "POST":
        form_values["category"] = request.form.get("category")
        form_values["sub_category"] = request.form.get("sub_category")
        form_values["region"] = request.form.get("region")
        form_values["segment"] = request.form.get("segment")
        form_values["ship_mode"] = request.form.get("ship_mode")
        form_values["quantity"] = int(request.form.get("quantity", 1))
        form_values["discount"] = float(request.form.get("discount", 0))

        result = predict_sales_profit(
            form_values["category"], form_values["sub_category"], form_values["region"],
            form_values["segment"], form_values["ship_mode"], form_values["quantity"], form_values["discount"],
        )

    return render_template(
        "predict.html",
        options=FORM_OPTIONS,
        form_values=form_values,
        result=result,
        metrics=METRICS["order_models"],
    )


@app.route("/forecast")
def forecast_page():
    n_months = int(request.args.get("months", 6))
    n_months = max(1, min(n_months, 12))
    future = forecast_future_months(n_months)
    return render_template(
        "forecast.html",
        history=MONTHLY_HISTORY,
        future=future,
        n_months=n_months,
        metrics=METRICS["forecast_model"],
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------
# JSON API routes
# ---------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    required = ["category", "sub_category", "region", "segment", "ship_mode", "quantity", "discount"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        result = predict_sales_profit(
            payload["category"], payload["sub_category"], payload["region"],
            payload["segment"], payload["ship_mode"], int(payload["quantity"]), float(payload["discount"]),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


@app.route("/api/forecast")
def api_forecast():
    n_months = int(request.args.get("months", 6))
    n_months = max(1, min(n_months, 12))
    return jsonify({"history": MONTHLY_HISTORY, "forecast": forecast_future_months(n_months)})


@app.route("/api/dashboard-data")
def api_dashboard_data():
    return jsonify(DASHBOARD_DATA)


@app.route("/api/form-options")
def api_form_options():
    return jsonify(FORM_OPTIONS)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

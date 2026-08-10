"""
train_model.py
---------------
Trains the ML models used by the Flask app:

1. Sales predictor   - RandomForestRegressor, predicts order Sales value
2. Profit predictor  - RandomForestRegressor, predicts order Profit value
3. Monthly forecast  - Linear regression on time index + seasonal month dummies,
                        used to forecast total sales for future months

All artifacts are saved into model/artifacts/ as .pkl files so the Flask app
can load them at startup without retraining.

Run with:  python model/train_model.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_data.xlsx")
ARTIFACT_DIR = os.path.join(BASE_DIR, "model", "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

FEATURES = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode", "Quantity", "Discount"]
CAT_FEATURES = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"]
NUM_FEATURES = ["Quantity", "Discount"]


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_order_level_models(df: pd.DataFrame) -> dict:
    """Predict Sales using a log1p target transform (Sales is heavily right-skewed;
    log-space training materially improves fit). Profit can be negative, so it is
    trained on a signed-log transform instead: sign(x) * log1p(|x|)."""
    metrics = {}
    X = df[FEATURES]

    # ---- Sales model (log1p transform) ----
    y_sales = np.log1p(df["Sales"])
    X_train, X_test, y_train, y_test = train_test_split(X, y_sales, test_size=0.2, random_state=42)
    sales_pipe = build_pipeline(RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1))
    sales_pipe.fit(X_train, y_train)
    pred_log = sales_pipe.predict(X_test)
    pred_real = np.expm1(pred_log)
    true_real = np.expm1(y_test)
    metrics["Sales"] = {
        "mae": round(float(mean_absolute_error(true_real, pred_real)), 2),
        "r2": round(float(r2_score(true_real, pred_real)), 4),
        "r2_log_space": round(float(r2_score(y_test, pred_log)), 4),
        "n_train": len(X_train), "n_test": len(X_test), "transform": "log1p",
    }
    joblib.dump(sales_pipe, os.path.join(ARTIFACT_DIR, "sales_model.pkl"))
    print(f"[Sales] MAE={metrics['Sales']['mae']:.2f}  R2={metrics['Sales']['r2']:.4f}  (log-space R2={metrics['Sales']['r2_log_space']:.4f}) -> saved sales_model.pkl")

    # ---- Profit model (signed-log transform, since profit can be negative) ----
    y_profit_raw = df["Profit"]
    y_profit = np.sign(y_profit_raw) * np.log1p(np.abs(y_profit_raw))
    X_train, X_test, y_train, y_test = train_test_split(X, y_profit, test_size=0.2, random_state=42)
    profit_pipe = build_pipeline(RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1))
    profit_pipe.fit(X_train, y_train)
    pred_slog = profit_pipe.predict(X_test)
    pred_real = np.sign(pred_slog) * (np.expm1(np.abs(pred_slog)))
    true_real = np.sign(y_test) * (np.expm1(np.abs(y_test)))
    metrics["Profit"] = {
        "mae": round(float(mean_absolute_error(true_real, pred_real)), 2),
        "r2": round(float(r2_score(true_real, pred_real)), 4),
        "r2_transform_space": round(float(r2_score(y_test, pred_slog)), 4),
        "n_train": len(X_train), "n_test": len(X_test), "transform": "signed_log1p",
    }
    joblib.dump(profit_pipe, os.path.join(ARTIFACT_DIR, "profit_model.pkl"))
    print(f"[Profit] MAE={metrics['Profit']['mae']:.2f}  R2={metrics['Profit']['r2']:.4f}  (transform-space R2={metrics['Profit']['r2_transform_space']:.4f}) -> saved profit_model.pkl")

    return metrics


def train_monthly_forecast_model(df: pd.DataFrame) -> dict:
    """Aggregate to monthly total sales, fit a trend + month-seasonality linear model,
    forecast the next 6 months beyond the dataset's last month."""
    d = df.copy()
    d["Order Date"] = pd.to_datetime(d["Order Date"])
    monthly = d.groupby(d["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly["Order Date"] = monthly["Order Date"].dt.to_timestamp()
    monthly = monthly.sort_values("Order Date").reset_index(drop=True)
    monthly["t"] = np.arange(len(monthly))
    monthly["month_num"] = monthly["Order Date"].dt.month

    # one-hot month seasonality (drop first to avoid collinearity with intercept)
    month_dummies = pd.get_dummies(monthly["month_num"], prefix="m", drop_first=True)
    X = pd.concat([monthly[["t"]], month_dummies], axis=1)
    y = monthly["Sales"]

    model = LinearRegression().fit(X, y)
    preds_hist = model.predict(X)
    r2 = float(r2_score(y, preds_hist))
    resid_std = float(np.std(y - preds_hist))

    joblib.dump(
        {"model": model, "month_dummy_cols": list(month_dummies.columns), "last_t": int(monthly["t"].iloc[-1]),
         "last_date": monthly["Order Date"].iloc[-1].strftime("%Y-%m-%d"), "resid_std": resid_std, "r2": r2},
        os.path.join(ARTIFACT_DIR, "forecast_model.pkl"),
    )

    history = [{"date": row["Order Date"].strftime("%Y-%m"), "sales": round(float(row["Sales"]), 2)}
               for _, row in monthly.iterrows()]
    with open(os.path.join(ARTIFACT_DIR, "monthly_history.json"), "w") as f:
        json.dump(history, f)

    print(f"[Forecast] R2={r2:.4f}  months_trained={len(monthly)}  -> saved forecast_model.pkl")
    return {"r2": round(r2, 4), "resid_std": round(resid_std, 2), "months_trained": len(monthly)}


def save_reference_data(df: pd.DataFrame):
    """Save dropdown option lists + dashboard aggregates the Flask app needs at runtime."""
    options = {
        "categories": sorted(df["Category"].dropna().unique().tolist()),
        "sub_categories_by_category": {
            cat: sorted(df[df["Category"] == cat]["Sub-Category"].dropna().unique().tolist())
            for cat in df["Category"].dropna().unique()
        },
        "regions": sorted(df["Region"].dropna().unique().tolist()),
        "segments": sorted(df["Segment"].dropna().unique().tolist()),
        "ship_modes": sorted(df["Ship Mode"].dropna().unique().tolist()),
    }
    with open(os.path.join(ARTIFACT_DIR, "form_options.json"), "w") as f:
        json.dump(options, f)

    dash = {
        "kpis": {
            "total_sales": round(float(df["Sales"].sum()), 2),
            "total_profit": round(float(df["Profit"].sum()), 2),
            "total_orders": int(df["Order ID"].nunique()),
            "avg_order_value": round(float(df["Sales"].mean()), 2),
            "total_customers": int(df["Customer ID"].nunique()),
        },
        "sales_by_category": df.groupby("Category")["Sales"].sum().round(2).to_dict(),
        "profit_by_category": df.groupby("Category")["Profit"].sum().round(2).to_dict(),
        "sales_by_region": df.groupby("Region")["Sales"].sum().round(2).to_dict(),
        "top_customers": df.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(10).round(2).to_dict(),
        "top_subcategories": df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10).round(2).to_dict(),
        "sales_by_segment": df.groupby("Segment")["Sales"].sum().round(2).to_dict(),
    }
    with open(os.path.join(ARTIFACT_DIR, "dashboard_data.json"), "w") as f:
        json.dump(dash, f)

    print("[Reference] saved form_options.json and dashboard_data.json")


def main():
    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=["Sales", "Profit", "Category", "Sub-Category", "Region", "Segment", "Ship Mode", "Quantity", "Discount"])

    metrics = {}
    metrics["order_models"] = train_order_level_models(df)
    metrics["forecast_model"] = train_monthly_forecast_model(df)
    save_reference_data(df)

    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nAll artifacts saved to:", ARTIFACT_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

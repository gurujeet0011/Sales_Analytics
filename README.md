# Superstore Sales Analytics — Flask App

A Flask web app that turns the original notebook (`Python project.ipynb`) into a proper
deployable application: an interactive dashboard, an ML-powered order prediction tool,
and a monthly sales forecast — all served from real trained models, not static images.

## What's inside

```
sales_app/
├── app.py                  # Flask routes (pages + JSON API)
├── wsgi.py                 # production entry point
├── Procfile                 # for Heroku/Render-style platforms
├── requirements.txt
├── data/
│   └── clean_data.xlsx       # the cleaned dataset from your notebook
├── model/
│   ├── train_model.py         # trains & saves all ML models
│   └── artifacts/               # generated: .pkl models + .json reference data
├── templates/
│   ├── base.html, dashboard.html, predict.html, forecast.html
└── static/css/style.css
```

## ML models included

1. **Sales predictor** — RandomForestRegressor, log1p-transformed target, features:
   Category, Sub-Category, Region, Segment, Ship Mode, Quantity, Discount.
2. **Profit predictor** — same features, signed-log1p transform (profit can be negative).
3. **Monthly sales forecaster** — linear trend + month-seasonality model on 48 months
   of aggregated history, forecasts 1–12 months ahead with a 95% confidence band.

Model performance is shown directly on the `/predict` page for transparency. Order-level
R² is modest (~0.18–0.14) because the dataset has no per-product price/SKU — that's the
single biggest real driver of an individual order's value and it isn't in this dataset.
The monthly forecast model is much stronger (R² ≈ 0.89) since it predicts aggregate trend,
not single-order noise.

## Run locally

```bash
cd sales_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the models (only needed once, or whenever data/clean_data.xlsx changes)
python model/train_model.py

# Start the app
python app.py
```

Then open **http://localhost:5000**.

## Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Dashboard with KPI cards and charts |
| `/predict` | GET/POST | Order prediction form + result |
| `/forecast` | GET | Monthly sales forecast (`?months=3/6/12`) |
| `/api/predict` | POST | JSON: `{category, sub_category, region, segment, ship_mode, quantity, discount}` → `{predicted_sales, predicted_profit, predicted_margin_pct}` |
| `/api/forecast` | GET | JSON monthly forecast |
| `/api/dashboard-data` | GET | JSON dashboard aggregates |
| `/api/form-options` | GET | JSON dropdown options |
| `/health` | GET | Health check |

Example API call:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"category":"Technology","sub_category":"Phones","region":"West","segment":"Consumer","ship_mode":"Second Class","quantity":3,"discount":0.1}'
```

## Deploying it live

This app is a standard Flask + gunicorn app, so it deploys to any Python host. A few
straightforward options:

**Render.com (free tier available)**
1. Push this folder to a GitHub repo.
2. New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt && python model/train_model.py`
4. Start command: `gunicorn wsgi:app`

**Railway.app**
1. Push to GitHub, "Deploy from repo".
2. Railway auto-detects the `Procfile`. Add a pre-deploy/build step to run
   `python model/train_model.py` (or commit the `model/artifacts/` folder directly so
   no training step is needed at deploy time).

**PythonAnywhere**
1. Upload the folder, create a virtualenv, `pip install -r requirements.txt`.
2. Run `python model/train_model.py` once via a Bash console.
3. Point the WSGI config file at `wsgi.app`.

**Docker (any host)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt && python model/train_model.py
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "wsgi:app"]
```

> Tip: commit `model/artifacts/*.pkl` and `*.json` to your repo so the deploy step
> doesn't need to retrain on every build — just re-run `train_model.py` locally
> whenever the dataset changes and commit the refreshed artifacts.

## Retraining on new data

Replace `data/clean_data.xlsx` with an updated export (same column names), then:
```bash
python model/train_model.py
```
This regenerates every `.pkl` and `.json` file the app depends on. No code changes needed.

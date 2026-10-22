# FloraSense C4 Forecasting API

FastAPI service for estimating dispatched supply after an export order is placed. It loads the trained XGBoost checkpoint and the preprocessing artifacts produced by the C4 scripts. It does not train a model when the API starts.

## Prepare the model artifacts

If the model or preprocessing artifacts are missing or the C4 source data has changed, install the training dependencies and run these from the repository root:

```powershell
python -m pip install -r requirements/base.txt -r requirements/c4.txt
python scripts/c4_time_series/c4_clean_sales.py
python scripts/c4_time_series/c4_clean_weather.py
python scripts/c4_time_series/c4_merge_master.py
python scripts/c4_time_series/c4_04_ml_preprocessing.py
python scripts/c4_time_series/c4_05_baseline_models.py
```

The API requires these generated files:

- `data/c4_time_series/processed/c4_master_training_data.csv`
- `data/c4_time_series/interim/cleaned_weather_weekly.csv`
- `data/c4_time_series/ml_ready/feature_meta.json`
- `data/c4_time_series/ml_ready/scaler.pkl`
- `models/c4_time_series/checkpoints/xgboost_model.pkl`

The model checkpoint and source data are local artifacts and are not committed to Git. Generate or restore them on the machine running the service.

## Install and run

From the repository root, install the shared API and lightweight C4 inference dependencies, then start Uvicorn:

```powershell
python -m pip install -r requirements/base.txt -r requirements/c4-api.txt
python -m uvicorn services.c4_forecasting.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. In a second terminal, start the frontend:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173/c4/forecast`. CORS allows localhost and 127.0.0.1 on port 5173 by default. Set `FLORASENSE_CORS_ORIGINS` to a comma-separated list to add other frontend origins.

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/api/v1/forecast/metadata` | Distinct customer, variety, size, and mode options from the training data |
| POST | `/api/v1/forecast/predict` | Predict dispatched quantity and calculate the supply-demand gap |

Example request:

```json
{
  "customer": "CUSTOMER-C",
  "plant_variety": "Codiaeum spp.",
  "size": "15/17cm",
  "mode": "Un-rooted",
  "ordered_quantity": 10000,
  "year": 2026,
  "week": 40,
  "avg_max_temp_c": 33.5,
  "avg_min_temp_c": 24.0,
  "weekly_rainfall_mm": 15.0
}
```

Successful responses include the original order context, non-negative integer dispatch estimate, exact `predicted_dispatched_quantity - ordered_quantity` gap, fulfillment rate, operational status, model label, and measured model inference time. A zero order has a null fulfillment rate.

The API uses the submitted target-week weather for same-week features and reads historical weekly weather for the 2-week and 4-week lag features. If those historical lag values are not present, it uses the training-only median and marks the corresponding missingness features, matching the preprocessing pipeline. The expected weather values should be available at the time the manager requests the forecast.

Status is derived from the gap: negative is `Deficit / Risk`, zero is `Balanced`, and positive is `Surplus`. Unknown categories or invalid request values return HTTP 422. Missing model/data artifacts return HTTP 503 with setup guidance.

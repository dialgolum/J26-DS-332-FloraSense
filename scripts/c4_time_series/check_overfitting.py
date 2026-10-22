# scripts/c4_time_series/check_all_models_overfitting.py
import os, json, joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("=" * 78)
print("  COMPONENT 4: COMPREHENSIVE OVERFITTING & GENERALIZATION AUDIT")
print("=" * 78)

# 1. Load Data Splits
ml_dir = os.path.join('data', 'c4_time_series', 'ml_ready')
results_dir = os.path.join('data', 'c4_time_series', 'results')

train_df = pd.read_csv(os.path.join(ml_dir, 'final_train_data.csv'))
test_df  = pd.read_csv(os.path.join(ml_dir, 'test_data.csv'))

with open(os.path.join(ml_dir, 'feature_meta.json')) as f:
    meta = json.load(f)

TARGET = meta['target_col']
FEATURE_COLS = [c for c in meta['feature_cols'] if c in train_df.columns and c in test_df.columns]

X_train, y_train = train_df[FEATURE_COLS].values, train_df[TARGET].values
X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df[TARGET].values

# 2. Evaluate Models
summary_rows = []

# A. XGBoost & Random Forest
tree_models = {
    'XGBoost': os.path.join('models', 'c4_time_series', 'checkpoints', 'xgboost_model.pkl'),
    'Random Forest': os.path.join('models', 'c4_time_series', 'checkpoints', 'rf_model.pkl')
}

for name, path in tree_models.items():
    if os.path.exists(path):
        model = joblib.load(path)
        tr_pred = model.predict(X_train)
        te_pred = model.predict(X_test)
        
        tr_r2, te_r2 = r2_score(y_train, tr_pred), r2_score(y_test, te_pred)
        tr_rmse, te_rmse = np.sqrt(mean_squared_error(y_train, tr_pred)), np.sqrt(mean_squared_error(y_test, te_pred))
        nonzero = y_test != 0
        te_mape = np.mean(np.abs((y_test[nonzero] - te_pred[nonzero]) / y_test[nonzero])) * 100
        
        ratio = te_rmse / tr_rmse if tr_rmse > 0 else np.inf
        status = "Overfit warning (large train/test gap)" if ratio >= 1.5 else "No large train/test gap"
        summary_rows.append({'Model': name, 'Train R²': tr_r2, 'Test R²': te_r2, 'Train RMSE': tr_rmse, 'Test RMSE': te_rmse, 'Test MAPE (%)': te_mape, 'Status': status})

# B. Deep Learning Models (from generated prediction outputs)
dl_files = {
    'Stacked Bi-LSTM': os.path.join(results_dir, 'predictions_stacked_lstm.csv'),
    'TFT': os.path.join(results_dir, 'predictions_tft.csv')
}

for name, fpath in dl_files.items():
    if os.path.exists(fpath):
        pdf = pd.read_csv(fpath)
        y_t, y_p = pdf['Actual'].values, pdf['Predicted'].values
        te_r2 = r2_score(y_t, y_p)
        te_rmse = np.sqrt(mean_squared_error(y_t, y_p))
        nonzero = y_t != 0
        te_mape = np.mean(np.abs((y_t[nonzero] - y_p[nonzero]) / y_t[nonzero])) * 100
        summary_rows.append({
            'Model': name, 'Train R²': np.nan, 'Test R²': te_r2,
            'Train RMSE': np.nan, 'Test RMSE': te_rmse,
            'Test MAPE (%)': te_mape,
            'Status': 'Test score only; use c4_08_overfit_check.py for fit diagnostics'
        })

# C. SARIMAX
sarimax_file = os.path.join(results_dir, 'predictions_sarimax.csv')
if os.path.exists(sarimax_file):
    sdf = pd.read_csv(sarimax_file)
    y_t, y_p = sdf['Actual_Weekly_Total'].values, sdf['Predicted_Weekly_Total'].values
    te_r2 = r2_score(y_t, y_p)
    te_rmse = np.sqrt(mean_squared_error(y_t, y_p))
    nonzero = y_t != 0
    te_mape = np.mean(np.abs((y_t[nonzero] - y_p[nonzero]) / y_t[nonzero])) * 100
    summary_rows.append({
        'Model': 'SARIMAX (Weekly Agg.)', 'Train R²': np.nan, 'Test R²': te_r2,
        'Train RMSE': np.nan, 'Test RMSE': te_rmse,
        'Test MAPE (%)': te_mape,
        'Status': 'Weekly test score only; different aggregation from row-level models'
    })

audit_df = pd.DataFrame(summary_rows)
print(audit_df.to_string(index=False))
print("\nFit diagnostics and 2025 validation checks: scripts/c4_time_series/c4_08_overfit_check.py")
print("=" * 78)

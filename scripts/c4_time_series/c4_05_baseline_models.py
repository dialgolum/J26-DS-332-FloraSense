import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

print("=" * 65)
print("  Script 5: Baseline Models -- XGBoost, Random Forest, SARIMAX")
print("=" * 65)

ml_dir      = os.path.join('data', 'c4_time_series', 'ml_ready')
results_dir = os.path.join('data', 'c4_time_series', 'results')
plots_dir   = os.path.join('data', 'c4_time_series', 'results', 'plots')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

train_df = pd.read_csv(os.path.join(ml_dir, 'final_train_data.csv'))
test_df  = pd.read_csv(os.path.join(ml_dir, 'test_data.csv'))

with open(os.path.join(ml_dir, 'feature_meta.json')) as f:
    meta = json.load(f)

TARGET       = meta['target_col']
FEATURE_COLS = [c for c in meta['feature_cols'] if c in train_df.columns and c in test_df.columns]

X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df[TARGET].values.astype(np.float32)
X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_test  = test_df[TARGET].values.astype(np.float32)

print(f"\nFeatures : {len(FEATURE_COLS)}")
print(f"Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

def evaluate(y_true, y_pred, model_name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{model_name}]")
    print(f"    MAE  : {mae:>10,.2f}")
    print(f"    RMSE : {rmse:>10,.2f}")
    print(f"    MAPE : {mape:>9.2f}%")
    print(f"    R2   : {r2:>10.4f}")
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE_%': mape, 'R2': r2}

def save_predictions(y_true, y_pred, model_name: str, test_df_ref: pd.DataFrame):
    id_cols = ['Dispatch_Date', 'Customer', 'Plant_Variety', 'Size', 'Mode', 'YEAR', 'WEEK']
    id_cols = [c for c in id_cols if c in test_df_ref.columns]
    pred_df = test_df_ref[id_cols].copy()
    pred_df['Actual']    = y_true
    pred_df['Predicted'] = np.round(y_pred).astype(int)
    pred_df['Residual']  = pred_df['Actual'] - pred_df['Predicted']
    fname = os.path.join(results_dir, f'predictions_{model_name.lower().replace(" ", "_")}.csv')
    pred_df.to_csv(fname, index=False)
    print(f"    Predictions saved -> {fname}")
    return pred_df

all_metrics = []

# 1. XGBoost Regressor
print("\n" + "-" * 50)
print("Training XGBoost Regressor...")
xgb_model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
all_metrics.append(evaluate(y_test, y_pred_xgb, "XGBoost"))
pred_df_xgb = save_predictions(y_test, y_pred_xgb, "XGBoost", test_df)

os.makedirs(os.path.join('models', 'c4_time_series', 'checkpoints'), exist_ok=True)
joblib.dump(xgb_model, os.path.join('models', 'c4_time_series', 'checkpoints', 'xgboost_model.pkl'))

importances = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS)
top20 = importances.nlargest(20).sort_values()
plt.figure(figsize=(10, 7))
colors = plt.cm.RdYlGn([i / len(top20) for i in range(len(top20))])
top20.plot(kind='barh', color=colors)
plt.title('XGBoost -- Top 20 Feature Importances', fontsize=13, fontweight='bold')
plt.xlabel('Gain Importance', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'xgboost_feature_importance.png'), dpi=300)
plt.close()

# 2. Random Forest Regressor
print("\n" + "-" * 50)
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
all_metrics.append(evaluate(y_test, y_pred_rf, "Random Forest"))
pred_df_rf = save_predictions(y_test, y_pred_rf, "Random Forest", test_df)
joblib.dump(rf_model, os.path.join('models', 'c4_time_series', 'checkpoints', 'rf_model.pkl'))

# 3. Simple naive baselines. These provide a useful skill yardstick for the
# learned models and use only information available before the 2026 test period.
print("\n" + "-" * 50)
print("Scoring naive baselines...")
group_cols = meta['cat_cols']
raw_master = pd.read_csv(os.path.join('data', 'c4_time_series', 'processed', 'c4_master_training_data.csv'))
raw_train = raw_master[raw_master['YEAR'] < 2026].copy()
raw_test = raw_master[raw_master['YEAR'] == 2026].copy().reset_index(drop=True)
if len(raw_test) != len(test_df):
    raise ValueError('Raw and processed test rows are not aligned; regenerate preprocessing outputs.')
naive_order_pred = raw_test['Ordered_Quantity'].to_numpy(dtype=float)
all_metrics.append(evaluate(y_test, naive_order_pred, 'Naive: Ordered Quantity'))
save_predictions(y_test, naive_order_pred, 'Naive Order', test_df)

train_group_totals = raw_train.groupby(group_cols, dropna=False).agg(
    dispatched=(TARGET, 'sum'), ordered=('Ordered_Quantity', 'sum')
)
global_ratio = raw_train[TARGET].sum() / max(raw_train['Ordered_Quantity'].sum(), 1e-8)
group_ratio = (train_group_totals['dispatched'] /
               train_group_totals['ordered'].replace(0, np.nan)).fillna(global_ratio)
test_keys = pd.MultiIndex.from_frame(raw_test[group_cols])
ratios = pd.Series(test_keys.map(group_ratio), index=raw_test.index).fillna(global_ratio).to_numpy()
naive_ratio_pred = raw_test['Ordered_Quantity'].to_numpy(dtype=float) * ratios
all_metrics.append(evaluate(y_test, naive_ratio_pred, 'Naive: Historical Dispatch/Order Ratio'))
save_predictions(y_test, naive_ratio_pred, 'Naive Ratio', test_df)

last_observed = (raw_train.sort_values('Dispatch_Date')
                 .groupby(group_cols, dropna=False).tail(1)
                 .set_index(group_cols)[TARGET])
last_keys = pd.MultiIndex.from_frame(raw_test[group_cols])
global_last = raw_train.sort_values('Dispatch_Date')[TARGET].iloc[-1]
naive_last_pred = pd.Series(last_keys.map(last_observed), index=raw_test.index).fillna(global_last).to_numpy(dtype=float)
all_metrics.append(evaluate(y_test, naive_last_pred, 'Naive: Last Training Dispatch'))
save_predictions(y_test, naive_last_pred, 'Naive Last', test_df)

# 4. SARIMAX (Weekly Aggregate Baseline)
print("\n" + "-" * 50)
print("Training SARIMAX (weekly aggregate baseline)...")
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    master_df = pd.read_csv(os.path.join('data', 'c4_time_series', 'processed', 'c4_master_training_data.csv'))
    weekly_agg = master_df.groupby(['YEAR', 'WEEK'], as_index=False).agg(
        Dispatched_Quantity=('Dispatched_Quantity', 'sum'),
        Ordered_Quantity=('Ordered_Quantity', 'sum'),
        Avg_Max_Temp_C=('Avg_Max_Temp_C', 'first'),
        Weekly_Rainfall_mm=('Weekly_Rainfall_mm', 'first'),
        Growing_Degree_Days_Weekly=('Growing_Degree_Days_Weekly', 'first')
    ).sort_values(['YEAR', 'WEEK']).reset_index(drop=True)

    train_w = weekly_agg[weekly_agg['YEAR'] < 2026].reset_index(drop=True)
    test_w  = weekly_agg[weekly_agg['YEAR'] == 2026].reset_index(drop=True)
    # Weather beyond week 39 of 2026 is unavailable; use only order totals,
    # which are available at the intended post-order prediction time.
    exog_cols = ['Ordered_Quantity']

    sarimax_model = SARIMAX(
        train_w['Dispatched_Quantity'],
        exog=train_w[exog_cols],
        order=(1, 0, 1),
        seasonal_order=(1, 0, 0, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarimax_result = sarimax_model.fit(disp=False, maxiter=200)
    sarimax_pred = sarimax_result.forecast(steps=len(test_w), exog=test_w[exog_cols])

    y_true_sarima = test_w['Dispatched_Quantity'].values
    y_pred_sarima = np.maximum(0, sarimax_pred.values)
    all_metrics.append(evaluate(y_true_sarima, y_pred_sarima, "SARIMAX (weekly agg.)"))

    sarima_pred_df = pd.DataFrame({
        'YEAR': test_w['YEAR'],
        'WEEK': test_w['WEEK'],
        'Actual_Weekly_Total': y_true_sarima,
        'Predicted_Weekly_Total': np.round(y_pred_sarima).astype(int),
        'Residual': y_true_sarima - np.round(y_pred_sarima)
    })
    sarima_pred_df.to_csv(os.path.join(results_dir, 'predictions_sarimax.csv'), index=False)
    print(f"    Predictions saved -> {os.path.join(results_dir, 'predictions_sarimax.csv')}")

    fig = sarimax_result.plot_diagnostics(figsize=(14, 8), lags=20)
    fig.suptitle('SARIMAX Model Diagnostics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sarimax_diagnostics.png'), dpi=200)
    plt.close()
except Exception as e:
    print(f"  [WARN]  SARIMAX failed: {e}. Skipping.")

# 4. Actual vs Predicted Plot (XGBoost)
print("\nGenerating Actual vs. Predicted plot (XGBoost)...")
if 'Plant_Variety' in pred_df_xgb.columns:
    top_varieties = pred_df_xgb['Plant_Variety'].value_counts().head(5).index.tolist()
    fig, axes = plt.subplots(len(top_varieties), 1, figsize=(14, 3.5 * len(top_varieties)))
    if len(top_varieties) == 1:
        axes = [axes]
    for ax, variety in zip(axes, top_varieties):
        sub = pred_df_xgb[pred_df_xgb['Plant_Variety'] == variety].sort_values('Dispatch_Date')
        ax.plot(sub['Dispatch_Date'], sub['Actual'], label='Actual', color='steelblue', lw=1.2)
        ax.plot(sub['Dispatch_Date'], sub['Predicted'], label='XGBoost', color='darkorange', lw=1.2, ls='--')
        ax.set_title(f'{variety}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    fig.suptitle('XGBoost: Actual vs. Predicted (Top 5 Plant Varieties -- 2026 Test)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'xgb_actual_vs_predicted.png'), dpi=300)
    plt.close()

metrics_df = pd.DataFrame(all_metrics).sort_values('RMSE')
metrics_df.to_csv(os.path.join(results_dir, 'baseline_metrics.csv'), index=False)
print("\n" + "=" * 65)
print("  BASELINE METRICS SUMMARY")
print("=" * 65)
print(metrics_df.to_string(index=False))
print(f"\n[OK] Script 5 Complete! Best baseline: {metrics_df.iloc[0]['Model']}")

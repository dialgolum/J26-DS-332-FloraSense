"""
Script 8: Complete Overfitting / Underfitting Diagnostic -- ALL MODELS
=======================================================================
Checks ALL trained models (XGBoost, Random Forest, SARIMAX, Stacked Bi-LSTM, TFT)
and answers: Is each model overfitting, underfitting, or well-fitted?

Five diagnostic checks:
  CHECK 1  Train vs Test RMSE ratio   (primary indicator for all models)
  CHECK 2  Walk-Forward CV (5 folds)  (stability check, tree models only)
  CHECK 3  Temporal Degradation       (Early-2026 vs Late-2026 accuracy)
  CHECK 4  Residual Autocorrelation   (Ljung-Box -- did we miss time patterns?)
  CHECK 5  Learning Curves            (trees: RMSE vs #trees; LSTM: loss curves)

Run from project root AFTER Scripts 5 and 6:
    python scripts/c4_time_series/c4_08_overfit_check.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# -- Paths ----------------------------------------------------------------------
ML_DIR      = os.path.join('data', 'c4_time_series', 'ml_ready')
RESULTS_DIR = os.path.join('data', 'c4_time_series', 'results')
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
CKPT_DIR    = os.path.join('models', 'c4_time_series', 'checkpoints')
DIAG_DIR    = os.path.join(RESULTS_DIR, 'overfit_diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

SEQ_LEN    = 8    # must match Script 6
HIDDEN_DIM = 128
N_LAYERS   = 2
DROPOUT    = 0.15

print("=" * 68)
print("  Script 8: Overfit / Underfit / Accuracy Diagnostic -- ALL MODELS")
print("=" * 68)

# -- Prerequisites check --------------------------------------------------------
for p in [os.path.join(ML_DIR, 'train_data.csv'),
          os.path.join(ML_DIR, 'validation_data.csv'),
          os.path.join(ML_DIR, 'final_train_data.csv'),
          os.path.join(ML_DIR, 'test_data.csv'),
          os.path.join(ML_DIR, 'feature_meta.json')]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. Run c4_04_ml_preprocessing.py first.")

core_train_df = pd.read_csv(os.path.join(ML_DIR, 'train_data.csv'))
val_df = pd.read_csv(os.path.join(ML_DIR, 'validation_data.csv'))
train_df = pd.read_csv(os.path.join(ML_DIR, 'final_train_data.csv'))
test_df  = pd.read_csv(os.path.join(ML_DIR, 'test_data.csv'))

with open(os.path.join(ML_DIR, 'feature_meta.json')) as f:
    meta = json.load(f)

TARGET       = meta['target_col']
FEATURE_COLS = [c for c in meta['feature_cols']
                if c in train_df.columns and c in test_df.columns]
GROUP_COLS = meta['cat_cols']

X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df[TARGET].values.astype(np.float32)
X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_test  = test_df[TARGET].values.astype(np.float32)

print(f"\nFeatures : {len(FEATURE_COLS)}")
print(f"Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

# -- Helper functions -----------------------------------------------------------
def rmse(yt, yp):   return float(np.sqrt(mean_squared_error(yt, yp)))
def mae_(yt, yp):   return float(mean_absolute_error(yt, yp))
def mape_(yt, yp):
    nz = yt != 0
    return float(np.mean(np.abs((yt[nz]-yp[nz])/yt[nz]))*100) if nz.sum()>0 else np.nan
def r2_(yt, yp):    return float(r2_score(yt, yp))

all_verdicts = []   # will hold one dict per model

def print_and_store_verdict(name, tr_rmse, ts_rmse, tr_r2, ts_r2,
                             tr_mae, ts_mae, tr_mape, ts_mape):
    ratio = ts_rmse / tr_rmse if tr_rmse > 0 else float('inf')

    print(f"\n  {'Model':<28} {'Train':>10}  {'Test':>10}  {'Ratio':>7}")
    print(f"  {'-'*58}")
    print(f"  {'RMSE':<28} {tr_rmse:>10,.1f}  {ts_rmse:>10,.1f}  {ratio:>7.3f}")
    print(f"  {'MAE':<28} {tr_mae:>10,.1f}  {ts_mae:>10,.1f}")
    print(f"  {'MAPE %':<28} {tr_mape:>10.2f}  {ts_mape:>10.2f}")
    print(f"  {'R2':<28} {tr_r2:>10.4f}  {ts_r2:>10.4f}")

    if tr_rmse > 2500 and ts_rmse > 2500:
        verdict = "UNDERFITTING   -- both errors high, model too simple"
        flag = "[UNDERFIT]"
    elif ratio < 1.20:
        verdict = "GOOD FIT       -- train/test close, generalises well"
        flag = "[GOOD FIT]"
    elif ratio < 1.50:
        verdict = "MILD OVERFIT   -- test slightly higher, still acceptable"
        flag = "[MILD OVF]"
    else:
        verdict = "OVERFIT        -- test >> train, memorising training data"
        flag = "[OVERFIT] "

    print(f"\n  Verdict for {name}: {flag} {verdict}")

    all_verdicts.append({
        'Model': name, 'Flag': flag,
        'Train_RMSE': round(tr_rmse,2), 'Test_RMSE': round(ts_rmse,2),
        'Ratio': round(ratio,3),
        'Train_R2': round(tr_r2,4), 'Test_R2': round(ts_r2,4),
        'Train_MAE': round(tr_mae,2), 'Test_MAE': round(ts_mae,2),
        'Train_MAPE': round(tr_mape,2), 'Test_MAPE': round(ts_mape,2),
        'Verdict': verdict
    })


# ==============================================================================
# CHECK 1: TRAIN vs TEST METRICS -- ALL MODELS
# ==============================================================================
print("\n" + "=" * 68)
print("CHECK 1: Train vs Test Accuracy -- Every Model")
print("  (Ratio = Test RMSE / Train RMSE;  >1.5 = overfit,  <1.2 = good)")
print("=" * 68)

# -- XGBoost --------------------------------------------------------------------
xgb_ckpt = os.path.join(CKPT_DIR, 'xgboost_model.pkl')
if os.path.exists(xgb_ckpt):
    print("\n--- XGBoost ---")
    xgb = joblib.load(xgb_ckpt)
    yp_tr = xgb.predict(X_train)
    yp_ts = xgb.predict(X_test)
    print_and_store_verdict('XGBoost',
        rmse(y_train,yp_tr), rmse(y_test,yp_ts),
        r2_(y_train,yp_tr),  r2_(y_test,yp_ts),
        mae_(y_train,yp_tr), mae_(y_test,yp_ts),
        mape_(y_train,yp_tr),mape_(y_test,yp_ts))
else:
    print("\n  [XGBoost] checkpoint not found -- run c4_05_baseline_models.py first.")

# -- Random Forest --------------------------------------------------------------
rf_ckpt = os.path.join(CKPT_DIR, 'rf_model.pkl')
if os.path.exists(rf_ckpt):
    print("\n--- Random Forest ---")
    rf = joblib.load(rf_ckpt)
    yp_tr = rf.predict(X_train)
    yp_ts = rf.predict(X_test)
    print_and_store_verdict('Random Forest',
        rmse(y_train,yp_tr), rmse(y_test,yp_ts),
        r2_(y_train,yp_tr),  r2_(y_test,yp_ts),
        mae_(y_train,yp_tr), mae_(y_test,yp_ts),
        mape_(y_train,yp_tr),mape_(y_test,yp_ts))
else:
    print("\n  [Random Forest] checkpoint not found -- run c4_05_baseline_models.py first.")

# -- SARIMAX (weekly-aggregate only -> test set only) ---------------------------
sarimax_pred_path = os.path.join(RESULTS_DIR, 'predictions_sarimax.csv')
if os.path.exists(sarimax_pred_path):
    print("\n--- SARIMAX (weekly aggregate) ---")
    sdf = pd.read_csv(sarimax_pred_path)
    if 'Actual_Weekly_Total' in sdf.columns:
        yt = sdf['Actual_Weekly_Total'].values.astype(float)
        yp = sdf['Predicted_Weekly_Total'].values.astype(float)
        print(f"  (SARIMAX is a univariate weekly model -- train metrics not applicable)")
        print(f"  Test  RMSE : {rmse(yt,yp):>10,.1f}")
        print(f"  Test  MAE  : {mae_(yt,yp):>10,.1f}")
        print(f"  Test  MAPE : {mape_(yt,yp):>9.2f}%")
        print(f"  Test  R2   : {r2_(yt,yp):>10.4f}")
        all_verdicts.append({
            'Model': 'SARIMAX (weekly)', 'Flag': '[TEST-ONLY]',
            'Train_RMSE': np.nan, 'Test_RMSE': round(rmse(yt,yp),2),
            'Ratio': np.nan, 'Train_R2': np.nan, 'Test_R2': round(r2_(yt,yp),4),
            'Train_MAE': np.nan, 'Test_MAE': round(mae_(yt,yp),2),
            'Train_MAPE': np.nan, 'Test_MAPE': round(mape_(yt,yp),2),
            'Verdict': 'Test-only model (univariate weekly aggregate)'
        })
else:
    print("\n  [SARIMAX] predictions not found -- run c4_05_baseline_models.py first.")

# -- Stacked Bi-LSTM ------------------------------------------------------------
lstm_ckpt = os.path.join(CKPT_DIR, 'lstm_best.pt')
lstm_pred_path = os.path.join(RESULTS_DIR, 'predictions_stacked_lstm.csv')

if os.path.exists(lstm_ckpt):
    print("\n--- Stacked Bi-LSTM ---")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        import joblib as _jl
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_scaler = _jl.load(os.path.join(ML_DIR, 'target_scaler.pkl'))

        def build_group_sequences(context_df, target_years):
            Xs, ys = [], []
            ordered = context_df.sort_values(GROUP_COLS + ['Dispatch_Date'])
            for _, group in ordered.groupby(GROUP_COLS, sort=False, dropna=False):
                xg = group[FEATURE_COLS].to_numpy(dtype=np.float32)
                yg = y_scaler.transform(group[[TARGET]]).flatten().astype(np.float32)
                for i, year in enumerate(group['YEAR'].to_numpy()):
                    if year not in target_years:
                        continue
                    window = xg[max(0, i - SEQ_LEN + 1):i + 1]
                    if len(window) < SEQ_LEN:
                        window = np.vstack([np.repeat(window[:1], SEQ_LEN-len(window), axis=0), window])
                    Xs.append(window); ys.append(yg[i])
            return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

        X_tr_seq, y_tr_seq = build_group_sequences(train_df, {2023, 2024, 2025})
        test_context = pd.concat([train_df, test_df], ignore_index=True)
        X_ts_seq, y_ts_seq = build_group_sequences(test_context, {2026})

        class TimeSeriesDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X); self.y = torch.tensor(y)
            def __len__(self): return len(self.y)
            def __getitem__(self, i): return self.X[i], self.y[i]

        class StackedLSTM(nn.Module):
            """Must EXACTLY match Script 6: residual skip, no attention."""
            def __init__(self, input_dim, hidden_dim, n_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                                    batch_first=True,
                                    dropout=dropout if n_layers > 1 else 0.0,
                                    bidirectional=True)
                # Residual skip: concat last timestep input features
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 2 + input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 1))
            def forward(self, x):
                lstm_out, _ = self.lstm(x)          # (B, seq, hidden*2)
                last_h = lstm_out[:, -1, :]         # (B, hidden*2)
                last_x = x[:, -1, :]                # (B, input_dim)  residual
                combined = torch.cat([last_h, last_x], dim=-1)
                return self.fc(combined).squeeze(-1)

        n_feat = X_tr_seq.shape[2]
        model  = StackedLSTM(n_feat, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
        model.load_state_dict(torch.load(lstm_ckpt, map_location=DEVICE, weights_only=True))
        model.eval()

        def predict_lstm_inv(X_seq):
            """Predict in scaled space, inverse-transform back to original units."""
            loader = DataLoader(TimeSeriesDataset(X_seq,
                                np.zeros(len(X_seq), dtype=np.float32)),
                                batch_size=256, shuffle=False)
            preds_sc = []
            with torch.no_grad():
                for Xb, _ in loader:
                    preds_sc.extend(model(Xb.to(DEVICE)).cpu().numpy())
            preds_sc = np.array(preds_sc).reshape(-1, 1)
            return np.maximum(0, y_scaler.inverse_transform(preds_sc).flatten())

        # Inverse-transform targets too for fair metric comparison
        y_tr_orig = y_scaler.inverse_transform(y_tr_seq.reshape(-1,1)).flatten()
        y_ts_orig = y_scaler.inverse_transform(y_ts_seq.reshape(-1,1)).flatten()
        yp_tr     = predict_lstm_inv(X_tr_seq)
        yp_ts     = predict_lstm_inv(X_ts_seq)

        print_and_store_verdict('Stacked Bi-LSTM',
            rmse(y_tr_orig, yp_tr), rmse(y_ts_orig, yp_ts),
            r2_(y_tr_orig, yp_tr),  r2_(y_ts_orig, yp_ts),
            mae_(y_tr_orig, yp_tr), mae_(y_ts_orig, yp_ts),
            mape_(y_tr_orig,yp_tr), mape_(y_ts_orig,yp_ts))

        LSTM_TRAIN_PREDS = yp_tr
        LSTM_TEST_PREDS  = yp_ts
        LSTM_Y_TRAIN     = y_tr_orig
        LSTM_Y_TEST      = y_ts_orig

    except Exception as e:
        print(f"  LSTM load failed: {e}")
        LSTM_TRAIN_PREDS = None
else:
    print("\n  [Stacked Bi-LSTM] checkpoint not found -- run c4_06_deep_learning_models.py first.")
    LSTM_TRAIN_PREDS = None

# -- TFT ------------------------------------------------------------------------
tft_pred_path = os.path.join(RESULTS_DIR, 'predictions_tft.csv')
if os.path.exists(tft_pred_path):
    print("\n--- TFT ---")
    tdf = pd.read_csv(tft_pred_path)
    if 'Actual' in tdf.columns and 'Predicted' in tdf.columns:
        yt = tdf['Actual'].values.astype(float)
        yp = tdf['Predicted'].values.astype(float)
        print(f"  Test  RMSE : {rmse(yt,yp):>10,.1f}")
        print(f"  Test  MAE  : {mae_(yt,yp):>10,.1f}")
        print(f"  Test  MAPE : {mape_(yt,yp):>9.2f}%")
        print(f"  Test  R2   : {r2_(yt,yp):>10.4f}")

        # Recreate the final TFT training dataset to compare in-sample fit with
        # the untouched 2026 test predictions. No epoch selection uses 2026.
        try:
            import torch
            import lightning.pytorch as pl
            from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
            from pytorch_forecasting.metrics import MAE as PFMAE
            from pytorch_forecasting.data import GroupNormalizer

            master = pd.read_csv(
                os.path.join('data', 'c4_time_series', 'processed', 'c4_master_training_data.csv'),
                parse_dates=['Dispatch_Date']
            ).sort_values('Dispatch_Date').reset_index(drop=True)
            master['WEEK_sin'] = np.sin(2 * np.pi * master['WEEK'] / 52.0)
            master['WEEK_cos'] = np.cos(2 * np.pi * master['WEEK'] / 52.0)
            master['time_idx'] = master.groupby(GROUP_COLS, sort=False).cumcount()
            master[GROUP_COLS] = master[GROUP_COLS].astype(str)
            tft_weather_cols = ['Avg_Max_Temp_C', 'Weekly_Rainfall_mm',
                                'Growing_Degree_Days_Weekly', 'Avg_Max_Temp_C_Lag2W',
                                'Weekly_Rainfall_mm_Lag2W', 'Growing_Degree_Days_Weekly_Lag2W']
            tft_missing_cols = []
            for col in tft_weather_cols:
                flag_col = f'{col}_missing'
                master[flag_col] = master[col].isna().astype(np.float32)
                tft_missing_cols.append(flag_col)
            tft_medians = master.loc[master['YEAR'] < 2025, tft_weather_cols].median()
            master[tft_weather_cols] = master[tft_weather_cols].fillna(tft_medians)
            float_cols = [TARGET, 'Ordered_Quantity', *tft_weather_cols,
                          'WEEK_sin', 'WEEK_cos', *tft_missing_cols]
            master[float_cols] = master[float_cols].astype(np.float32)
            fit_rows = master[master['YEAR'] < 2026].copy()
            tft_ds = TimeSeriesDataSet(
                fit_rows, time_idx='time_idx', target=TARGET, group_ids=GROUP_COLS,
                min_encoder_length=1, max_encoder_length=SEQ_LEN,
                min_prediction_length=1, max_prediction_length=1,
                static_categoricals=GROUP_COLS,
                time_varying_known_reals=['time_idx', 'WEEK_sin', 'WEEK_cos',
                                          'Ordered_Quantity', 'Avg_Max_Temp_C',
                                          'Weekly_Rainfall_mm', 'Growing_Degree_Days_Weekly',
                                          'Avg_Max_Temp_C_missing', 'Weekly_Rainfall_mm_missing',
                                          'Growing_Degree_Days_Weekly_missing'],
                time_varying_unknown_reals=[TARGET, 'Avg_Max_Temp_C_Lag2W',
                                            'Weekly_Rainfall_mm_Lag2W', 'Growing_Degree_Days_Weekly_Lag2W',
                                            'Avg_Max_Temp_C_Lag2W_missing',
                                            'Weekly_Rainfall_mm_Lag2W_missing',
                                            'Growing_Degree_Days_Weekly_Lag2W_missing'],
                target_normalizer=GroupNormalizer(groups=GROUP_COLS),
                add_relative_time_idx=True, add_target_scales=True,
                add_encoder_length=True, allow_missing_timesteps=True
            )
            fit_eval_ds = TimeSeriesDataSet.from_dataset(
                tft_ds, fit_rows, predict=False, stop_randomization=True
            ).filter(lambda idx: (
                (idx['time_idx_first_prediction'] - idx['time_idx_first'] == SEQ_LEN)
            ))
            fit_loader = fit_eval_ds.to_dataloader(train=False, batch_size=256, num_workers=0)
            tft_model = TemporalFusionTransformer.from_dataset(
                tft_ds, learning_rate=1e-3, hidden_size=64, attention_head_size=4,
                dropout=0.15, hidden_continuous_size=32, output_size=1, loss=PFMAE()
            )
            tft_ckpt = os.path.join(CKPT_DIR, 'tft_final.pt')
            tft_model.load_state_dict(torch.load(tft_ckpt, map_location='cpu', weights_only=True))
            fit_pred = tft_model.predict(fit_loader, return_y=True,
                                         trainer_kwargs={'accelerator': 'auto'})
            ytr = fit_pred.y[0].cpu().numpy().reshape(-1)
            ptr = fit_pred.output.cpu().numpy().reshape(-1)
            print_and_store_verdict('TFT', rmse(ytr,ptr), rmse(yt,yp),
                r2_(ytr,ptr), r2_(yt,yp), mae_(ytr,ptr), mae_(yt,yp),
                mape_(ytr,ptr), mape_(yt,yp))
        except Exception as e:
            print(f"  TFT train-fit comparison unavailable: {e}")
            all_verdicts.append({
                'Model': 'TFT', 'Flag': '[TEST-ONLY]',
                'Train_RMSE': np.nan, 'Test_RMSE': round(rmse(yt,yp),2),
                'Ratio': np.nan, 'Train_R2': np.nan, 'Test_R2': round(r2_(yt,yp),4),
                'Train_MAE': np.nan, 'Test_MAE': round(mae_(yt,yp),2),
                'Train_MAPE': np.nan, 'Test_MAPE': round(mape_(yt,yp),2),
                'Verdict': '2025 validation used for epoch selection; train-fit comparison unavailable'
            })
else:
    print("\n  [TFT] predictions not found -- run c4_06_deep_learning_models.py first.")


# ==============================================================================
# CHECK 2: WALK-FORWARD CROSS-VALIDATION (Tree models)
# ==============================================================================
print("\n" + "=" * 68)
print("CHECK 2: Walk-Forward Cross-Validation -- 5 Folds (Tree Models)")
print("  High variance across folds = unstable / prone to overfit")
print("=" * 68)

tscv = TimeSeriesSplit(n_splits=5)
cv_summary = []

for ModelClass, params, label in [
    (XGBRegressor,
     dict(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8,
          colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.1,
          reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0),
     'XGBoost'),
    (RandomForestRegressor,
     dict(n_estimators=150, max_depth=15, min_samples_leaf=5,
          max_features='sqrt', random_state=42, n_jobs=-1),
     'Random Forest'),
]:
    fold_train_rmse, fold_val_rmse = [], []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xvl, yvl = X_train[val_idx], y_train[val_idx]
        m = ModelClass(**params)
        m.fit(Xtr, ytr)
        fold_train_rmse.append(rmse(ytr, m.predict(Xtr)))
        fold_val_rmse.append(rmse(yvl, m.predict(Xvl)))

    mean_val   = float(np.mean(fold_val_rmse))
    std_val    = float(np.std(fold_val_rmse))
    cv_pct     = (std_val / mean_val) * 100

    print(f"\n  {label}")
    print(f"  Fold  : {'  '.join(f'F{i+1}' for i in range(5))}")
    print(f"  Train : {'  '.join(f'{v:5.0f}' for v in fold_train_rmse)}")
    print(f"  Val   : {'  '.join(f'{v:5.0f}' for v in fold_val_rmse)}")
    print(f"  Mean Val RMSE: {mean_val:,.1f}  Std: {std_val:,.1f}  CV%: {cv_pct:.1f}%")

    if cv_pct < 15:   stability = "Stable -- consistent across all time periods"
    elif cv_pct < 30: stability = "Moderate -- some sensitivity to training period"
    else:             stability = "Unstable -- large spread, likely to overfit on some periods"
    print(f"  --> {stability}")

    cv_summary.append({
        'model': label, 'folds': fold_val_rmse,
        'mean': mean_val, 'std': std_val, 'cv_pct': cv_pct
    })

    # Plot
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ['#4CAF50' if v <= mean_val * 1.2 else '#F44336' for v in fold_val_rmse]
    ax.bar(range(1, 6), fold_val_rmse, color=colors, edgecolor='white', alpha=0.85, label='Val RMSE')
    ax.plot(range(1, 6), fold_train_rmse, 'o--', color='steelblue', lw=1.5, label='Train RMSE')
    ax.axhline(mean_val, color='navy', lw=1.5, ls='--', label=f'Mean={mean_val:.0f}')
    ax.fill_between([-0.5, 5.5], mean_val-std_val, mean_val+std_val,
                    alpha=0.15, color='navy', label=f'+-1std={std_val:.0f}')
    ax.set_xticks(range(1,6)); ax.set_xlabel('Fold (chronological)'); ax.set_ylabel('RMSE')
    ax.set_title(f'{label} -- Walk-Forward CV (Train vs Val RMSE per Fold)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_DIR, f'{label.lower().replace(" ","_")}_cv_folds.png'), dpi=300)
    plt.close()
    print(f"  Saved: {label.lower().replace(' ','_')}_cv_folds.png")


# ==============================================================================
# CHECK 3: TEMPORAL DEGRADATION -- Early 2026 vs Late 2026
# ==============================================================================
print("\n" + "=" * 68)
print("CHECK 3: Temporal Degradation -- Early (Wks 1-25) vs Late (Wks 26+)")
print("  Unique to time series: does accuracy drop over time? (concept drift)")
print("=" * 68)

drift_results = []

if 'WEEK' in test_df.columns:
    early_mask = (test_df['WEEK'] <= 25).values
    late_mask  = (test_df['WEEK'] >  25).values
    print(f"  Early 2026 (Weeks 1-25) : {early_mask.sum():,} rows")
    print(f"  Late  2026 (Weeks 26+)  : {late_mask.sum():,} rows\n")

    # Tree models
    for label, ckpt_file in [('XGBoost', 'xgboost_model.pkl'),
                              ('Random Forest', 'rf_model.pkl')]:
        cp = os.path.join(CKPT_DIR, ckpt_file)
        if not os.path.exists(cp): continue
        m   = joblib.load(cp)
        yp  = m.predict(X_test)
        e_r = rmse(y_test[early_mask], yp[early_mask]) if early_mask.sum()>0 else np.nan
        l_r = rmse(y_test[late_mask],  yp[late_mask])  if late_mask.sum()>0  else np.nan
        drift = l_r/e_r if e_r>0 else np.nan
        flag  = ("No drift (stable)" if drift < 1.15 else
                 "Mild drift" if drift < 1.40 else
                 "Significant drift -- consider rolling retrain")
        print(f"  {label:<22} Early RMSE: {e_r:,.1f}  Late RMSE: {l_r:,.1f}  Ratio: {drift:.3f} --> {flag}")
        drift_results.append({'Model': label, 'Early_RMSE': e_r, 'Late_RMSE': l_r, 'Drift_Ratio': drift, 'Flag': flag})

    # LSTM
    if LSTM_TRAIN_PREDS is not None:
        # Script 6 creates one prediction for every 2026 row using grouped history.
        early_m_lstm = early_mask
        late_m_lstm  = late_mask
        e_r = rmse(LSTM_Y_TEST[early_m_lstm], LSTM_TEST_PREDS[early_m_lstm]) if early_m_lstm.sum()>0 else np.nan
        l_r = rmse(LSTM_Y_TEST[late_m_lstm],  LSTM_TEST_PREDS[late_m_lstm])  if late_m_lstm.sum()>0  else np.nan
        drift = l_r/e_r if e_r>0 else np.nan
        flag  = ("No drift (stable)" if drift < 1.15 else
                 "Mild drift" if drift < 1.40 else
                 "Significant drift -- consider rolling retrain")
        print(f"  {'Stacked Bi-LSTM':<22} Early RMSE: {e_r:,.1f}  Late RMSE: {l_r:,.1f}  Ratio: {drift:.3f} --> {flag}")
        drift_results.append({'Model': 'Stacked Bi-LSTM', 'Early_RMSE': e_r, 'Late_RMSE': l_r, 'Drift_Ratio': drift, 'Flag': flag})
else:
    print("  WEEK column not found in test set -- skipping.")


# ==============================================================================
# CHECK 4: RESIDUAL AUTOCORRELATION (Ljung-Box)
# ==============================================================================
print("\n" + "=" * 68)
print("CHECK 4: Residual Autocorrelation -- Ljung-Box Test")
print("  p > 0.05: residuals are random (good) | p < 0.05: pattern remains (underfit)")
print("=" * 68)

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf

    pred_sources = []
    # Tree models from predictions CSVs
    for label, pred_file in [('XGBoost',      'predictions_xgboost.csv'),
                              ('Random Forest','predictions_random_forest.csv'),
                              ('SARIMAX',      'predictions_sarimax.csv')]:
        pp = os.path.join(RESULTS_DIR, pred_file)
        if not os.path.exists(pp): continue
        pdf = pd.read_csv(pp)
        if 'Actual' in pdf.columns:
            pred_sources.append((label, pdf['Actual'].values.astype(float),
                                          pdf['Predicted'].values.astype(float), pdf))
        elif 'Actual_Weekly_Total' in pdf.columns:
            pred_sources.append((label, pdf['Actual_Weekly_Total'].values.astype(float),
                                          pdf['Predicted_Weekly_Total'].values.astype(float), pdf))

    # LSTM from live predictions
    if LSTM_TRAIN_PREDS is not None:
        pred_sources.append(('Stacked Bi-LSTM', LSTM_Y_TEST, LSTM_TEST_PREDS, None))

    for label, yt, yp, pdf in pred_sources:
        residuals = yt - yp
        # Aggregate to weekly to remove within-week duplication for tree models
        if pdf is not None and 'WEEK' in pdf.columns and 'YEAR' in pdf.columns:
            wkly = pd.Series(residuals, name='r').groupby(
                [pdf['YEAR'], pdf['WEEK']]).mean().values
        else:
            wkly = residuals

        lb  = acorr_ljungbox(wkly, lags=[10], return_df=True)
        pv  = float(lb['lb_pvalue'].values[0])
        stat= float(lb['lb_stat'].values[0])
        interp = ("White noise (no pattern left)" if pv > 0.05 else
                  "Autocorrelated -- model missed time structure")
        flag   = "GOOD" if pv > 0.05 else "WARN"
        print(f"  [{flag}] {label:<22} Ljung-Box stat={stat:.2f}  p={pv:.4f}  --> {interp}")

        # ACF plot
        fig, ax = plt.subplots(figsize=(9, 3.5))
        plot_acf(wkly, lags=min(20, len(wkly)//2-1), ax=ax, alpha=0.05)
        title_str = (f'{label} -- Residual ACF\n'
                     f'Ljung-Box p={pv:.4f}  '
                     f'{"[White Noise - Good]" if pv>0.05 else "[Autocorrelated - Pattern Remains]"}')
        ax.set_title(title_str, fontsize=10, fontweight='bold')
        plt.tight_layout()
        safe = label.lower().replace(' ', '_').replace('-','_')
        plt.savefig(os.path.join(DIAG_DIR, f'{safe}_residual_acf.png'), dpi=300)
        plt.close()

except ImportError:
    print("  statsmodels not installed -- skipping. pip install statsmodels")
except Exception as e:
    print(f"  Ljung-Box test failed: {e}")


# ==============================================================================
# CHECK 5: LEARNING CURVES
# 5A: XGBoost -- RMSE vs n_estimators (shows when overfitting begins)
# 5B: LSTM    -- reference the saved training curve image + compute gap
# ==============================================================================
print("\n" + "=" * 68)
print("CHECK 5: Learning Curves")
print("  Ideal: train & test converge | Overfit: gap widens | Underfit: both plateau high")
print("=" * 68)

# 5A: XGBoost learning curve using 2025 validation only (never the 2026 test set)
print("\n  5A. XGBoost -- RMSE vs Number of Trees (estimators 50 to 500)")
n_tree_range    = [50, 100, 150, 200, 300, 400, 500]
lc_train, lc_test = [], []
X_curve_train = core_train_df[FEATURE_COLS].values.astype(np.float32)
y_curve_train = core_train_df[TARGET].values.astype(np.float32)
X_curve_val = val_df[FEATURE_COLS].values.astype(np.float32)
y_curve_val = val_df[TARGET].values.astype(np.float32)

for n in n_tree_range:
    m = XGBRegressor(n_estimators=n, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                     reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                     n_jobs=-1, verbosity=0)
    m.fit(X_curve_train, y_curve_train)
    lc_train.append(rmse(y_curve_train, m.predict(X_curve_train)))
    lc_test.append( rmse(y_curve_val,  m.predict(X_curve_val)))
    print(f"    n={n:4d}  Train RMSE: {lc_train[-1]:>8,.1f}  |  2025 Val RMSE: {lc_test[-1]:>8,.1f}")

opt_idx = int(np.argmin(lc_test))
print(f"\n  Lowest 2025 validation RMSE: {n_tree_range[opt_idx]} trees ({lc_test[opt_idx]:,.1f})")
if opt_idx < len(n_tree_range)-1:
    print(f"  WARNING: Validation RMSE increases after {n_tree_range[opt_idx]} trees -- slight overfit beyond this")
else:
    print(f"  GOOD: Validation RMSE still decreasing at 500 trees -- no overfit in this range")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(n_tree_range, lc_train, 'o-', color='steelblue',  lw=1.5, label='Train RMSE')
ax.plot(n_tree_range, lc_test,  's-', color='darkorange', lw=1.5, label='Validation RMSE (2025)')
ax.axvline(n_tree_range[opt_idx], color='green', lw=1.2, ls=':', label=f'Optimal ({n_tree_range[opt_idx]} trees)')
ax.fill_between(n_tree_range, lc_train, lc_test,
                alpha=0.12, color='red', label='Generalisation Gap')
ax.set_xlabel('n_estimators (Number of Trees)', fontsize=10)
ax.set_ylabel('RMSE', fontsize=10)
ax.set_title('XGBoost Learning Curve -- Train vs Test RMSE\n'
             'Large red gap = overfit  |  Both high = underfit  |  Converging = good fit',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIAG_DIR, 'xgboost_learning_curve.png'), dpi=300)
plt.close()
print(f"  Saved: xgboost_learning_curve.png")

# 5B: LSTM -- parse existing learning curve PNG reference + compute train-test gap
print("\n  5B. Stacked Bi-LSTM -- Learning Curve Analysis")
lstm_lc_path = os.path.join(PLOTS_DIR, 'lstm_learning_curve.png')
if os.path.exists(lstm_lc_path):
    print(f"  LSTM learning curve image exists: {lstm_lc_path}")
    print(f"  Interpretation guide:")
    print(f"    - If orange (val) line falls and then rises while blue (train) keeps falling -> OVERFIT")
    print(f"    - If both lines fall and stabilise close together -> GOOD FIT")
    print(f"    - If both lines stay high and barely move -> UNDERFITTING (increase model size)")
    if LSTM_TRAIN_PREDS is not None:
        tr_r = rmse(LSTM_Y_TRAIN, LSTM_TRAIN_PREDS)
        ts_r = rmse(LSTM_Y_TEST,  LSTM_TEST_PREDS)
        ratio = ts_r/tr_r
        print(f"\n  LSTM Train RMSE: {tr_r:,.1f}  |  Test RMSE: {ts_r:,.1f}  |  Ratio: {ratio:.3f}")
        if ratio < 1.2:
            print(f"  Train/test RMSE are close; no large generalization gap. "
                  f"Check R2/MAPE above for absolute accuracy.")
        elif ratio < 1.5:
            print(f"  MILD OVERFIT -- test is somewhat higher than train")
        else:
            print(f"  OVERFIT -- test is significantly higher than train")
else:
    print(f"  LSTM learning curve image not found (run c4_06_deep_learning_models.py first)")


# ==============================================================================
# FINAL DASHBOARD -- Consolidated comparison chart + summary table
# ==============================================================================
print("\n" + "=" * 68)
print("  FINAL ACCURACY & FIT COMPARISON -- ALL MODELS")
print("=" * 68)

if all_verdicts:
    vdf = pd.DataFrame(all_verdicts)

    # Print summary table
    print(f"\n{'Model':<26} {'Flag':<12} {'Train_RMSE':>11} {'Test_RMSE':>10} {'Ratio':>7} {'Test_R2':>8} {'Test_MAPE%':>10}")
    print("-" * 88)
    for _, row in vdf.iterrows():
        tr = f"{row['Train_RMSE']:>11,.1f}" if not pd.isna(row['Train_RMSE']) else f"{'N/A':>11}"
        rt = f"{row['Ratio']:>7.3f}"         if not pd.isna(row['Ratio'])       else f"{'N/A':>7}"
        print(f"{row['Model']:<26} {row['Flag']:<12} {tr} {row['Test_RMSE']:>10,.1f} {rt} "
              f"{row['Test_R2']:>8.4f} {row['Test_MAPE']:>10.2f}")
    print()
    for _, row in vdf.iterrows():
        print(f"  {row['Model']}: {row['Verdict']}")

    # Save CSV
    vdf.to_csv(os.path.join(DIAG_DIR, 'overfit_summary.csv'), index=False)
    print(f"\n  Saved: overfit_summary.csv")

    # -- Dashboard chart --------------------------------------------------------
    # Models that have both train AND test RMSE
    comp_df = vdf.dropna(subset=['Train_RMSE', 'Test_RMSE']).copy()
    models_with_both = list(comp_df['Model'])

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('FloraSense C4 -- Model Accuracy & Overfitting Dashboard (2026 Test Set)',
                 fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Plot 1: Test RMSE comparison (ALL models)
    ax1 = fig.add_subplot(gs[0, 0])
    colors1 = ['#4CAF50' if r < 1.2 else '#FF9800' if r < 1.5 else '#F44336'
                for r in (vdf['Ratio'].fillna(1.0))]
    bars = ax1.barh(vdf['Model'], vdf['Test_RMSE'], color=colors1, edgecolor='white', alpha=0.85)
    for bar, val in zip(bars, vdf['Test_RMSE']):
        ax1.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                 f'{val:,.0f}', va='center', fontsize=8)
    ax1.set_xlabel('RMSE (lower is better)'); ax1.set_title('Test RMSE -- All Models', fontweight='bold', fontsize=10)
    ax1.tick_params(axis='y', labelsize=8)

    # Plot 2: Test R2 comparison (ALL models)
    ax2 = fig.add_subplot(gs[0, 1])
    r2_colors = ['#4CAF50' if r > 0.7 else '#FF9800' if r > 0.4 else '#F44336'
                 for r in vdf['Test_R2']]
    bars2 = ax2.barh(vdf['Model'], vdf['Test_R2'], color=r2_colors, edgecolor='white', alpha=0.85)
    for bar, val in zip(bars2, vdf['Test_R2']):
        ax2.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8)
    ax2.set_xlabel('R2 (higher is better)'); ax2.set_title('Test R2 -- All Models', fontweight='bold', fontsize=10)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.axvline(0, color='black', lw=0.8, ls='--')

    # Plot 3: MAPE comparison (ALL models)
    ax3 = fig.add_subplot(gs[0, 2])
    mape_colors = ['#4CAF50' if m < 15 else '#FF9800' if m < 25 else '#F44336'
                   for m in vdf['Test_MAPE']]
    bars3 = ax3.barh(vdf['Model'], vdf['Test_MAPE'], color=mape_colors, edgecolor='white', alpha=0.85)
    for bar, val in zip(bars3, vdf['Test_MAPE']):
        ax3.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=8)
    ax3.set_xlabel('MAPE % (lower is better)'); ax3.set_title('Test MAPE % -- All Models', fontweight='bold', fontsize=10)
    ax3.tick_params(axis='y', labelsize=8)

    # Plot 4: Train vs Test RMSE (models with both)
    ax4 = fig.add_subplot(gs[1, 0])
    if len(comp_df) > 0:
        x = np.arange(len(comp_df))
        w = 0.35
        ax4.bar(x-w/2, comp_df['Train_RMSE'], w, label='Train RMSE', color='steelblue',   alpha=0.85)
        ax4.bar(x+w/2, comp_df['Test_RMSE'],  w, label='Test RMSE',  color='darkorange', alpha=0.85)
        ax4.set_xticks(x); ax4.set_xticklabels(comp_df['Model'], rotation=20, ha='right', fontsize=8)
        ax4.set_ylabel('RMSE'); ax4.legend(fontsize=8)
        ax4.set_title('Train vs Test RMSE\n(gap = overfitting)', fontweight='bold', fontsize=10)

    # Plot 5: Generalisation Ratio
    ax5 = fig.add_subplot(gs[1, 1])
    if len(comp_df) > 0:
        ratio_colors = ['#4CAF50' if r < 1.2 else '#FF9800' if r < 1.5 else '#F44336'
                        for r in comp_df['Ratio']]
        ax5.bar(comp_df['Model'], comp_df['Ratio'], color=ratio_colors, edgecolor='white', alpha=0.85)
        ax5.axhline(1.0,  color='green',  lw=1.2, ls='--', label='Perfect (1.0)')
        ax5.axhline(1.2,  color='orange', lw=1.2, ls='--', label='Mild overfit (1.2)')
        ax5.axhline(1.5,  color='red',    lw=1.2, ls='--', label='Overfit threshold (1.5)')
        ax5.set_xticks(range(len(comp_df['Model'])))
        ax5.set_xticklabels(comp_df['Model'], rotation=20, ha='right', fontsize=8)
        ax5.set_ylabel('Test/Train RMSE Ratio')
        ax5.set_title('Generalisation Ratio\n(<1.2 = good fit)', fontweight='bold', fontsize=10)
        ax5.legend(fontsize=7)

    # Plot 6: Verdict legend
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    legend_text = (
        "HOW TO READ THIS DASHBOARD\n\n"
        "RMSE / MAE / MAPE:\n"
        "  Lower = better accuracy\n\n"
        "R2 Score:\n"
        "  >0.7  Excellent\n"
        "  0.4-0.7  Acceptable\n"
        "  <0.4  Poor\n\n"
        "MAPE %:\n"
        "  <15%  Excellent\n"
        "  15-25%  Acceptable\n"
        "  >25%  Poor\n\n"
        "Train/Test Ratio:\n"
        "  <1.20  Good Fit\n"
        "  1.20-1.50  Mild Overfit\n"
        "  >1.50  Significant Overfit\n\n"
        "Bar Colours:\n"
        "  Green = Good\n"
        "  Orange = Caution\n"
        "  Red = Problem"
    )
    ax6.text(0.02, 0.98, legend_text, va='top', ha='left', fontsize=8.5,
             transform=ax6.transAxes, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.8))
    ax6.set_title('Interpretation Guide', fontweight='bold', fontsize=10)

    plt.savefig(os.path.join(DIAG_DIR, 'MASTER_accuracy_overfit_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: MASTER_accuracy_overfit_dashboard.png")

# -- Temporal degradation chart -------------------------------------------------
if drift_results:
    ddf = pd.DataFrame(drift_results)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(ddf)); w = 0.35
    ax.bar(x-w/2, ddf['Early_RMSE'], w, label='Early 2026 (Wks 1-25)', color='steelblue',  alpha=0.85)
    ax.bar(x+w/2, ddf['Late_RMSE'],  w, label='Late 2026  (Wks 26+)',  color='tomato', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ddf['Model'], fontsize=9)
    ax.set_ylabel('RMSE'); ax.legend(fontsize=9)
    ax.set_title('Temporal Degradation: Early vs Late 2026 Test Accuracy\n'
                 'Large red > blue = concept drift / model getting worse over time',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_DIR, 'temporal_degradation.png'), dpi=300)
    plt.close()
    print(f"  Saved: temporal_degradation.png")

# -- Print run-order reminder ----------------------------------------------------
print("""
  ----------------------------------------------------------------
  FILES GENERATED in data/c4_time_series/results/overfit_diagnostics/
  ----------------------------------------------------------------
  MASTER_accuracy_overfit_dashboard.png  <- START HERE, shows all models
  xgboost_learning_curve.png             <- overfit point for XGBoost
  xgboost_cv_folds.png                   <- fold stability XGBoost
  random_forest_cv_folds.png             <- fold stability RF
  temporal_degradation.png               <- concept drift check
  *_residual_acf.png                     <- autocorrelation per model
  overfit_summary.csv                    <- full table with verdicts
  ----------------------------------------------------------------
  For LSTM: check experiments/results/plots/lstm_learning_curve.png
            (train loss blue vs val loss orange -- gap = overfit)
  ----------------------------------------------------------------
""")
print(f"Done! All diagnostic files saved to: {DIAG_DIR}")

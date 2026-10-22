import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

print("=" * 65)
print("  Script 6: Deep Learning -- Stacked LSTM + TFT")
print("=" * 65)

ML_DIR      = os.path.join('data', 'c4_time_series', 'ml_ready')
RESULTS_DIR = os.path.join('data', 'c4_time_series', 'results')
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
CKPT_DIR    = os.path.join('models', 'c4_time_series', 'checkpoints')
for d in [RESULTS_DIR, PLOTS_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

SEQ_LEN    = 8
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 1e-3
HIDDEN_DIM = 128
N_LAYERS   = 2
DROPOUT    = 0.15
PATIENCE   = 10

train_df = pd.read_csv(os.path.join(ML_DIR, 'train_data.csv'))
val_df   = pd.read_csv(os.path.join(ML_DIR, 'validation_data.csv'))
final_train_df = pd.read_csv(os.path.join(ML_DIR, 'final_train_data.csv'))
test_df  = pd.read_csv(os.path.join(ML_DIR, 'test_data.csv'))
y_scaler = joblib.load(os.path.join(ML_DIR, 'target_scaler.pkl'))

with open(os.path.join(ML_DIR, 'feature_meta.json')) as f:
    meta = json.load(f)

TARGET       = meta['target_col']
GROUP_COLS   = meta['cat_cols']
FEATURE_COLS = [c for c in meta['feature_cols']
                if c in train_df.columns and c in val_df.columns
                and c in final_train_df.columns and c in test_df.columns]

print(f"Features: {len(FEATURE_COLS)} | Train rows: {len(train_df):,} | "
      f"Validation rows: {len(val_df):,} | Final train rows: {len(final_train_df):,} | "
      f"Test rows: {len(test_df):,}")

# Make one sequence per customer/variety/size/mode series. Validation and test
# windows include prior feature history, but each target row belongs to exactly
# one split. This also prevents windows from crossing between unrelated series.
def build_sequences(context_df: pd.DataFrame, target_years):
    ordered = context_df.sort_values(GROUP_COLS + ['Dispatch_Date']).copy()
    X_seqs, y_targets, target_rows = [], [], []
    for _, group in ordered.groupby(GROUP_COLS, sort=False, dropna=False):
        X_group = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_group = y_scaler.transform(group[[TARGET]]).flatten().astype(np.float32)
        years = group['YEAR'].to_numpy()
        for i, year in enumerate(years):
            if year not in target_years:
                continue
            start = max(0, i - SEQ_LEN + 1)
            window = X_group[start:i + 1]
            if len(window) < SEQ_LEN:
                padding = np.repeat(window[:1], SEQ_LEN - len(window), axis=0)
                window = np.vstack([padding, window])
            X_seqs.append(window)
            y_targets.append(y_group[i])
            target_rows.append(group.iloc[i])
    return (np.asarray(X_seqs, dtype=np.float32),
            np.asarray(y_targets, dtype=np.float32), pd.DataFrame(target_rows).reset_index(drop=True))

X_train_seq, y_train_seq, _ = build_sequences(train_df, {2023, 2024})
val_context = pd.concat([train_df, val_df], ignore_index=True)
X_val_seq, y_val_seq, val_rows = build_sequences(val_context, {2025})
X_final_train_seq, y_final_train_seq, _ = build_sequences(final_train_df, {2023, 2024, 2025})
test_context = pd.concat([final_train_df, test_df], ignore_index=True)
X_test_seq, y_test_seq, test_rows = build_sequences(test_context, {2026})
id_cols = [c for c in ['Dispatch_Date', 'Customer', 'Plant_Variety', 'Size', 'Mode', 'YEAR', 'WEEK']
           if c in test_rows.columns]

print(f"\nSequence shapes -> Train: {X_train_seq.shape}, Validation: {X_val_seq.shape}, "
      f"Final train: {X_final_train_seq.shape}, Test: {X_test_seq.shape}")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Avoid severe CPU oversubscription on small batches in local runs.
torch.set_num_threads(min(4, os.cpu_count() or 1))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train_seq, y_train_seq), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TimeSeriesDataset(X_val_seq, y_val_seq), batch_size=BATCH_SIZE, shuffle=False)
final_train_loader = DataLoader(TimeSeriesDataset(X_final_train_seq, y_final_train_seq), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TimeSeriesDataset(X_test_seq, y_test_seq), batch_size=BATCH_SIZE, shuffle=False)

class StackedLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        current_x   = x[:, -1, :]
        combined    = torch.cat([last_hidden, current_x], dim=1)
        return self.fc(combined).squeeze(-1)

def train_model(model, train_loader, validation_loader, epochs, lr, patience, ckpt_path, model_name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)
    best_loss = float('inf')
    no_improve = 0
    train_losses, val_losses = [], []
    best_epoch = epochs

    print(f"\n  Training {model_name} for up to {epochs} epochs (early stop patience={patience})...")
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        avg_tr = np.mean(batch_losses)

        train_losses.append(avg_tr)
        if validation_loader is not None:
            model.eval()
            val_losses_b = []
            with torch.no_grad():
                for X_b, y_b in validation_loader:
                    X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                    val_losses_b.append(criterion(model(X_b), y_b).item())
            avg_val = float(np.mean(val_losses_b))
            val_losses.append(avg_val)
            scheduler.step(avg_val)
            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:03d}/{epochs}  Train Loss: {avg_tr:.4f}  "
                      f"Val Loss: {avg_val:.4f}  LR: {optimizer.param_groups[0]['lr']:.1e}")
            if avg_val < best_loss:
                best_loss = avg_val
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    [STOP] Early stopping at epoch {epoch}")
                    break
        else:
            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:03d}/{epochs}  Train Loss: {avg_tr:.4f}")

    if validation_loader is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    else:
        torch.save(model.state_dict(), ckpt_path)
    return model, train_losses, val_losses, best_epoch

def evaluate(y_true, y_pred, model_name):
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

all_metrics = []

if os.environ.get('C4_SKIP_LSTM') == '1':
    pred_lstm_df = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions_stacked_lstm.csv'))
    all_metrics.append(evaluate(pred_lstm_df['Actual'].to_numpy(), pred_lstm_df['Predicted'].to_numpy(), 'Stacked Bi-LSTM'))
else:
    # 6A. Train Stacked Bi-LSTM
    print("\n" + "-" * 50)
    print("Model A: Stacked Bidirectional LSTM with Residual Feature Skip")
    torch.manual_seed(42)
    lstm_model = StackedLSTM(X_train_seq.shape[2], HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    lstm_ckpt = os.path.join(CKPT_DIR, 'lstm_best.pt')
    lstm_model, lstm_tr_loss, lstm_val_loss, selected_lstm_epochs = train_model(
        lstm_model, train_loader, val_loader, EPOCHS, LR, PATIENCE, lstm_ckpt,
        "Stacked Bi-LSTM validation fit (2023-2024 / 2025)"
    )
    print(f"  Best validation epoch: {selected_lstm_epochs}")
    
    # Refit with the selected epoch count on all pre-test data (2023-2025).
    torch.manual_seed(42)
    lstm_model = StackedLSTM(X_final_train_seq.shape[2], HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    lstm_model, _, _, _ = train_model(
        lstm_model, final_train_loader, None, selected_lstm_epochs, LR, PATIENCE,
        lstm_ckpt, "Stacked Bi-LSTM final fit (2023-2025)"
    )
    
    lstm_model.eval()
    preds_scaled = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            preds_scaled.extend(lstm_model(X_b.to(DEVICE)).cpu().numpy())
    
    y_pred_lstm = np.maximum(0, y_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten())
    y_true_lstm = test_rows[TARGET].values.astype(np.float32)
    
    all_metrics.append(evaluate(y_true_lstm, y_pred_lstm, "Stacked Bi-LSTM"))
    
    pred_lstm_df = test_rows[id_cols].copy()
    pred_lstm_df['Actual']    = y_true_lstm
    pred_lstm_df['Predicted'] = np.round(y_pred_lstm).astype(int)
    pred_lstm_df['Residual']  = pred_lstm_df['Actual'] - pred_lstm_df['Predicted']
    pred_lstm_df.to_csv(os.path.join(RESULTS_DIR, 'predictions_stacked_lstm.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lstm_tr_loss, label='Train Loss', color='steelblue', lw=1.2)
    ax.plot(lstm_val_loss, label='Val Loss', color='darkorange', lw=1.2)
    ax.set_title('Stacked Bi-LSTM -- Training & Validation Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Scaled Huber Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'lstm_learning_curve.png'), dpi=300)
    plt.close()
    
    # 6B. Temporal Fusion Transformer (TFT)
print("\n" + "-" * 50)
print("Model B: Temporal Fusion Transformer (TFT)")
try:
    import lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import MAE as PFMAE
    from pytorch_forecasting.data import GroupNormalizer

    pl.seed_everything(42, workers=True)

    raw_master = pd.read_csv(
        os.path.join('data', 'c4_time_series', 'processed', 'c4_master_training_data.csv'),
        parse_dates=['Dispatch_Date']
    )
    raw_master = raw_master.sort_values('Dispatch_Date').reset_index(drop=True)
    raw_master['WEEK_sin'] = np.sin(2 * np.pi * raw_master['WEEK'] / 52.0)
    raw_master['WEEK_cos'] = np.cos(2 * np.pi * raw_master['WEEK'] / 52.0)
    raw_master['time_idx'] = raw_master.groupby(['Customer', 'Plant_Variety', 'Size', 'Mode']).cumcount()

    for col in ['Customer', 'Plant_Variety', 'Size', 'Mode']:
        raw_master[col] = raw_master[col].astype(str)

    # Preserve unavailable-weather information and impute from pre-validation
    # data only. Missing test-week weather must not be copied from an earlier week.
    tft_weather_cols = [
        'Avg_Max_Temp_C', 'Weekly_Rainfall_mm', 'Growing_Degree_Days_Weekly',
        'Avg_Max_Temp_C_Lag2W', 'Weekly_Rainfall_mm_Lag2W',
        'Growing_Degree_Days_Weekly_Lag2W'
    ]
    tft_missing_cols = []
    for col in tft_weather_cols:
        flag_col = f'{col}_missing'
        raw_master[flag_col] = raw_master[col].isna().astype(np.float32)
        tft_missing_cols.append(flag_col)
    tft_medians = raw_master.loc[raw_master['YEAR'] < 2025, tft_weather_cols].median()
    if tft_medians.isna().any():
        raise ValueError('TFT has an all-missing weather feature in its initial training data')
    raw_master[tft_weather_cols] = raw_master[tft_weather_cols].fillna(tft_medians)

    # Cast target and real columns to float32 for PyTorch Forecasting.
    float_cols = [
        TARGET, 'Ordered_Quantity', 'Avg_Max_Temp_C', 'Weekly_Rainfall_mm',
        'Growing_Degree_Days_Weekly', 'Avg_Max_Temp_C_Lag2W',
        'Weekly_Rainfall_mm_Lag2W', 'Growing_Degree_Days_Weekly_Lag2W',
        'WEEK_sin', 'WEEK_cos', *tft_missing_cols
    ]
    for fc in float_cols:
        raw_master[fc] = raw_master[fc].astype(np.float32)

    def make_tft_dataset(frame):
        return TimeSeriesDataSet(
            frame,
            time_idx='time_idx',
            target=TARGET,
            group_ids=['Customer', 'Plant_Variety', 'Size', 'Mode'],
            min_encoder_length=1,
            max_encoder_length=SEQ_LEN,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=['Customer', 'Plant_Variety', 'Size', 'Mode'],
            time_varying_known_reals=[
                'time_idx', 'WEEK_sin', 'WEEK_cos', 'Ordered_Quantity',
                'Avg_Max_Temp_C', 'Weekly_Rainfall_mm', 'Growing_Degree_Days_Weekly',
                'Avg_Max_Temp_C_missing', 'Weekly_Rainfall_mm_missing',
                'Growing_Degree_Days_Weekly_missing'
            ],
            time_varying_unknown_reals=[
                TARGET, 'Avg_Max_Temp_C_Lag2W', 'Weekly_Rainfall_mm_Lag2W',
                'Growing_Degree_Days_Weekly_Lag2W', 'Avg_Max_Temp_C_Lag2W_missing',
                'Weekly_Rainfall_mm_Lag2W_missing',
                'Growing_Degree_Days_Weekly_Lag2W_missing'
            ],
            target_normalizer=GroupNormalizer(groups=GROUP_COLS),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

    def only_prediction_year(dataset, frame, year):
        key_cols = GROUP_COLS + ['time_idx']
        year_by_key = {
            tuple(row[col] for col in key_cols): row.YEAR
            for _, row in frame[key_cols + ['YEAR']].iterrows()
        }
        return dataset.filter(lambda idx: np.array([
            year_by_key.get(tuple(getattr(row, col) for col in GROUP_COLS) +
                            (getattr(row, 'time_idx_first_prediction'),)) == year
            and getattr(row, 'time_idx_first_prediction') - getattr(row, 'time_idx_first') == SEQ_LEN
            for row in idx.itertuples(index=False)
        ]))

    def tft_model(dataset):
        return TemporalFusionTransformer.from_dataset(
            dataset, learning_rate=LR, hidden_size=64, attention_head_size=4,
            dropout=DROPOUT, hidden_continuous_size=32, output_size=1, loss=PFMAE()
        )

    # Select epoch count only on 2025. The 2026 test data is not passed to fit().
    initial_tft_train = raw_master[raw_master['YEAR'] < 2025].copy()
    initial_tft_ds = make_tft_dataset(initial_tft_train)
    validation_context = raw_master[raw_master['YEAR'] <= 2025].copy()
    tft_val_ds = only_prediction_year(
        TimeSeriesDataSet.from_dataset(initial_tft_ds, validation_context,
                                       predict=False, stop_randomization=True),
        validation_context, 2025
    )
    tft_initial_loader = initial_tft_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    tft_val_loader = tft_val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    from lightning.pytorch.callbacks import ModelCheckpoint
    selection_checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
    tft = tft_model(initial_tft_ds)
    print(f"  TFT Parameters: {tft.size() / 1e3:.1f}k")
    selection_trainer = pl.Trainer(
        max_epochs=25, accelerator='auto', gradient_clip_val=0.1,
        limit_train_batches=50, enable_progress_bar=True, enable_model_summary=False,
        callbacks=[selection_checkpoint], logger=False
    )
    selection_trainer.fit(tft, tft_initial_loader, tft_val_loader)
    if not selection_checkpoint.best_model_path:
        raise RuntimeError("TFT validation did not produce a best checkpoint")
    best_state = torch.load(selection_checkpoint.best_model_path, map_location=DEVICE, weights_only=False)
    selected_tft_epochs = int(best_state['epoch']) + 1
    print(f"  Best 2025 validation epoch: {selected_tft_epochs}")

    # Refit from scratch on all pre-test rows using the selected epoch count.
    final_tft_train = raw_master[raw_master['YEAR'] < 2026].copy()
    final_tft_ds = make_tft_dataset(final_tft_train)
    final_tft_loader = final_tft_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    tft = tft_model(final_tft_ds)
    final_trainer = pl.Trainer(
        max_epochs=selected_tft_epochs, accelerator='auto', gradient_clip_val=0.1,
        limit_train_batches=50, enable_progress_bar=True, enable_model_summary=False,
        logger=False, enable_checkpointing=False
    )
    final_trainer.fit(tft, final_tft_loader)
    torch.save(tft.state_dict(), os.path.join(CKPT_DIR, 'tft_final.pt'))

    # Score only 2026 prediction targets. Group/time keys restore row alignment.
    test_context_tft = raw_master.copy()
    tft_test_ds = only_prediction_year(
        TimeSeriesDataSet.from_dataset(final_tft_ds, test_context_tft,
                                       predict=False, stop_randomization=True),
        test_context_tft, 2026
    )
    tft_test_loader = tft_test_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    predictions = tft.predict(tft_test_loader, return_y=True, return_index=True,
                              trainer_kwargs={"accelerator": "auto"})
    pred_index = predictions.index.reset_index(drop=True)
    row_by_key = {
        tuple(row[col] for col in GROUP_COLS) + (int(row['time_idx']),): int(i)
        for i, row in raw_master[GROUP_COLS + ['time_idx']].iterrows()
    }
    pred_index['Target_Row'] = [
        row_by_key[tuple(getattr(row, col) for col in GROUP_COLS) + (int(row.time_idx),)]
        for row in pred_index.itertuples(index=False)
    ]
    tft_pred_values = predictions.output.detach().cpu().numpy().reshape(-1)
    pred_index['Predicted'] = np.maximum(0, tft_pred_values)
    target_metadata = raw_master.reset_index(names='Target_Row')[
        ['Target_Row', 'Dispatch_Date', 'YEAR', 'WEEK', TARGET]
    ]
    tft_pred_df = pred_index.merge(
        target_metadata, on='Target_Row', how='left'
    ).sort_values('Dispatch_Date')
    y_true_tft = tft_pred_df[TARGET].to_numpy(dtype=float)
    y_pred_tft = tft_pred_df['Predicted'].to_numpy(dtype=float)

    all_metrics.append(evaluate(y_true_tft, y_pred_tft, "TFT"))
    tft_pred_df = tft_pred_df[id_cols + ['Target_Row']].copy()
    tft_pred_df['Actual'] = y_true_tft
    tft_pred_df['Predicted'] = np.round(y_pred_tft).astype(int)
    tft_pred_df['Residual'] = tft_pred_df['Actual'] - tft_pred_df['Predicted']
    tft_pred_df.drop(columns=['Target_Row']).to_csv(
        os.path.join(RESULTS_DIR, 'predictions_tft.csv'), index=False
    )
    print("  TFT predictions saved.")
except Exception as e:
    print(f"  [WARN]  TFT skipped/failed: {e}")

# 7. Actual vs Predicted Plot (LSTM)
if 'Plant_Variety' in pred_lstm_df.columns:
    top_vars = pred_lstm_df['Plant_Variety'].value_counts().head(5).index.tolist()
    fig, axes = plt.subplots(len(top_vars), 1, figsize=(14, 3.5 * len(top_vars)))
    if len(top_vars) == 1:
        axes = [axes]
    for ax, variety in zip(axes, top_vars):
        sub = pred_lstm_df[pred_lstm_df['Plant_Variety'] == variety].sort_values('Dispatch_Date')
        ax.plot(sub['Dispatch_Date'], sub['Actual'], label='Actual', color='steelblue', lw=1.2)
        ax.plot(sub['Dispatch_Date'], sub['Predicted'], label='LSTM', color='crimson', lw=1.2, ls='--')
        ax.set_title(f'{variety}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.tick_params(axis='x', rotation=30, labelsize=7)
    fig.suptitle('Stacked Bi-LSTM: Actual vs Predicted -- 2026 Test Set', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'lstm_actual_vs_predicted.png'), dpi=300)
    plt.close()

dl_metrics_df = pd.DataFrame(all_metrics)
dl_metrics_df.to_csv(os.path.join(RESULTS_DIR, 'dl_model_metrics.csv'), index=False)
print("\n" + "=" * 65)
print("  DEEP LEARNING METRICS SUMMARY")
print("=" * 65)
print(dl_metrics_df.to_string(index=False))
print(f"\n[OK] Script 6 Complete!")

"""
Script 7: Model Evaluation & Comparison Report
===============================================
Aggregates results from all baseline and deep learning models into a single
publication-ready evaluation report, including:

  1. Consolidated metrics table (MAE, RMSE, MAPE, R2)
  2. Grouped bar chart comparison across all models
  3. Actual vs. Predicted overlays (all models, top variety)
  4. Residual analysis plots
  5. Supply-demand gap forecast accuracy
  6. Per-customer performance breakdown

Run from project root:
    python scripts/c4_time_series/c4_07_evaluation_report.py
"""
import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

print("=" * 65)
print("  Script 7: Final Evaluation & Comparison Report")
print("=" * 65)

# -- 0. Paths ------------------------------------------------------------------
RESULTS_DIR = os.path.join('data', 'c4_time_series', 'results')
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
REPORT_DIR  = os.path.join(RESULTS_DIR, 'final_report')
os.makedirs(REPORT_DIR, exist_ok=True)

# -- 1. Load All Prediction CSVs -----------------------------------------------
prediction_files = {
    'XGBoost':         os.path.join(RESULTS_DIR, 'predictions_xgboost.csv'),
    'Random Forest':   os.path.join(RESULTS_DIR, 'predictions_random_forest.csv'),
    'Stacked Bi-LSTM': os.path.join(RESULTS_DIR, 'predictions_stacked_lstm.csv'),
    'TFT':             os.path.join(RESULTS_DIR, 'predictions_tft.csv'),
    'Naive: Ordered Quantity': os.path.join(RESULTS_DIR, 'predictions_naive_order.csv'),
    'Naive: Historical Dispatch/Order Ratio': os.path.join(RESULTS_DIR, 'predictions_naive_ratio.csv'),
    'Naive: Last Training Dispatch': os.path.join(RESULTS_DIR, 'predictions_naive_last.csv'),
}

# Also look for SARIMAX weekly aggregate
sarimax_path = os.path.join(RESULTS_DIR, 'predictions_sarimax.csv')


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Compute regression metrics and return as dict."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = (np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
            if nonzero.sum() > 0 else np.nan)
    r2   = r2_score(y_true, y_pred)
    return {'Model': model_name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
            'MAPE_%': round(mape, 2), 'R2': round(r2, 4)}


# -- 2. Load row-level prediction files ----------------------------------------
all_metrics = []
pred_frames = {}

for model_name, pred_path in prediction_files.items():
    if os.path.exists(pred_path):
        pdf = pd.read_csv(pred_path)
        if 'Actual' in pdf.columns and 'Predicted' in pdf.columns:
            y_true = pdf['Actual'].values.astype(float)
            y_pred = pdf['Predicted'].values.astype(float)
            all_metrics.append(pd.DataFrame([compute_metrics(y_true, y_pred, model_name)]))
            pred_frames[model_name] = pdf
            print(f"  Loaded predictions: {model_name} ({len(pdf):,} rows)")
    else:
        print(f"  [WARN]  No predictions found for {model_name} at {pred_path}")

# SARIMAX operates on weekly aggregates, so it is reported in the weekly
# comparison below and is deliberately excluded from this row-level ranking.
weekly_frames = {}
if os.path.exists(sarimax_path):
    sdf = pd.read_csv(sarimax_path)
    if 'Actual_Weekly_Total' in sdf.columns:
        weekly_frames['SARIMAX (weekly agg.)'] = sdf[
            ['YEAR', 'WEEK', 'Actual_Weekly_Total', 'Predicted_Weekly_Total']
        ].rename(columns={
            'Actual_Weekly_Total': 'Actual', 'Predicted_Weekly_Total': 'Predicted'
        })
        print(f"  Loaded predictions: SARIMAX ({len(sdf):,} weekly rows)")

if not all_metrics:
    raise RuntimeError(
        "No metrics or predictions found. Run Scripts 5 & 6 first, then come back to Script 7."
    )

# -- 3. Build Consolidated Metrics Table --------------------------------------
metrics_df = pd.concat(all_metrics, ignore_index=True)
# De-duplicate: keep best RMSE per model
metrics_df = (metrics_df.sort_values('RMSE')
                         .groupby('Model', as_index=False)
                         .first()
                         .sort_values('RMSE')
                         .reset_index(drop=True))

metrics_df['Rank'] = metrics_df['RMSE'].rank().astype(int)
print("\n" + "=" * 65)
print("  FINAL MODEL COMPARISON TABLE")
print("=" * 65)
print(metrics_df[['Rank', 'Model', 'MAE', 'RMSE', 'MAPE_%', 'R2']].to_string(index=False))

metrics_df.to_csv(os.path.join(REPORT_DIR, 'final_model_comparison.csv'), index=False)
print(f"\n  Saved -> {os.path.join(REPORT_DIR, 'final_model_comparison.csv')}")

# Compare all models at a common weekly-total level, including SARIMAX.
for model_name, pdf in pred_frames.items():
    if {'YEAR', 'WEEK', 'Actual', 'Predicted'}.issubset(pdf.columns):
        weekly_frames[model_name] = pdf.groupby(['YEAR', 'WEEK'], as_index=False).agg(
            Actual=('Actual', 'sum'), Predicted=('Predicted', 'sum')
        )

weekly_metrics = []
for model_name, frame in weekly_frames.items():
    if {'Actual', 'Predicted'}.issubset(frame.columns):
        weekly_metrics.append(compute_metrics(
            frame['Actual'].to_numpy(dtype=float),
            frame['Predicted'].to_numpy(dtype=float), model_name
        ) | {'n_weeks': len(frame)})
if weekly_metrics:
    weekly_metrics_df = (pd.DataFrame(weekly_metrics).sort_values('RMSE')
                         .reset_index(drop=True))
    weekly_metrics_df.insert(0, 'Rank', np.arange(1, len(weekly_metrics_df) + 1))
    weekly_metrics_df.to_csv(os.path.join(REPORT_DIR, 'weekly_model_comparison.csv'), index=False)
    print("\nWEEKLY TOTAL COMPARISON (all models, same scale)")
    print(weekly_metrics_df.to_string(index=False))
    print(f"  Saved -> {os.path.join(REPORT_DIR, 'weekly_model_comparison.csv')}")

best_model_name = metrics_df.iloc[0]['Model']
print(f"\n  ? Best Model: {best_model_name} (RMSE = {metrics_df.iloc[0]['RMSE']:,.2f})")

# -- 4. Plot 1: Grouped Bar Chart -- All Model Metrics -------------------------
print("\nGenerating comparison bar charts...")
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
palette = sns.color_palette('Set2', len(metrics_df))

for ax, metric, label, fmt in zip(
    axes,
    ['MAE', 'RMSE', 'MAPE_%', 'R2'],
    ['MAE (lower = better)', 'RMSE (lower = better)',
     'MAPE % (lower = better)', 'R2 (higher = better)'],
    ['{:.0f}', '{:.0f}', '{:.1f}%', '{:.3f}']
):
    bars = ax.barh(metrics_df['Model'], metrics_df[metric], color=palette, edgecolor='white', lw=0.5)
    ax.set_xlabel(label, fontsize=9)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                fmt.format(val), va='center', fontsize=7.5)

fig.suptitle('C4 Forecasting -- Model Comparison Dashboard (2026 Test Set)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'model_comparison_bars.png'), dpi=300)
plt.close()
print(f"  Saved -> model_comparison_bars.png")

# -- 5. Plot 2: Actual vs Predicted Overlay (all available models, top variety) -
print("\nGenerating Actual vs. Predicted comparison...")
# Find common variety across all prediction frames
if pred_frames:
    variety_counts = {}
    for model, pdf in pred_frames.items():
        if 'Plant_Variety' in pdf.columns:
            for v in pdf['Plant_Variety'].unique():
                variety_counts[v] = variety_counts.get(v, 0) + 1
    # Pick varieties present in ALL loaded prediction files
    all_count = max(variety_counts.values()) if variety_counts else 0
    top_variety_candidates = [v for v, c in variety_counts.items() if c == all_count]
    if not top_variety_candidates:
        top_variety_candidates = list(variety_counts.keys())
    top_variety = top_variety_candidates[0]   # pick first match

    n_models = len(pred_frames)
    if n_models > 0:
        fig, ax = plt.subplots(figsize=(16, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, n_models + 1))
        first = True
        for idx, (model, pdf) in enumerate(pred_frames.items()):
            if 'Plant_Variety' not in pdf.columns:
                sub = pdf
            else:
                sub = pdf[pdf['Plant_Variety'] == top_variety]
            if len(sub) == 0:
                continue
            if 'Dispatch_Date' in sub.columns:
                sub = sub.sort_values('Dispatch_Date')
                x_axis = sub['Dispatch_Date']
            else:
                x_axis = np.arange(len(sub))

            if first:
                ax.plot(x_axis, sub['Actual'], label='Actual',
                        color='black', lw=2, zorder=5)
                first = False
            ax.plot(x_axis, sub['Predicted'], label=model,
                    color=colors[idx], lw=1.3, linestyle='--', alpha=0.85)

        ax.set_title(f'Actual vs. Predicted -- {top_variety} (2026 Test Set)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Date / Index', fontsize=10)
        ax.set_ylabel('Dispatched Quantity', fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'all_models_actual_vs_predicted.png'), dpi=300)
        plt.close()
        print(f"  Saved -> all_models_actual_vs_predicted.png  (variety: {top_variety})")

# -- 6. Plot 3: Residual Analysis (best ML model) ------------------------------
print("\nGenerating residual analysis plots...")
best_pred_key = next((k for k in [best_model_name, 'Stacked Bi-LSTM', 'XGBoost', 'Random Forest']
                       if k in pred_frames), None)
if best_pred_key:
    pdf_best = pred_frames[best_pred_key]
    residuals = pdf_best['Actual'] - pdf_best['Predicted']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Residuals over index / date
    ax = axes[0]
    ax.plot(residuals.values, color='steelblue', alpha=0.6, lw=0.8)
    ax.axhline(0, color='red', lw=1, linestyle='--')
    ax.fill_between(range(len(residuals)), residuals.values, 0,
                    where=residuals.values > 0, alpha=0.15, color='green', label='Over-supply')
    ax.fill_between(range(len(residuals)), residuals.values, 0,
                    where=residuals.values < 0, alpha=0.15, color='red', label='Under-supply')
    ax.set_title(f'Residuals over Time\n({best_pred_key})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Sample Index'); ax.set_ylabel('Residual (Actual ? Predicted)')
    ax.legend(fontsize=8)

    # Residual distribution
    ax = axes[1]
    ax.hist(residuals, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(0, color='red', lw=1.5, linestyle='--', label='Zero')
    ax.axvline(residuals.mean(), color='orange', lw=1.5, linestyle=':', label=f'Mean={residuals.mean():.1f}')
    ax.set_title('Residual Distribution', fontsize=10, fontweight='bold')
    ax.set_xlabel('Residual'); ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    # Scatter: Actual vs Predicted
    ax = axes[2]
    ax.scatter(pdf_best['Actual'], pdf_best['Predicted'], alpha=0.3, s=8, color='steelblue')
    lim_max = max(pdf_best['Actual'].max(), pdf_best['Predicted'].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], 'r--', lw=1.5, label='Perfect Forecast')
    ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
    ax.set_title('Actual vs. Predicted (scatter)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8)

    fig.suptitle(f'Residual Analysis -- {best_pred_key}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'residual_analysis.png'), dpi=300)
    plt.close()
    print(f"  Saved -> residual_analysis.png")

    # Residual by YEAR/WEEK seasonality
    if 'WEEK' in pdf_best.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        weekly_resid = pdf_best.groupby('WEEK')['Residual'].mean() if 'Residual' in pdf_best.columns \
            else pdf_best.assign(Residual=residuals).groupby('WEEK')['Residual'].mean()
        ax.bar(weekly_resid.index, weekly_resid.values,
               color=['#2196F3' if v >= 0 else '#F44336' for v in weekly_resid.values], alpha=0.8)
        ax.axhline(0, color='black', lw=1)
        ax.set_title(f'Average Residual by Week Number -- {best_pred_key}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Week of Year'); ax.set_ylabel('Mean Residual')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'residual_by_week.png'), dpi=300)
        plt.close()
        print(f"  Saved -> residual_by_week.png")

# -- 7. Per-Customer Performance (best model) ---------------------------------
print("\nGenerating per-customer performance breakdown...")
if best_pred_key and 'Customer' in pred_frames.get(best_pred_key, pd.DataFrame()).columns:
    pdf_best = pred_frames[best_pred_key]
    customer_metrics = []
    for cust, grp in pdf_best.groupby('Customer'):
        y_t = grp['Actual'].values.astype(float)
        y_p = grp['Predicted'].values.astype(float)
        if len(y_t) < 5:
            continue
        m = compute_metrics(y_t, y_p, cust)
        m['n_transactions'] = len(grp)
        customer_metrics.append(m)
    if customer_metrics:
        cust_df = pd.DataFrame(customer_metrics).sort_values('RMSE')
        cust_df.to_csv(os.path.join(REPORT_DIR, 'per_customer_metrics.csv'), index=False)
        print(f"\n  Per-Customer Metrics ({best_pred_key}):")
        print(cust_df[['Model', 'MAE', 'RMSE', 'MAPE_%', 'R2', 'n_transactions']].to_string(index=False))
        print(f"  Saved -> per_customer_metrics.csv")

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(cust_df['Model'], cust_df['RMSE'],
                      color=sns.color_palette('coolwarm', len(cust_df)), edgecolor='white')
        ax.set_title(f'RMSE per Customer -- {best_pred_key}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Customer'); ax.set_ylabel('RMSE')
        ax.tick_params(axis='x', rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'per_customer_rmse.png'), dpi=300)
        plt.close()
        print(f"  Saved -> per_customer_rmse.png")

# -- 8. Summary Text Report ----------------------------------------------------
report_lines = [
    "=" * 65,
    "  FLORASENSE C4 -- FORECASTING EVALUATION REPORT",
    f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    "=" * 65,
    "",
    "MODELS EVALUATED:",
]
for _, row in metrics_df.iterrows():
    report_lines.append(
        f"  {row['Rank']}. {row['Model']:<30} "
        f"MAE={row['MAE']:>8,.1f}  RMSE={row['RMSE']:>8,.1f}  "
        f"MAPE={row['MAPE_%']:>5.1f}%  R2={row['R2']:.4f}"
    )

report_lines += [
    "",
    f"BEST MODEL : {best_model_name}",
    f"RMSE       : {metrics_df.iloc[0]['RMSE']:,.2f}",
    f"MAE        : {metrics_df.iloc[0]['MAE']:,.2f}",
    f"MAPE       : {metrics_df.iloc[0]['MAPE_%']:.2f}%",
    f"R2         : {metrics_df.iloc[0]['R2']:.4f}",
    "",
    "FILES GENERATED:",
    f"  {os.path.join(REPORT_DIR, 'final_model_comparison.csv')}",
    f"  {os.path.join(REPORT_DIR, 'weekly_model_comparison.csv')}",
    f"  {os.path.join(REPORT_DIR, 'model_comparison_bars.png')}",
    f"  {os.path.join(REPORT_DIR, 'all_models_actual_vs_predicted.png')}",
    f"  {os.path.join(REPORT_DIR, 'residual_analysis.png')}",
    f"  {os.path.join(REPORT_DIR, 'per_customer_metrics.csv')}",
    "=" * 65,
]
report_text = '\n'.join(report_lines)
print('\n' + report_text)
with open(os.path.join(REPORT_DIR, 'evaluation_summary.txt'), 'w') as f:
    f.write(report_text)

print(f"\n[OK] Script 7 Complete! Full report saved to: {REPORT_DIR}")

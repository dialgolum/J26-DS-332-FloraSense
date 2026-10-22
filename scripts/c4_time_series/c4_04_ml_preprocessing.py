import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("Running Script 4: Machine Learning Preprocessing & Chronological Split...")

# 1. Load Master Dataset
master_path = os.path.join('data', 'c4_time_series', 'processed', 'c4_master_training_data.csv')
if not os.path.exists(master_path):
    raise FileNotFoundError(f"Missing {master_path}. Run c4_merge_master.py first.")

df = pd.read_csv(master_path)
df['Dispatch_Date'] = pd.to_datetime(df['Dispatch_Date'])
df['WEEK_sin'] = np.sin(2 * np.pi * df['WEEK'] / 52.0)
df['WEEK_cos'] = np.cos(2 * np.pi * df['WEEK'] / 52.0)

# 2. Define Target and Feature Groups
target_col = 'Dispatched_Quantity'
cat_cols = ['Customer', 'Plant_Variety', 'Size', 'Mode']
num_cols = [
    'Ordered_Quantity',
    'Avg_Max_Temp_C', 'Avg_Min_Temp_C', 'Weekly_Rainfall_mm',
    'Temp_Diff_C', 'Growing_Degree_Days_Weekly',
    'Avg_Max_Temp_C_Lag2W', 'Weekly_Rainfall_mm_Lag2W', 'Growing_Degree_Days_Weekly_Lag2W',
    'Avg_Max_Temp_C_Lag4W', 'Weekly_Rainfall_mm_Lag4W', 'Growing_Degree_Days_Weekly_Lag4W'
]

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Target: '{target_col}'")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features  ({len(num_cols)}): {num_cols}")

# Keep missing-weather indicators before imputation. This distinguishes unknown
# future weather from an observed value that happens to equal the median.
weather_feature_cols = [c for c in num_cols if c != 'Ordered_Quantity']
missing_flag_cols = []
for col in weather_feature_cols:
    flag_col = f'{col}_missing'
    df[flag_col] = df[col].isna().astype(np.float32)
    missing_flag_cols.append(flag_col)

# 4. One-Hot Encode Categorical Features AND Preserve Original Metadata Columns for Reporting
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
for c in cat_cols:
    df_encoded[c] = df[c]

# Convert boolean one-hot columns to float32 for PyTorch/XGBoost compatibility
bool_cols = df_encoded.select_dtypes(include=['bool']).columns.tolist()
df_encoded[bool_cols] = df_encoded[bool_cols].astype(np.float32)

print(f"\nAfter one-hot encoding (+ metadata cols): {df_encoded.shape[1]} columns")

# 5. Chronological Train / Validation / Test Split
#    2023-2024: initial training, 2025: validation, 2026: final test.
train_df = df_encoded[df_encoded['YEAR'] < 2025].copy()
val_df   = df_encoded[df_encoded['YEAR'] == 2025].copy()
final_train_df = df_encoded[df_encoded['YEAR'] < 2026].copy()
test_df  = df_encoded[df_encoded['YEAR'] == 2026].copy()

print(f"\nInitial training set (2023-2024): {train_df.shape[0]:,} rows x {train_df.shape[1]} cols")
print(f"Validation set (2025):            {val_df.shape[0]:,} rows x {val_df.shape[1]} cols")
print(f"Final training set (2023-2025):   {final_train_df.shape[0]:,} rows x {final_train_df.shape[1]} cols")
print(f"Final test set (2026):            {test_df.shape[0]:,} rows x {test_df.shape[1]} cols")

# Impute from initial training data only, then apply the same medians to every
# split. This handles unavailable weather and the earliest unobservable lags.
imputation_medians = train_df[num_cols].median()
if imputation_medians.isna().any():
    missing = imputation_medians[imputation_medians.isna()].index.tolist()
    raise ValueError(f'Cannot impute all-missing training features: {missing}')
for split_df in [train_df, val_df, final_train_df, test_df]:
    split_df[num_cols] = split_df[num_cols].fillna(imputation_medians)

# 6. Feature Scaling -- fit on initial training data only. The same transforms
#    are used for validation, final training, and the untouched test set.
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
val_df[num_cols] = scaler.transform(val_df[num_cols])
final_train_df[num_cols] = scaler.transform(final_train_df[num_cols])
test_df[num_cols]  = scaler.transform(test_df[num_cols])

# Fit the target scaler on initial training labels only.
target_scaler = StandardScaler()
target_scaler.fit(train_df[[target_col]])

# 7. Save Processed Splits, Scalers, and Feature Metadata
processed_dir = os.path.join('data', 'c4_time_series', 'ml_ready')
os.makedirs(processed_dir, exist_ok=True)

train_path  = os.path.join(processed_dir, 'train_data.csv')
val_path    = os.path.join(processed_dir, 'validation_data.csv')
final_train_path = os.path.join(processed_dir, 'final_train_data.csv')
test_path   = os.path.join(processed_dir, 'test_data.csv')
scaler_path = os.path.join(processed_dir, 'scaler.pkl')
y_scaler_path = os.path.join(processed_dir, 'target_scaler.pkl')

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
final_train_df.to_csv(final_train_path, index=False)
test_df.to_csv(test_path, index=False)
joblib.dump(scaler, scaler_path)
joblib.dump(target_scaler, y_scaler_path)

exclude_cols = ['Dispatch_Date', target_col, 'Supply_Demand_Gap', 'YEAR', 'WEEK'] + cat_cols
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

feature_meta = {
    'target_col': target_col,
    'num_cols': num_cols,
    'week_cycle_cols': ['WEEK_sin', 'WEEK_cos'],
    'missing_flag_cols': missing_flag_cols,
    'cat_cols': cat_cols,
    'feature_cols': feature_cols,
    'n_features': len(feature_cols)
}
with open(os.path.join(processed_dir, 'feature_meta.json'), 'w') as f:
    json.dump(feature_meta, f, indent=2)

print(f"\n[OK] Preprocessing Complete! Total features for modelling: {len(feature_cols)}")

import os
import pandas as pd

print("Running Script 2: Merging Cleaned Sales with Weekly Weather & Lag Features...")

# 1. Define Input & Output Paths
sales_path = os.path.join('data', 'c4_time_series', 'interim', 'cleaned_sales_2023_2026.csv')
weather_path = os.path.join('data', 'c4_time_series', 'interim', 'cleaned_weather_weekly.csv')
master_out = os.path.join('data', 'c4_time_series', 'processed', 'c4_master_training_data.csv')

if not os.path.exists(sales_path):
    raise FileNotFoundError(f"Missing {sales_path}. Run c4_clean_sales.py first.")
if not os.path.exists(weather_path):
    raise FileNotFoundError(f"Missing {weather_path}. Please place 'cleaned_weather_weekly.csv' in data/c4_time_series/interim/.")

sales_df = pd.read_csv(sales_path)
weather_df = pd.read_csv(weather_path)

# 2. Map Dispatch_Date to Nursery Calendar (YEAR, WEEK) anchored at Monday 2023-01-02
sales_df['Dispatch_Date_dt'] = pd.to_datetime(sales_df['Dispatch_Date'])
days_since_anchor = (sales_df['Dispatch_Date_dt'] - pd.Timestamp('2023-01-02')).dt.days
week_seq = days_since_anchor // 7
sales_df['YEAR'] = 2023 + (week_seq // 52)
sales_df['WEEK'] = 1 + (week_seq % 52)
sales_df = sales_df.drop(columns=['Dispatch_Date_dt'])

# 3. Extend weekly weather table through Weeks 40-43 of 2026 so October 2026 shipments
# receive exact 2-week and 4-week lagged September weather (Weeks 36-39) with 0 NaNs
w_df = weather_df.sort_values(['YEAR', 'WEEK']).reset_index(drop=True).copy()
last_w = w_df.iloc[-1]
ext_rows = []
for add_wk in [40, 41, 42, 43]:
    if not ((w_df['YEAR'] == 2026) & (w_df['WEEK'] == add_wk)).any():
        r_copy = last_w.copy()
        r_copy['YEAR'] = 2026
        r_copy['WEEK'] = add_wk
        ext_rows.append(r_copy)
if ext_rows:
    w_df = pd.concat([w_df, pd.DataFrame(ext_rows)], ignore_index=True)

# 4. Engineer 2-Week (14-day) and 4-Week (28-day) Agronomic Weather Lags
for lag in [2, 4]:
    w_df[f'Avg_Max_Temp_C_Lag{lag}W'] = w_df['Avg_Max_Temp_C'].shift(lag).bfill()
    w_df[f'Weekly_Rainfall_mm_Lag{lag}W'] = w_df['Weekly_Rainfall_mm'].shift(lag).bfill()
    w_df[f'Growing_Degree_Days_Weekly_Lag{lag}W'] = w_df['Growing_Degree_Days_Weekly'].shift(lag).bfill()

# 5. Merge Sales with Weather Lags and Compute Supply_Demand_Gap
master_df = pd.merge(sales_df, w_df, on=['YEAR', 'WEEK'], how='left')
master_df['Supply_Demand_Gap'] = master_df['Dispatched_Quantity'] - master_df['Ordered_Quantity']

master_cols = [
    'Dispatch_Date', 'YEAR', 'WEEK', 'Customer', 'Plant_Variety', 'Size', 'Mode',
    'Ordered_Quantity', 'Dispatched_Quantity', 'Supply_Demand_Gap',
    'Avg_Max_Temp_C', 'Avg_Min_Temp_C', 'Weekly_Rainfall_mm', 'Temp_Diff_C', 'Growing_Degree_Days_Weekly',
    'Avg_Max_Temp_C_Lag2W', 'Weekly_Rainfall_mm_Lag2W', 'Growing_Degree_Days_Weekly_Lag2W',
    'Avg_Max_Temp_C_Lag4W', 'Weekly_Rainfall_mm_Lag4W', 'Growing_Degree_Days_Weekly_Lag4W'
]
master_df = master_df[master_cols]

# 6. Save Master Training Dataset
os.makedirs(os.path.dirname(master_out), exist_ok=True)
master_df.to_csv(master_out, index=False)

print(f"Master Merge Complete! Final Shape: {master_df.shape[0]} rows, {master_df.shape[1]} columns.")
print(f"Missing (NaN) values count: {master_df.isnull().sum().sum()}")
print(f"Saved to: {master_out}")
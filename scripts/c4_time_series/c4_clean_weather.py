import pandas as pd
import numpy as np
import os

print("Starting Real Mike Flora Weather Cleaning Pipeline...")

excel_path = 'data/c4_time_series/raw/METEOROLOGICAL DATA.xlsx'
if not os.path.exists(excel_path):
    print(f"Error: Place '{excel_path}' in data/c4_time_series/raw/ first!")
    exit()

# 1. Helper function to unpivot (melt) Monday-Saturday sheets into chronological rows
def extract_sheet(sheet_name, value_col):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.iloc[0]]
    df = df[1:].reset_index(drop=True)
    df['YEAR'] = df['YEAR'].ffill().astype(int)
    df['WEEK'] = df['WEEK'].astype(int)

    days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY']
    day_offset = {'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5}

    melted = df.melt(id_vars=['YEAR', 'WEEK'], value_vars=days, var_name='Day_Name', value_name=value_col)
    melted['Day_Offset'] = melted['Day_Name'].map(day_offset)
    melted = melted.sort_values(['YEAR', 'WEEK', 'Day_Offset']).reset_index(drop=True)
    return melted

# 2. Extract and combine Rainfall, Min Temp, and Max Temp
rain_df = extract_sheet('RAINFALL', 'Rainfall_mm')
min_df = extract_sheet('MINIMUM TEM', 'Min_Temp_C')
max_df = extract_sheet('MAXIMUM TEM', 'Max_Temp_C')

raw = rain_df[['YEAR', 'WEEK', 'Day_Name', 'Day_Offset', 'Rainfall_mm']].copy()
raw['Min_Temp_C'] = min_df['Min_Temp_C']
raw['Max_Temp_C'] = max_df['Max_Temp_C']

# Filter out future unrecorded weeks in late 2026 (keep up to 2026 Week 39)
raw = raw[~((raw['YEAR'] == 2026) & (raw['WEEK'] > 39))].copy()

# 3. Clean text strings ('P/d' for Poya Day, 'H/d' for Holiday) -> convert to NaN
for col in ['Rainfall_mm', 'Min_Temp_C', 'Max_Temp_C']:
    raw[col] = pd.to_numeric(raw[col], errors='coerce')

# Fix clerical typo on 2026 Week 32 Tuesday (34.0 entered in Min_Temp_C while Max_Temp_C was NaN)
typo_mask = (raw['Min_Temp_C'] >= 33.0) & (raw['Max_Temp_C'].isna())
raw.loc[typo_mask, 'Max_Temp_C'] = raw.loc[typo_mask, 'Min_Temp_C']
raw.loc[typo_mask, 'Min_Temp_C'] = np.nan

# 4. Save Cleaned WEEKLY Dataset (in case sales data tomorrow is weekly)
weekly_df = raw.groupby(['YEAR', 'WEEK'], as_index=False).agg(
    Avg_Max_Temp_C=('Max_Temp_C', 'mean'),
    Avg_Min_Temp_C=('Min_Temp_C', 'mean'),
    Weekly_Rainfall_mm=('Rainfall_mm', 'sum')
)
weekly_df['Avg_Max_Temp_C'] = weekly_df['Avg_Max_Temp_C'].interpolate(method='linear')
weekly_df['Avg_Min_Temp_C'] = weekly_df['Avg_Min_Temp_C'].interpolate(method='linear')
weekly_df['Temp_Diff_C'] = weekly_df['Avg_Max_Temp_C'] - weekly_df['Avg_Min_Temp_C']
weekly_df['Growing_Degree_Days_Weekly'] = (
    np.maximum(0, ((weekly_df['Avg_Max_Temp_C'] + weekly_df['Avg_Min_Temp_C']) / 2.0) - 15.0) * 7.0
)

os.makedirs('data/c4_time_series/interim', exist_ok=True)
weekly_df.to_csv('data/c4_time_series/interim/cleaned_weather_weekly.csv', index=False)

# 5. Build Continuous DAILY Dataset (inserting missing Sundays & interpolating)
raw['Week_Seq'] = (raw['YEAR'] - 2023) * 52 + (raw['WEEK'] - 1)
base_monday = pd.Timestamp('2023-01-02')
raw['Date'] = base_monday + pd.to_timedelta(raw['Week_Seq'] * 7 + raw['Day_Offset'], unit='D')

daily_df = raw.set_index('Date')[['YEAR', 'WEEK', 'Max_Temp_C', 'Min_Temp_C', 'Rainfall_mm']].copy()
full_dates = pd.date_range(start=daily_df.index.min(), end=daily_df.index.max(), freq='D')
daily_df = daily_df.reindex(full_dates)
daily_df.index.name = 'Date'
daily_df = daily_df.reset_index()

# Forward-fill YEAR/WEEK, linearly interpolate temperatures, & zero-fill rainfall
daily_df['YEAR'] = daily_df['YEAR'].ffill().astype(int)
daily_df['WEEK'] = daily_df['WEEK'].ffill().astype(int)
daily_df['Max_Temp_C'] = daily_df['Max_Temp_C'].interpolate(method='linear').bfill().ffill()
daily_df['Min_Temp_C'] = daily_df['Min_Temp_C'].interpolate(method='linear').bfill().ffill()
daily_df['Rainfall_mm'] = daily_df['Rainfall_mm'].fillna(0.0)

# 6. Feature Engineering (Rolling Accumulations & Agronomic Features)
daily_df['Diurnal_Temp_Range'] = daily_df['Max_Temp_C'] - daily_df['Min_Temp_C']
daily_df['Growing_Degree_Days'] = np.maximum(0, ((daily_df['Max_Temp_C'] + daily_df['Min_Temp_C']) / 2.0) - 15.0)
daily_df['7_Day_Avg_Max_Temp'] = daily_df['Max_Temp_C'].rolling(window=7, min_periods=1).mean()
daily_df['7_Day_Total_Rain'] = daily_df['Rainfall_mm'].rolling(window=7, min_periods=1).sum()

daily_df.to_csv('data/c4_time_series/interim/cleaned_weather.csv', index=False)
print(f"Saved {len(daily_df)} daily records to data/c4_time_series/interim/cleaned_weather.csv")
print(f"Saved {len(weekly_df)} weekly records to data/c4_time_series/interim/cleaned_weather_weekly.csv")
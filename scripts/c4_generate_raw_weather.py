import pandas as pd
import numpy as np
import os

print("Generating mock raw weather data...")

# 1. Generate daily dates for 2 years
dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
raw_weather = pd.DataFrame({'Date': dates})

# 2. Simulate realistic Sri Lankan micro-climate numbers
np.random.seed(42)
raw_weather['Max_Temp_C'] = np.random.uniform(28.0, 36.0, size=len(dates))
raw_weather['Min_Temp_C'] = np.random.uniform(22.0, 26.0, size=len(dates))
raw_weather['Rainfall_mm'] = np.random.exponential(scale=8.0, size=len(dates))

# 3. Inject missing data (Simulating workers forgetting to log)
raw_weather.loc[15:20, 'Max_Temp_C'] = np.nan
raw_weather.loc[45:50, 'Min_Temp_C'] = np.nan
raw_weather.loc[80:90, 'Rainfall_mm'] = np.nan

# 4. Save to the raw data folder
os.makedirs('data/raw', exist_ok=True)
output_path = 'data/raw/mock_raw_weather.csv'
raw_weather.to_csv(output_path, index=False)

print(f"Raw data with missing values saved to {output_path}")
print("Missing values count:")
print(raw_weather.isnull().sum())
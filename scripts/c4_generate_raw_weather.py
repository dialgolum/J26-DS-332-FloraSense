import pandas as pd
import numpy as np
import os

print("Generating realistic Sri Lankan micro-climate weather data...")

np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
n = len(dates)
day_of_year = dates.dayofyear

# 1. Seasonal baseline (Warmer in March-May, slightly cooler in Dec-Jan)
seasonal_temp = 31.5 + 2.0 * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)

# 2. Smooth day-to-day weather waves (Autoregressive memory so days don't jump wildly)
temp_wave = np.zeros(n)
for i in range(1, n):
    temp_wave[i] = 0.7 * temp_wave[i-1] + np.random.normal(0, 0.8)

# 3. Realistic Rainfall ( ~55% dry days with 0.0mm + bimodal monsoon peaks)
rain_prob = 0.35 + 0.25 * (np.sin(2 * np.pi * (day_of_year - 100) / 182.6) ** 2)
is_raining = np.random.uniform(0, 1, size=n) < rain_prob
rainfall = np.where(is_raining, np.random.exponential(scale=12.0, size=n), 0.0)
rainfall = np.round(rainfall, 1)

# 4. Physical correlation: Heavy rain cools down the daytime Max Temp
rain_cooling = np.clip(rainfall * 0.08, 0, 3.5)
max_temp = np.round(seasonal_temp + temp_wave - rain_cooling, 1)
min_temp = np.round(max_temp - np.random.uniform(6.5, 8.5, size=n), 1)

raw_weather = pd.DataFrame({
    'Date': dates,
    'Max_Temp_C': max_temp,
    'Min_Temp_C': min_temp,
    'Rainfall_mm': rainfall
})

# 5. Inject realistic missing gaps (Workers forgetting to log on certain days)
raw_weather.loc[15:20, 'Max_Temp_C'] = np.nan
raw_weather.loc[45:50, 'Min_Temp_C'] = np.nan
raw_weather.loc[80:90, 'Rainfall_mm'] = np.nan

os.makedirs('data/raw', exist_ok=True)
output_path = 'data/raw/mock_raw_weather.csv'
raw_weather.to_csv(output_path, index=False)

print(f"Realistic mock weather saved to {output_path}")
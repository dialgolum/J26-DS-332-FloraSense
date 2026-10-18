import pandas as pd
import os

print("Starting Weather Data Cleaning Pipeline...")

# 1. Load the raw dataset
try:
    # Update this filename tomorrow to match the real Excel file they give you
    df = pd.read_csv('data/raw/mock_raw_weather.csv') 
except FileNotFoundError:
    print("Error: Raw data file not found in data/raw/")
    exit()

# Ensure the Date column is a proper datetime object and sort chronologically
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. Clean the Data
# Use linear interpolation for temperature (connects the dots between missing days)
df['Max_Temp_C'] = df['Max_Temp_C'].interpolate(method='linear')
df['Min_Temp_C'] = df['Min_Temp_C'].interpolate(method='linear')

# Fill missing rainfall with 0 (Assuming a blank log means it did not rain)
df['Rainfall_mm'] = df['Rainfall_mm'].fillna(0)

# 3. Feature Engineering for Time-Series (Rolling Accumulations)
df['7_Day_Avg_Max_Temp'] = df['Max_Temp_C'].rolling(window=7, min_periods=1).mean()
df['7_Day_Total_Rain'] = df['Rainfall_mm'].rolling(window=7, min_periods=1).sum()

# 4. Save the preprocessed data
os.makedirs('data/interim', exist_ok=True)
output_path = 'data/interim/cleaned_weather.csv'
df.to_csv(output_path, index=False)

print(f"Cleaning complete. Data saved to {output_path}")
print("Missing values remaining:")
print(df.isnull().sum())
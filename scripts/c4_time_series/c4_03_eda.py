import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Running Script 3: Exploratory Data Analysis (EDA)...")

# 1. Load Master Dataset
master_path = os.path.join('data', 'c4_time_series', 'processed', 'c4_master_training_data.csv')
if not os.path.exists(master_path):
    raise FileNotFoundError(f"Missing {master_path}. Run the master merge script first.")

df = pd.read_csv(master_path)
df['Dispatch_Date'] = pd.to_datetime(df['Dispatch_Date'])

# Create output directory for EDA charts
eda_out_dir = os.path.join('data', 'c4_time_series', 'eda_outputs')
os.makedirs(eda_out_dir, exist_ok=True)

# 2. Key Statistical Summary
print("\n--- Summary Statistics ---")
print(df[['Ordered_Quantity', 'Dispatched_Quantity', 'Supply_Demand_Gap',
          'Avg_Max_Temp_C', 'Weekly_Rainfall_mm']].describe())

print(f"\nDataset shape: {df.shape}")
print(f"Date range: {df['Dispatch_Date'].min().date()} to {df['Dispatch_Date'].max().date()}")
print(f"Unique customers: {df['Customer'].nunique()}")
print(f"Unique plant varieties: {df['Plant_Variety'].nunique()} → {sorted(df['Plant_Variety'].unique())}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# 3. Visualization 1: Top Exported Plant Varieties
plt.figure(figsize=(10, 6))
top_plants = df.groupby('Plant_Variety')['Dispatched_Quantity'].sum().sort_values(ascending=True)
colors = plt.cm.YlGn([0.4 + 0.6 * i / len(top_plants) for i in range(len(top_plants))])
top_plants.plot(kind='barh', color=colors)
plt.title('Total Dispatched Quantity by Plant Variety (2023–2026)', fontsize=13, fontweight='bold')
plt.xlabel('Total Dispatched Quantity', fontsize=10)
plt.ylabel('Plant Variety', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(eda_out_dir, 'top_plant_varieties.png'), dpi=300)
plt.close()
print("Saved: top_plant_varieties.png")

# 4. Visualization 2: Weekly Trends of Supply vs Demand over Time
plt.figure(figsize=(14, 6))
weekly_trend = df.groupby('Dispatch_Date')[['Ordered_Quantity', 'Dispatched_Quantity']].sum()
plt.plot(weekly_trend.index, weekly_trend['Ordered_Quantity'],
         label='Ordered Quantity (Demand)', color='orange', alpha=0.85, linewidth=1.2)
plt.plot(weekly_trend.index, weekly_trend['Dispatched_Quantity'],
         label='Dispatched Quantity (Supply)', color='dodgerblue', alpha=0.85, linewidth=1.2)
plt.fill_between(weekly_trend.index,
                 weekly_trend['Ordered_Quantity'],
                 weekly_trend['Dispatched_Quantity'],
                 alpha=0.15, color='red', label='Gap')
plt.title('Overall Export Demand vs. Supply Trend (2023–2026)', fontsize=13, fontweight='bold')
plt.xlabel('Dispatch Date', fontsize=10)
plt.ylabel('Total Quantity', fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(eda_out_dir, 'supply_vs_demand_trend.png'), dpi=300)
plt.close()
print("Saved: supply_vs_demand_trend.png")

# 5. Visualization 3: Correlation Heatmap (Weather, Lags, and Gaps)
plt.figure(figsize=(12, 10))
numeric_cols = [
    'Ordered_Quantity', 'Dispatched_Quantity', 'Supply_Demand_Gap',
    'Avg_Max_Temp_C', 'Weekly_Rainfall_mm', 'Growing_Degree_Days_Weekly',
    'Avg_Max_Temp_C_Lag2W', 'Weekly_Rainfall_mm_Lag2W', 'Growing_Degree_Days_Weekly_Lag2W'
]
corr = df[numeric_cols].corr()
mask = corr.abs() < 0.05   # suppress near-zero noise
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix: Weather Lags, Quantities & Supply-Demand Gap',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(eda_out_dir, 'correlation_heatmap.png'), dpi=300)
plt.close()
print("Saved: correlation_heatmap.png")

# 6. Visualization 4: Annual Supply-Demand Gap by Year
plt.figure(figsize=(10, 5))
annual_gap = df.groupby('YEAR')['Supply_Demand_Gap'].agg(['sum', 'mean'])
ax = annual_gap['sum'].plot(kind='bar', color=['#4CAF50', '#2196F3', '#FF9800', '#E91E63'],
                             edgecolor='white', linewidth=0.8)
plt.title('Total Supply–Demand Gap by Year', fontsize=13, fontweight='bold')
plt.xlabel('Year', fontsize=10)
plt.ylabel('Cumulative Gap (Dispatched − Ordered)', fontsize=10)
plt.xticks(rotation=0)
for bar, val in zip(ax.patches, annual_gap['sum']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(eda_out_dir, 'annual_gap_by_year.png'), dpi=300)
plt.close()
print("Saved: annual_gap_by_year.png")

# 7. Visualization 5: Weekly Seasonal Pattern (avg dispatched by WEEK number)
plt.figure(figsize=(14, 5))
weekly_seasonal = df.groupby('WEEK')['Dispatched_Quantity'].mean()
plt.plot(weekly_seasonal.index, weekly_seasonal.values,
         color='teal', linewidth=1.5, marker='o', markersize=3)
plt.fill_between(weekly_seasonal.index, weekly_seasonal.values, alpha=0.2, color='teal')
plt.title('Average Dispatched Quantity by Week Number (Seasonality Pattern)',
          fontsize=13, fontweight='bold')
plt.xlabel('Week of Year', fontsize=10)
plt.ylabel('Avg Dispatched Quantity per Transaction', fontsize=10)
plt.xticks(range(1, 53, 4))
plt.tight_layout()
plt.savefig(os.path.join(eda_out_dir, 'weekly_seasonality.png'), dpi=300)
plt.close()
print("Saved: weekly_seasonality.png")

print(f"\n✅ EDA Complete! All charts saved to: {eda_out_dir}")

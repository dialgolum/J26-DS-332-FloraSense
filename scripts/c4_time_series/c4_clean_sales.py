import os
import pandas as pd

print("Running Script 1: Cleaning & Standardizing Raw Export Sales Data...")

# 1. Define Input & Output Paths
raw_sales_path = os.path.join('data', 'c4_time_series', 'raw', 'PROJECT.csv')
clean_sales_out = os.path.join('data', 'c4_time_series', 'interim', 'cleaned_sales_2023_2026.csv')

if not os.path.exists(raw_sales_path):
    raise FileNotFoundError(
        f"Cannot find {raw_sales_path}. Please place 'PROJECT.csv' in data/c4_time_series/raw/."
    )

raw_df = pd.read_csv(raw_sales_path)
initial_rows = len(raw_df)

# 2. Remove duplicate packing slip entries and 'SUB TOTAL' summary rows
df_clean = raw_df.drop_duplicates().copy()
df_clean = df_clean[df_clean['Mode'] != 'SUB TOTAL']

# 3. Fix clerical year typos ('.0206' -> '.2026') and parse mixed date formats to YYYY-MM-DD
df_clean['Dispatch_Date'] = (
    df_clean['Dispatch_Date'].astype(str).str.strip().str.replace('.0206', '.2026', regex=False)
)
df_clean['Dispatch_Date_dt'] = pd.to_datetime(df_clean['Dispatch_Date'], format='mixed', dayfirst=True)

# 4. Filter out unrecorded ('NaN') or 'Pending' dispatch rows and convert quantities to integers
df_clean = df_clean.dropna(subset=['Dispatched_Quantity'])
df_clean = df_clean[df_clean['Dispatched_Quantity'] != 'Pending'].copy()
df_clean['Ordered_Quantity'] = df_clean['Ordered_Quantity'].astype(int)
df_clean['Dispatched_Quantity'] = df_clean['Dispatched_Quantity'].astype(float).astype(int)

# 5. Standardize Customer IDs, Plant_Variety taxonomy, Size, and Mode
df_clean['Customer'] = (
    df_clean['Customer'].astype(str).str.strip()
    .str.replace('CUSTOMER- ', 'CUSTOMER-', regex=False)
    .str.replace('CUSTOMER -', 'CUSTOMER-', regex=False)
)

def standardize_plant_variety(val):
    s = ' '.join(str(val).strip().split())  # Collapse double spaces such as 'Codiaeum  spp.'
    taxonomic_map = {
        'Codiaeum SPP.': 'Codiaeum spp.',
        'Monstera SPP.': 'Monstera spp.',
        'Monsteraspp.': 'Monstera spp.',
        'Zamioculcasspp.': 'Zamioculcas spp.',
        'Aglaonema': 'Aglaonema spp.',
        'Dracaena': 'Dracaena spp.',
    }
    return taxonomic_map.get(s, s)

df_clean['Plant_Variety'] = df_clean['Plant_Variety'].apply(standardize_plant_variety)
df_clean['Size'] = df_clean['Size'].astype(str).str.strip().str.lower()
df_clean['Mode'] = df_clean['Mode'].astype(str).str.strip().replace({'un-rooted': 'Un-rooted'})

# 6. Sort strictly chronologically by Dispatch_Date, Customer, and Plant_Variety
df_clean = df_clean.sort_values(['Dispatch_Date_dt', 'Customer', 'Plant_Variety']).reset_index(drop=True)
df_clean['Dispatch_Date'] = df_clean['Dispatch_Date_dt'].dt.strftime('%Y-%m-%d')
df_clean = df_clean.drop(columns=['Dispatch_Date_dt'])

# 7. Save Cleaned Sales Dataset
os.makedirs(os.path.dirname(clean_sales_out), exist_ok=True)
df_clean.to_csv(clean_sales_out, index=False)

print(f"Cleaning Complete! Removed {initial_rows - len(df_clean)} anomalous/duplicate rows.")
print(f"Cleaned Dataset Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")
print(f"Standardized Plant Varieties ({df_clean['Plant_Variety'].nunique()}): {sorted(df_clean['Plant_Variety'].unique())}")
print(f"Saved to: {clean_sales_out}")
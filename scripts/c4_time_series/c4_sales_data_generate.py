import os
from itertools import groupby
import pandas as pd
import numpy as np
from google.colab import files

print("Initializing Raw Floriculture Export Sales Data Generator...")

# 1. Upload or verify source files
excel_path = 'PROJECT  (1).xlsx'
weather_path = 'cleaned_weather_weekly.csv'

if not os.path.exists(excel_path):
    if os.path.exists('PROJECT (1).xlsx'):
        excel_path = 'PROJECT (1).xlsx'
    else:
        print("Please upload 'PROJECT  (1).xlsx':")
        uploaded_excel = files.upload()
        for fn in uploaded_excel.keys():
            os.rename(fn, excel_path)

if not os.path.exists(weather_path):
    print("Please upload 'cleaned_weather_weekly.csv':")
    uploaded_weather = files.upload()
    for fn in uploaded_weather.keys():
        os.rename(fn, weather_path)

# 2. Extract explicit shipping schedules from Mike Flora's Excel sheets
xls = pd.ExcelFile(excel_path)

df_b = pd.read_excel(excel_path, sheet_name='CUSTOMER-B')
raw_dates_b = df_b.iloc[18:33, 4].dropna().astype(str).str.strip().tolist()

df_c = pd.read_excel(excel_path, sheet_name='CUSTOMER -C')
raw_dates_c = df_c.iloc[18:40, 5].dropna().astype(str).str.strip().tolist()

df_d = pd.read_excel(excel_path, sheet_name='CUSTOMER -D ')
raw_dates_d = df_d.iloc[20:30, 4].dropna().astype(str).str.strip().tolist()

df_e = pd.read_excel(excel_path, sheet_name='CUSTOMER -E')
raw_dates_e = df_e.iloc[11:19, 3].dropna().astype(str).str.strip().tolist()

df_f = pd.read_excel(excel_path, sheet_name='CUSTOMER -F')
raw_dates_f = df_f.iloc[12:21, 3].dropna().astype(str).str.strip().tolist()

def build_date_dict(raw_date_list):
    d_dict = {}
    for d_str in raw_date_list:
        clean_d = d_str.replace('.0206', '.2026')
        dt = pd.to_datetime(clean_d, format='%d.%m.%Y')
        wk = dt.isocalendar().week
        d_dict[wk] = (dt, d_str)
    return d_dict

# Exact 2026 lookup table matching raw strings
exact_2026_schedules = {
    'CUSTOMER-B': build_date_dict(raw_dates_b),
    'CUSTOMER -C': build_date_dict(raw_dates_c),
    'CUSTOMER -D ': build_date_dict(raw_dates_d),
    'CUSTOMER -E': build_date_dict(raw_dates_e),
    'CUSTOMER -F': build_date_dict(raw_dates_f),
}

# Weeks of year active for standing orders
b_weeks = set(list(exact_2026_schedules['CUSTOMER-B'].keys()) + [44, 46, 48, 50, 52])
c_weeks = set(list(exact_2026_schedules['CUSTOMER -C'].keys()) + [43, 45, 47, 49, 51])
d_weeks = set(list(exact_2026_schedules['CUSTOMER -D '].keys()) + [44, 48, 52])
e_weeks = set(list(exact_2026_schedules['CUSTOMER -E'].keys()) + [43, 47, 51])
f_weeks = set(list(exact_2026_schedules['CUSTOMER -F'].keys()) + [46, 50])

# 3. Product line item catalog (38 items preserving authentic raw labels)
raw_catalog = [
    # CUSTOMER- A (5 items - Ships Every Thursday)
    {'Customer': 'CUSTOMER- A', 'Plant_Variety': 'Dracaena sanderiana spp.', 'Size': '50/55cm', 'Mode': 'Un-rooted', 'Base_Qty': 11760},
    {'Customer': 'CUSTOMER- A', 'Plant_Variety': 'Dracaena godseffiana spp.', 'Size': '50cm', 'Mode': 'Un-rooted', 'Base_Qty': 550},
    {'Customer': 'CUSTOMER- A', 'Plant_Variety': 'Livistona spp.', 'Size': '20/25cm', 'Mode': 'Leaf', 'Base_Qty': 110},
    {'Customer': 'CUSTOMER- A', 'Plant_Variety': 'Dracaena reflexa variegata spp.', 'Size': '50cm', 'Mode': 'Un-rooted', 'Base_Qty': 1800},
    {'Customer': 'CUSTOMER- A', 'Plant_Variety': 'Cordyline spp.', 'Size': '50cm', 'Mode': 'Un-rooted', 'Base_Qty': 1050},

    # CUSTOMER-B (6 items - Scheduled Saturdays)
    {'Customer': 'CUSTOMER-B', 'Plant_Variety': 'Codiaeum spp.', 'Size': '15/17cm', 'Mode': 'un-rooted', 'Base_Qty': 10000},
    {'Customer': 'CUSTOMER-B', 'Plant_Variety': 'Cordyline spp.', 'Size': '10cm', 'Mode': '1cpn-rooted', 'Base_Qty': 2800},
    {'Customer': 'CUSTOMER-B', 'Plant_Variety': 'Codiaeum spp.', 'Size': '10cm', 'Mode': '1cpn-rooted', 'Base_Qty': 3300},
    {'Customer': 'CUSTOMER-B', 'Plant_Variety': 'Polyscias spp.', 'Size': '10cm', 'Mode': '1cpn-rooted', 'Base_Qty': 6600},
    {'Customer': 'CUSTOMER-B', 'Plant_Variety': 'Philodendron spp.', 'Size': '6x4/8x5cm', 'Mode': '3p-oasis-rooted', 'Base_Qty': 6650},
    {'Customer': 'CUSTOMER-B', 'Plant_Variety': 'Epipremnum spp.', 'Size': '6x4/8x5cm', 'Mode': '3p-oasis-rooted', 'Base_Qty': 5000},

    # CUSTOMER -C (8 items - Fridays + Sundays)
    {'Customer': 'CUSTOMER -C', 'Plant_Variety': 'Codiaeum  spp.', 'Size': '15/17cm', 'Mode': 'Un-rooted', 'Base_Qty': 33509},
    {'Customer': 'CUSTOMER -C', 'Plant_Variety': 'Epipremnum spp.', 'Size': '6X4/8X5cm', 'Mode': 'Un-rooted', 'Base_Qty': 10000},
    {'Customer': 'CUSTOMER -C', 'Plant_Variety': 'Philodendron spp.', 'Size': '6X4/8X5cm', 'Mode': 'Un-rooted', 'Base_Qty': 10000},
    {'Customer': 'CUSTOMER -C', 'Plant_Variety': 'Aglaonema ', 'Size': '20/25cm', 'Mode': 'Un-rooted', 'Base_Qty': 3250},
    {'Customer': 'CUSTOMER -C', 'Plant_Variety': 'Dracaena spp.', 'Size': '20/25cm', 'Mode': '5p-jiffy-pellet-rooted', 'Base_Qty': 1750},
    {'Customer': 'CUSTOMER -C', 'Plant_Variety': 'Plumeria spp.', 'Size': '20cm', 'Mode': '1cpn-rooted', 'Base_Qty': 800},
    {'Customer': 'CUSTOMER -C', 'Plant_Variety': 'Philodendron spp.', 'Size': '6X4/8X5cm', 'Mode': '5p-jiffy-pellet-rooted', 'Base_Qty': 1500},
    {'Customer': 'CUSTOMER -C', 'Plant_Variety': 'Schefflera spp.', 'Size': '15/20cm', 'Mode': '1p-jiffy-pellet-rooted', 'Base_Qty': 500},

    # CUSTOMER -D  (11 items - Scheduled Saturdays)
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Cordyline spp.', 'Size': '20/22cm', 'Mode': 'Un-rooted', 'Base_Qty': 2000},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Dracaena sanderiana spp.', 'Size': '20cm', 'Mode': 'Un-rooted', 'Base_Qty': 6000},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Aglaonema maria', 'Size': '30/35cm', 'Mode': '3p-oasis-rooted', 'Base_Qty': 1280},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Codiaeum spp.', 'Size': '10cm', 'Mode': '1cp-oasis-rooted', 'Base_Qty': 4270},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Cordyline spp.', 'Size': '8cm', 'Mode': '1cpn-rooted', 'Base_Qty': 550},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Dracaena spp.', 'Size': '20/24cm', 'Mode': '3p-oasis-rooted', 'Base_Qty': 3650},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Epipremnum spp.', 'Size': '6x4/8x5cm', 'Mode': '3p-oasis-rooted', 'Base_Qty': 3650},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Monsteraspp.', 'Size': '6x4/8x5cm', 'Mode': '3p-oasis-rooted', 'Base_Qty': 200},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Philodendron spp.', 'Size': '6x4/8x5cm', 'Mode': '5p-oasis-rooted', 'Base_Qty': 300},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Polyscias spp.', 'Size': '10cm', 'Mode': '1cp-oasis-rooted', 'Base_Qty': 4660},
    {'Customer': 'CUSTOMER -D ', 'Plant_Variety': 'Zamioculcasspp.', 'Size': '15/20cm', 'Mode': '1p-bare-rooted', 'Base_Qty': 500},

    # CUSTOMER -E (4 items - Scheduled Saturdays)
    {'Customer': 'CUSTOMER -E', 'Plant_Variety': 'Codiaeum SPP.', 'Size': '20/25cm', 'Mode': '1p-oasis-rooted', 'Base_Qty': 1000},
    {'Customer': 'CUSTOMER -E', 'Plant_Variety': 'Dracaena ', 'Size': '20/25cm', 'Mode': '5p-oasis-rooted', 'Base_Qty': 3750},
    {'Customer': 'CUSTOMER -E', 'Plant_Variety': 'Monstera SPP.', 'Size': '6x4/8x5cm', 'Mode': '5p-oasis-rooted', 'Base_Qty': 500},
    {'Customer': 'CUSTOMER -E', 'Plant_Variety': 'Plumeria spp.', 'Size': '30cm', 'Mode': '1cpn-rooted', 'Base_Qty': 200},

    # CUSTOMER -F (4 items - Mondays + Tuesday)
    {'Customer': 'CUSTOMER -F', 'Plant_Variety': 'Aglaonema spp.', 'Size': '40/45cm', 'Mode': '1p-oasis-rooted', 'Base_Qty': 800},
    {'Customer': 'CUSTOMER -F', 'Plant_Variety': 'Codiaeum spp.', 'Size': '20/25cm', 'Mode': '1ppn-rooted', 'Base_Qty': 250},
    {'Customer': 'CUSTOMER -F', 'Plant_Variety': 'Dracaena spp.', 'Size': '12/15cm', 'Mode': '1p-bare-rooted', 'Base_Qty': 4350},
    {'Customer': 'CUSTOMER -F', 'Plant_Variety': 'Epipremnum spp.', 'Size': '6x4/8x5cm', 'Mode': '5p-oasis-rooted', 'Base_Qty': 2000},
]

# 4. Prepare weekly weather lookup and extend for October 2026 Weeks 40-43
weekly_weather = pd.read_csv(weather_path)
w_df = weekly_weather.sort_values(['YEAR', 'WEEK']).reset_index(drop=True).copy()

for lag in [2, 4]:
    w_df[f'Avg_Max_Temp_C_Lag{lag}W'] = w_df['Avg_Max_Temp_C'].shift(lag).bfill()
    w_df[f'Weekly_Rainfall_mm_Lag{lag}W'] = w_df['Weekly_Rainfall_mm'].shift(lag).bfill()
    w_df[f'Growing_Degree_Days_Weekly_Lag{lag}W'] = w_df['Growing_Degree_Days_Weekly'].shift(lag).bfill()

base_monday = pd.Timestamp('2023-01-02')
w_df['Week_Seq'] = (w_df['YEAR'] - 2023) * 52 + (w_df['WEEK'] - 1)
w_df['Week_Start_Date'] = base_monday + pd.to_timedelta(w_df['Week_Seq'] * 7, unit='D')

# Append Weeks 40 to 43 in 2026 to align with October contract dates
ext_rows = []
last_row = w_df.iloc[-1]
for add_wk in [40, 41, 42, 43]:
    ext_row = last_row.copy()
    ext_row['YEAR'] = 2026
    ext_row['WEEK'] = add_wk
    ext_row['Week_Seq'] = (2026 - 2023) * 52 + (add_wk - 1)
    ext_row['Week_Start_Date'] = base_monday + pd.to_timedelta(ext_row['Week_Seq'] * 7, unit='D')
    ext_rows.append(ext_row)

w_df = pd.concat([w_df, pd.DataFrame(ext_rows)], ignore_index=True)

# 5. Generate historical dispatches using exact dates and weekend patterns
np.random.seed(42)
raw_rows = []

for _, w_row in w_df.iterrows():
    yr, wk = int(w_row['YEAR']), int(w_row['WEEK'])
    wk_start = w_row['Week_Start_Date']
    yoy_factor = {2023: 0.91, 2024: 0.95, 2025: 0.98, 2026: 1.00}[yr]
    seasonal_wave = 1.0 + 0.06 * np.sin(2 * np.pi * (wk - 6) / 52.0) + 0.04 * np.cos(4 * np.pi * (wk - 14) / 52.0)

    for cust_label, group_items in groupby(raw_catalog, key=lambda x: x['Customer']):
        items_list = list(group_items)
        cust_clean = cust_label.replace(' ', '')

        is_primary_schedule = (
            (cust_clean == 'CUSTOMER-A') or
            (cust_clean == 'CUSTOMER-B' and wk in b_weeks) or
            (cust_clean == 'CUSTOMER-C' and wk in c_weeks) or
            (cust_clean == 'CUSTOMER-D' and wk in d_weeks) or
            (cust_clean == 'CUSTOMER-E' and wk in e_weeks) or
            (cust_clean == 'CUSTOMER-F' and wk in f_weeks)
        )

        # Allow supplementary/spot orders on ~35% of off-weeks
        if not is_primary_schedule:
            if yr == 2026 or np.random.rand() > 0.35:
                continue

        # In 2026: Strict matching of all explicit contract dates
        if yr == 2026 and cust_label in exact_2026_schedules:
            if wk in exact_2026_schedules[cust_label]:
                _, date_str = exact_2026_schedules[cust_label][wk]
            else:
                continue
        elif cust_clean == 'CUSTOMER-A':
            # Thursday dispatch
            dt_obj = wk_start + pd.Timedelta(days=3)
            date_str = dt_obj.strftime('%d.%m.%Y' if np.random.rand() < 0.25 else '%Y-%m-%d')
        elif cust_clean in ['CUSTOMER-C', 'CUSTOMER -C']:
            # Fridays with ~10% Sunday air-cargo consignments (e.g. 15.03 & 05.07)
            r = np.random.rand()
            if r < 0.10:
                dt_obj = wk_start + pd.Timedelta(days=6)  # Sunday
            elif r < 0.15:
                dt_obj = wk_start + pd.Timedelta(days=5)  # Saturday
            else:
                dt_obj = wk_start + pd.Timedelta(days=4)  # Friday
            date_str = dt_obj.strftime('%d.%m.%Y' if np.random.rand() < 0.20 else '%Y-%m-%d')
        elif cust_clean in ['CUSTOMER-B', 'CUSTOMER-D', 'CUSTOMER-E']:
            # Saturdays with ~8% Sunday packhouse shifts
            r = np.random.rand()
            if r < 0.08:
                dt_obj = wk_start + pd.Timedelta(days=6)  # Sunday
            else:
                dt_obj = wk_start + pd.Timedelta(days=5)  # Saturday
            date_str = dt_obj.strftime('%d.%m.%Y' if np.random.rand() < 0.20 else '%Y-%m-%d')
        elif cust_clean == 'CUSTOMER-F':
            # Mondays with ~10% Tuesday consignments (e.g. 07.04)
            r = np.random.rand()
            if r < 0.10:
                dt_obj = wk_start + pd.Timedelta(days=1)  # Tuesday
            else:
                dt_obj = wk_start + pd.Timedelta(days=0)  # Monday
            date_str = dt_obj.strftime('%d.%m.%Y' if np.random.rand() < 0.20 else '%Y-%m-%d')

        batch_ordered_sum = 0
        batch_dispatched_sum = 0
        spot_ratio = np.random.uniform(0.45, 0.65)

        for item in items_list:
            if is_primary_schedule:
                if yr == 2026:
                    ordered_qty = item['Base_Qty']
                else:
                    ordered_qty = int(round((item['Base_Qty'] * yoy_factor * seasonal_wave * np.random.normal(1.0, 0.02)) / 10.0) * 10)
            else:
                ordered_qty = int(round((item['Base_Qty'] * spot_ratio * yoy_factor * seasonal_wave) / 10.0) * 10)
            ordered_qty = max(50, ordered_qty)

            # Physiological weather lag sensitivity
            is_rooted = 'rooted' in item['Mode'].lower() and 'un-rooted' not in item['Mode'].lower()
            if not is_rooted:
                rel_temp = w_row['Avg_Max_Temp_C_Lag2W']
                rel_rain = w_row['Weekly_Rainfall_mm_Lag2W']
                rel_gdd = w_row['Growing_Degree_Days_Weekly_Lag2W']
            else:
                rel_temp = 0.4 * w_row['Avg_Max_Temp_C_Lag2W'] + 0.6 * w_row['Avg_Max_Temp_C_Lag4W']
                rel_rain = 0.4 * w_row['Weekly_Rainfall_mm_Lag2W'] + 0.6 * w_row['Weekly_Rainfall_mm_Lag4W']
                rel_gdd = 0.4 * w_row['Growing_Degree_Days_Weekly_Lag2W'] + 0.6 * w_row['Growing_Degree_Days_Weekly_Lag4W']

            heat_loss = max(0.0, (rel_temp - 34.0) * 0.042)
            rain_loss = max(0.0, (rel_rain - 55.0) * 0.0011)
            gdd_gain = np.clip((rel_gdd - 101.5) * 0.0022, -0.025, 0.030)

            # Realistic packing buffers
            fulfillment_ratio = np.clip(1.015 + gdd_gain - heat_loss - rain_loss + np.random.normal(0, 0.012), 0.76, 1.03)
            raw_disp = ordered_qty * fulfillment_ratio
            if ordered_qty >= 500 and ordered_qty != 33509:
                dispatched_qty = int(round(raw_disp / 5.0) * 5)
            else:
                dispatched_qty = int(round(raw_disp))

            r_miss = np.random.rand()
            if r_miss < 0.005:
                disp_val = np.nan
            elif r_miss < 0.010:
                disp_val = "Pending"
            else:
                disp_val = dispatched_qty
                batch_dispatched_sum += dispatched_qty

            batch_ordered_sum += ordered_qty

            row_dict = {
                'Dispatch_Date': date_str,
                'Customer': item['Customer'],
                'Plant_Variety': item['Plant_Variety'],
                'Size': item['Size'],
                'Mode': item['Mode'],
                'Ordered_Quantity': ordered_qty,
                'Dispatched_Quantity': disp_val
            }
            raw_rows.append(row_dict)

            # Inline duplicate packing slips
            if np.random.rand() < 0.008 and isinstance(disp_val, int):
                raw_rows.append(row_dict.copy())

        # Mathematical SUB TOTAL records
        if cust_clean in ['CUSTOMER-B', 'CUSTOMER-C', 'CUSTOMER-D'] and wk % 14 == 0:
            raw_rows.append({
                'Dispatch_Date': np.nan,
                'Customer': 'SUB TOTAL',
                'Plant_Variety': np.nan,
                'Size': np.nan,
                'Mode': 'SUB TOTAL',
                'Ordered_Quantity': batch_ordered_sum,
                'Dispatched_Quantity': batch_dispatched_sum
            })

raw_df = pd.DataFrame(raw_rows)
raw_sales_out = 'raw_export_sales_2023_2026.csv'
raw_df.to_csv(raw_sales_out, index=False)

print(f"\nGenerated {len(raw_df)} total records (saved to {raw_sales_out}).")

# Day-of-week verification
valid_rows = raw_df[raw_df['Mode'] != 'SUB TOTAL'].copy()
valid_rows['dt'] = pd.to_datetime(valid_rows['Dispatch_Date'].astype(str).str.replace('.0206', '.2026', regex=False), format='mixed', dayfirst=True)
print("\nVerified Day-of-Week Consignment Breakdown:")
print(valid_rows['dt'].dt.day_name().value_counts())

files.download(raw_sales_out)
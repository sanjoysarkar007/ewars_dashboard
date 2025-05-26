# pipeline/process_data.py

import pandas as pd
import os

# Paths (adjust as needed)
DATA_DIR = "../dengue_ewars_dashboard/data"
OUTPUT_DIR = "../dengue_ewars_dashboard/data"

DENGUE_FILE = os.path.join(DATA_DIR, "dengue.csv")
WEATHER_FILE = os.path.join(DATA_DIR, "met_data.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_data.csv")

# Thresholds (can also be moved to utils/config.py)
ALERT_THRESHOLD = 100

def load_data():
    dengue = pd.read_csv(DENGUE_FILE)
    weather = pd.read_csv(WEATHER_FILE)
    return dengue, weather

def clean_and_merge(dengue, weather):
    # Standardize district names if needed (optional)
    dengue['district'] = dengue['district'].str.strip().str.lower()
    weather['district'] = weather['district'].str.strip().str.lower()

    # Merge on district, year, week
    merged = pd.merge(dengue, weather, on=['district', 'year', 'week'], how='left')

    # Rename for clarity
    merged.rename(columns={
        'weekly hospitalized': 'cases',
        'Temp': 'temp',
        'Rainfall': 'rainfall',
        'humidity': 'humidity'
    }, inplace=True)

    # Fill missing weather data with NaN or method
    merged[['temp', 'rainfall', 'humidity']] = merged[['temp', 'rainfall', 'humidity']].fillna(method='ffill')

    # Create alert column
    merged['alert'] = merged['cases'] > ALERT_THRESHOLD

    return merged

def save_data(merged):
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Merged data saved to {OUTPUT_FILE}")

def main():
    dengue, weather = load_data()
    merged = clean_and_merge(dengue, weather)
    save_data(merged)

if __name__ == "__main__":
    main()

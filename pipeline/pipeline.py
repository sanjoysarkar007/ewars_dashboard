
import pandas as pd
import requests
import os

# ------------------------------
# CONFIGURATION
# ------------------------------
DENGUE_DATA = '../dengue_ewars_dashboard/data/dengue.csv'
MET_DATA = '../dengue_ewars_dashboard/data/met_data.csv'
OUTPUT_FILE = '../dengue_ewars_dashboard/data/final_ewars_dataset.csv'
ALERT_THRESHOLD = 10

# ------------------------------
# STEP 1: Load Health Data
# ------------------------------
def load_health_data():
    print("📥 Loading DHIS2 data...")
    return pd.read_csv(DENGUE_DATA)

# ------------------------------
# STEP 2: Load Meteorological Data
# ------------------------------
def load_met_data():
    print("📥 Loading BMD data...")
    return pd.read_csv(MET_DATA)

# ------------------------------
# STEP 3: Merge the Data
# ------------------------------
def merge_data(health_df, met_df):
    print("🔗 Merging datasets...")
   
    merged_df = pd.merge(health_df, met_df, on=['district', 'year', 'week'], how='left')
    return merged_df

# ------------------------------
# STEP 4: Alert Logic
# ------------------------------
def add_alerts(df):
    print("🚨 Applying alert threshold...")
    df['alert'] = df['weekly hospitalized'].apply(lambda x: 'TRUE' if x >= ALERT_THRESHOLD else 'FALSE')
    return df

# ------------------------------
# STEP 5: Save Output
# ------------------------------
def save_output(df):
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Final dataset saved to: {OUTPUT_FILE}")

# ------------------------------
# MAIN PIPELINE RUNNER
# ------------------------------
def run_pipeline():
    print("🚀 Starting EWARS pipeline...\n")
    health_df = load_health_data()
    met_df = load_met_data()
    merged_df = merge_data(health_df, met_df)
    final_df = add_alerts(merged_df)
    save_output(final_df)
    print("\n✅ Pipeline completed successfully.")

# ------------------------------
# EXECUTE
# ------------------------------
if __name__ == "__main__":
    run_pipeline()

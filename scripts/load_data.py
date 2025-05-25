import pandas as pd

# Load Health Data
health_df = pd.read_csv('../dengue_ewars_dashboard/data/dengue.csv')
print("✅ Loaded health data:")
print(health_df.head())

# Load Meteorological Data
met_df = pd.read_csv('../dengue_ewars_dashboard/data/met_data.csv')
print("✅ Loaded meteorological data:")
print(met_df.head())




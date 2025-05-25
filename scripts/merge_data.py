import pandas as pd

# Load health data
health_df = pd.read_csv('../dengue_ewars_dashboard/data/dengue.csv')

# Load meteorological data
met_df = pd.read_csv('../dengue_ewars_dashboard/data/met_data.csv')


# Merge on district and date
merged_df = pd.merge(health_df, met_df, on=['district', 'year', 'week'], how='left')

# Show preview
print("✅ Merged dataset:")
print(merged_df.head())

# Save merged data
merged_df.to_csv('../dengue_ewars_dashboard/data/merged_data.csv', index=False)
print("\n💾 Merged data saved to: merged_data.csv")

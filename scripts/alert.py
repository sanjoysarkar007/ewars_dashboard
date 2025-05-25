import pandas as pd

# Load merged data
df = pd.read_csv('../dengue_ewars_dashboard/data/merged_data.csv')

# Define threshold
threshold = 10

# Trigger alerts
df['alert'] = df['weekly hospitalized'].apply(lambda x: 'TRUE' if x >= threshold else 'FALSE')

# Preview alerts
print(df[['district', 'year', 'weekly hospitalized', 'alert']].head(10))

# Save alert data
df.to_csv('../dengue_ewars_dashboard/data/merged_with_alerts.csv', index=False)
print("\n💾 Alerts saved to: merged_with_alerts.csv")

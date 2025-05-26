import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import plotly.graph_objects as go

# Load preprocessed data
df = pd.read_csv("/mnt/d/dengue_ewars_dashboard/data/final_ewars_dataset.csv")
geo_df = gpd.read_file("/mnt/d/dengue_ewars_dashboard/data/bangladesh.geojson")

# Sidebar filters
weeks = sorted(df['week'].unique())
districts = sorted(df['district'].unique())

selected_week = st.sidebar.selectbox("Select Week", ['All'] + weeks)
selected_district = st.sidebar.selectbox("Select District", districts)

# Filtered data
if selected_week == 'All':
    filtered_df = df
else:
    filtered_df = df[df['week'] == selected_week]

district_df = df[df['district'] == selected_district]

# Merge with geo data for map
latest = filtered_df[filtered_df['year'] == filtered_df['year'].max()]
map_df = geo_df.merge(latest, left_on="NAME_3", right_on="district")

# Create folium map
m = folium.Map(location=[23.6850, 90.3563], zoom_start=7)

folium.Choropleth(
    geo_data=map_df,
    name="choropleth",
    data=map_df,
    columns=["district", "weekly hospitalized"],
    key_on="feature.properties.NAME_3",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Weekly Hospitalized Cases",
).add_to(m)

# Add popups with basic info
for _, row in map_df.iterrows():
    popup_text = f"""
    <b>{row['district'].title()}</b><br>
    Week {row['week']}<br>
    Cases: {row['weekly hospitalized']}<br>
    Temp: {row['temp']}°C<br>
    Rainfall: {row['rainfall']}mm<br>
    Humidity: {row['humidity']}%
    """
    folium.Marker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        popup=popup_text,
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

folium_static(m)

# Plot trends for selected district
fig = go.Figure()
fig.add_trace(go.Scatter(x=district_df['week'], y=district_df['weekly hospitalized'], mode='lines+markers', name='Cases'))
fig.add_trace(go.Scatter(x=district_df['week'], y=district_df['temp'], mode='lines+markers', name='Temperature'))
fig.add_trace(go.Scatter(x=district_df['week'], y=district_df['rainfall'], mode='lines+markers', name='Rainfall'))
fig.add_trace(go.Scatter(x=district_df['week'], y=district_df['humidity'], mode='lines+markers', name='Humidity'))

# Optional threshold line (customizable)
alert_threshold = st.sidebar.slider("Set Alert Threshold", min_value=0, max_value=1000, value=300)
fig.add_trace(go.Scatter(x=district_df['week'], y=[alert_threshold]*len(district_df), mode='lines', name='Threshold', line=dict(dash='dot')))

fig.update_layout(title=f"Weather & Case Trends for {selected_district}", xaxis_title="Week", yaxis_title="Values")
st.plotly_chart(fig)

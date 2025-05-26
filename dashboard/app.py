import streamlit as st
import pandas as pd
import plotly.express as px


# Load dataset
df = pd.read_csv("/mnt/d/dengue_ewars_dashboard/data/final_ewars_dataset.csv")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Data")
districts = df['district'].unique().tolist()
years = df['year'].unique().tolist()

selected_districts = st.sidebar.multiselect("Select District(s)", districts, default=districts)
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)

# Filter data
filtered_df = df[
    (df['district'].isin(selected_districts)) &
    (df['year'].isin(selected_years))
]

import streamlit as st

# Unique weeks and districts
weeks = sorted(df['week'].unique())
weeks_with_all = ['All'] + weeks  # Add 'All' option


# Sidebar filters
selected_week = st.sidebar.selectbox("Select Week", weeks_with_all)


# Filter data based on selection
if selected_week == 'All':
    filtered_df = df.copy()
else:
    filtered_df = df[df['week'] == selected_week]




# --- Dashboard Title ---
st.title("🦟 Dengue + Weather Early Warning Dashboard")
st.markdown("Track dengue cases with weather trends and alert signals.")

# --- KPI Cards ---
total_cases = filtered_df["weekly hospitalized"].sum()
total_alerts = filtered_df["alert"].sum()
avg_temp = round(filtered_df["temp"].mean(), 2)
avg_rainfall = round(filtered_df["rainfall"].mean(), 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", total_cases)
col2.metric("Total Alerts", total_alerts)
col3.metric("Avg. temp (°C)", avg_temp)
col4.metric("Avg. rainfall (mm)", avg_rainfall)

st.markdown("---")

# --- Line Chart: Dengue Trends ---
st.subheader("📈 Dengue Cases Over Weeks")

fig1 = px.line(
    filtered_df,
    x="week",
    y="weekly hospitalized",
    color="district",
    line_group="year",
    markers=True,
    title="Weekly Hospitalized Dengue Cases"
)
st.plotly_chart(fig1, use_container_width=True)

# --- Table: Alerts ---
st.subheader("🚨 District-Week Level Alert Flags")
alerts_df = filtered_df[filtered_df["alert"] == True][["district", "year", "week", "weekly hospitalized", "temp", "rainfall"]]
st.dataframe(alerts_df.sort_values(by=["district", "year", "week"]))

# --- Expandable Raw Data View ---
with st.expander("📄 View Raw Merged Data"):
    st.dataframe(filtered_df)

import plotly.graph_objects as go
import streamlit as st

st.subheader("📊 Weather Parameters vs Dengue Hospitalizations")

# Sidebar input for threshold
user_threshold = st.sidebar.number_input(
    "🔔 Set Alert Threshold for Hospitalizations", 
    min_value=0, max_value=1000, value=50, step=5
)

# User input: district and year
sel_col1, sel_col2 = st.columns(2)
selected_district = sel_col1.selectbox("Choose District", sorted(df["district"].unique()))
selected_year = sel_col2.selectbox("Choose Year", sorted(df["year"].unique()))

# Filter data
district_df = df[(df["district"] == selected_district) & (df["year"] == selected_year)].sort_values("week")

# Create multi-axis chart
fig2 = go.Figure()

# Hospitalized cases
fig2.add_trace(go.Scatter(
    x=district_df["week"],
    y=district_df["weekly hospitalized"],
    name="🧑‍⚕️ Weekly Hospitalized Cases",
    mode="lines+markers",
    line=dict(color="firebrick", width=3),
    hovertemplate="Week %{x}<br>Cases: %{y}<extra></extra>"
))

# Threshold line
fig2.add_trace(go.Scatter(
    x=district_df["week"],
    y=[user_threshold]*len(district_df),
    name="🔴 Alert Threshold",
    mode="lines",
    line=dict(color="firebrick", width=2, dash="dot"),
    hovertemplate="Threshold: %{y}<extra></extra>"
))

# Temperature
fig2.add_trace(go.Scatter(
    x=district_df["week"],
    y=district_df["temp"],
    name="🌡️ Temperature (°C)",
    yaxis="y2",
    mode="lines+markers",
    line=dict(color="orange"),
    hovertemplate="Temp: %{y}°C<extra></extra>"
))

# Rainfall
fig2.add_trace(go.Scatter(
    x=district_df["week"],
    y=district_df["rainfall"],
    name="🌧️ Rainfall (mm)",
    yaxis="y3",
    mode="lines+markers",
    line=dict(color="blue"),
    hovertemplate="Rainfall: %{y}mm<extra></extra>"
))

# Humidity
fig2.add_trace(go.Scatter(
    x=district_df["week"],
    y=district_df["humidity"],
    name="💧 Humidity (%)",
    yaxis="y4",
    mode="lines+markers",
    line=dict(color="green"),
    hovertemplate="Humidity: %{y}%<extra></extra>"
))

# Update layout with enhanced readability
fig2.update_layout(
    title=f"Weather vs Dengue Trend in {selected_district} ({selected_year})",
    xaxis=dict(title="Week"),
    yaxis=dict(
        title=dict(text="Hospitalized Cases", font=dict(color="firebrick")),
        tickfont=dict(color="firebrick")
    ),
    yaxis2=dict(
        title=dict(text="Temperature (°C)", font=dict(color="orange")),
        tickfont=dict(color="orange"),
        overlaying="y",
        side="right"
    ),
    yaxis3=dict(
        title=dict(text="Rainfall (mm)", font=dict(color="blue")),
        tickfont=dict(color="blue"),
        overlaying="y",
        side="left",
        position=0.05
    ),
    yaxis4=dict(
        title=dict(text="Humidity (%)", font=dict(color="green")),
        tickfont=dict(color="green"),
        overlaying="y",
        side="right",
        position=0.95
    ),
    legend=dict(orientation="h", y=-0.3),
    margin=dict(l=60, r=60, t=60, b=60),
    height=550
)

st.plotly_chart(fig2, use_container_width=True)

import geopandas as gpd
import folium
from streamlit_folium import folium_static

# Load spatial data and latest case data
@st.cache_data
def load_geo():
    return gpd.read_file("/mnt/d/dengue_ewars_dashboard/data/bangladesh.geojson")

geo_df = load_geo()

# Get latest data (assumes latest year/week combo)
latest = df[df[["year", "week"]].apply(tuple, axis=1) == df[["year", "week"]].apply(tuple, axis=1).max()]
map_df = geo_df.merge(latest, left_on="NAME_3", right_on="district")

# Choropleth Map
st.subheader("🗺️ District-wise Dengue Cases (Latest Week)")
m = folium.Map(location=[23.685, 90.3563], zoom_start=7, tiles="CartoDB positron")

choropleth = folium.Choropleth(
    geo_data=map_df,
    name="Choropleth",
    data=map_df,
    columns=["district", "weekly hospitalized"],
    key_on="feature.properties.NAME_3",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Weekly Dengue weekly hospitalized",
    highlight=True,
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



# Add district labels
for _, r in map_df.iterrows():
    tooltip_text = f"{r['district']}: {r['weekly hospitalized']} cases"
    folium.Tooltip(tooltip_text).add_to(folium.GeoJson(r["geometry"]))

folium.LayerControl().add_to(m)
folium_static(m, width=900, height=600)

import seaborn as sns
import matplotlib.pyplot as plt

heat_data = df.pivot_table(index='district', columns='week', values='alert', aggfunc='max')
plt.figure(figsize=(10, 20))
sns.heatmap(heat_data, cmap="Reds", cbar=True, linewidths=0.5, linecolor='gray')

st.pyplot(plt)


import plotly.graph_objects as go

fig = go.Figure()

# Weekly hospitalized cases
fig.add_trace(go.Scatter(
    x=district_df['week'], y=district_df['weekly hospitalized'],
    mode='lines+markers', name='Cases',
    line=dict(color='red')
))

# Temperature
fig.add_trace(go.Scatter(
    x=district_df['week'], y=district_df['rainfall'],
    mode='lines+markers', name='Temperature',
    yaxis='y2', line=dict(color='blue')
))

# Layout with secondary axis
fig.update_layout(
    title="Cases and Weather Trend",
    yaxis=dict(title='Cases'),
    yaxis2=dict(title='Temperature', overlaying='y', side='right'),
    legend=dict(x=0, y=1.1, orientation='h')
)
st.plotly_chart(fig)





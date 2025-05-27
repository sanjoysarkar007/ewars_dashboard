import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.forecast import prepare_forecast_data, train_and_forecast

# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px


import streamlit as st
import pandas as pd

# --- Load dataset ---
df = pd.read_csv("data/final_ewars_dataset.csv")

# --- Dashboard Title ---
st.title("🦟 Dengue + Weather Early Warning Dashboard")
st.markdown("Track dengue cases with weather trends and alert signals.")

# --- Top Filters ---
st.subheader("🔍 Filter Data")

# Create columns for filters
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    districts = df['district'].unique().tolist()
    selected_districts = st.multiselect("Select District(s)", districts, default=districts)

with col2:
    years = df['year'].unique().tolist()
    selected_years = st.multiselect("Select Year(s)", years, default=years)

with col3:
    weeks = sorted(df['week'].unique())
    weeks_with_all = ['All'] + weeks
    selected_week = st.selectbox("Select Week", weeks_with_all)

# --- Filter data ---
filtered_df = df[
    (df['district'].isin(selected_districts)) &
    (df['year'].isin(selected_years))
]

if selected_week != 'All':
    filtered_df = filtered_df[filtered_df['week'] == selected_week]

# --- KPI Cards ---
st.markdown("---")
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

st.subheader("🚨 District-Week Level Alert Flags")

# Filter only necessary columns
alerts_df = filtered_df[["district", "year", "week", "weekly hospitalized", "temp", "rainfall", "alert"]]

# Custom styling function
def highlight_alert(val):
    color = 'red' if val is True else 'green'
    return f'background-color: {color}; color: white'

# Apply style only to 'alert' column
styled_df = alerts_df.sort_values(by=["district", "year", "week"]).style.applymap(highlight_alert, subset=['alert'])

st.write(styled_df)



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
import streamlit as st
from streamlit_folium import folium_static

# Load spatial data and latest case data
@st.cache_data
def load_geo():
    return gpd.read_file("data/bangladesh.geojson")

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
    legend_name="Weekly Dengue hospitalized cases",
    highlight=True,
).add_to(m)

# Add tooltips with hospitalized case numbers
geojson = folium.GeoJson(
    map_df,
    tooltip=folium.GeoJsonTooltip(
        fields=["district", "weekly hospitalized"],
        aliases=["District", "Hospitalized Cases"],
        localize=True,
        sticky=True
    )
).add_to(m)

# Add popups with detailed info
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

folium.LayerControl().add_to(m)

# Display the map
folium_static(m, width=900, height=600)


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
    x=district_df['week'], y=district_df['temp'],
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



from utils.forecast import prepare_forecast_data, train_and_forecast
import plotly.graph_objects as go

# Create forecast section in Streamlit
st.header("🔮 Forecasting Dengue Cases")

# Sidebar filter for district
forecast_district = st.selectbox("Select District for Forecast", df['district'].unique())

# Number of weeks to forecast
num_weeks = st.slider("Number of Weeks to Forecast", 4, 24, step=4, value=12)

# Run forecast
df_prophet = prepare_forecast_data(df, forecast_district)
forecast = train_and_forecast(df_prophet, periods=num_weeks)

# Plot using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines+markers', name='Actual Cases'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))

st.plotly_chart(fig, use_container_width=True)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.forecast import prepare_forecast_data, train_and_forecast

# Core Libraries

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import statsmodels.api as sm

# === Load Data ===

df = pd.read_csv("/mnt/d/dengue_ewars_dashboard/data/final_ewars_dataset.csv")

# === Title & Description ===

st.title("🦟 Dengue & Weather Early Warning Dashboard")
st.markdown("Monitor dengue outbreaks with integrated weather trends, spatial mapping, and predictive analytics.")

# === Filter Section ===

st.sidebar.header("🔍 Filter Data")
districts = df['district'].unique().tolist()
years = df['year'].unique().tolist()
weeks = sorted(df['week'].unique())
weeks_with_all = ['All'] + weeks

selected_districts = st.sidebar.multiselect("Select District(s)", districts, default=districts)
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
selected_week = st.sidebar.selectbox("Select Week", weeks_with_all)

filtered_df = df[(df['district'].isin(selected_districts)) & (df['year'].isin(selected_years))]
if selected_week != 'All': filtered_df = filtered_df[filtered_df['week'] == selected_week]

# === KPI Metrics ===

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("🧑‍⚕️ Total Cases", filtered_df["weekly hospitalized"].sum())
col2.metric("🚨 Total Alerts", filtered_df["alert"].sum())
col3.metric("🌡️ Avg. Temp (°C)", round(filtered_df["temp"].mean(), 2))
col4.metric("🌧️ Avg. Rainfall (mm)", round(filtered_df["rainfall"].mean(), 2))
st.markdown("---")

# === Line Chart: Weekly Cases ===

st.subheader("📈 Dengue Cases Over Time")
fig1 = px.line(
filtered_df, x="week", y="weekly hospitalized", color="district",
line_group="year", markers=True, title="Weekly Hospitalized Dengue Cases"
)
st.plotly_chart(fig1, use_container_width=True)

# === Alerts Table ===
st.subheader("🚨 District-Week Alert Table")
def highlight_alert(val):
    color = 'red' if val else 'green'
    return f'background-color: {color}; color: white'

alerts_df = filtered_df[["district", "year", "week", "weekly hospitalized", "temp", "rainfall", "alert"]]
st.dataframe(alerts_df.style.applymap(highlight_alert, subset=['alert']))

# === Multi-axis Weather vs Dengue ===

st.subheader("📊 Weather Factors vs Dengue Cases")
user_threshold = st.sidebar.slider("Set Alert Threshold", 0, 1000, 50, 5)
sel_col1, sel_col2 = st.columns(2)
selected_district = sel_col1.selectbox("Choose District", sorted(df["district"].unique()))
selected_year = sel_col2.selectbox("Choose Year", sorted(df["year"].unique()))

district_df = df[(df["district"] == selected_district) & (df["year"] == selected_year)].sort_values("week")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=district_df["week"], y=district_df["weekly hospitalized"], name="Cases", line=dict(color="crimson")))
fig2.add_trace(go.Scatter(x=district_df["week"], y=[user_threshold]*len(district_df), name="Alert Threshold", line=dict(dash="dot", color="red")))
fig2.add_trace(go.Scatter(x=district_df["week"], y=district_df["temp"], name="Temperature", yaxis="y2", line=dict(color="orange")))
fig2.add_trace(go.Scatter(x=district_df["week"], y=district_df["rainfall"], name="Rainfall", yaxis="y3", line=dict(color="blue")))
fig2.add_trace(go.Scatter(x=district_df["week"], y=district_df["humidity"], name="Humidity", yaxis="y4", line=dict(color="green")))
fig2.update_layout(
title=f"Weather vs Dengue Trend in {selected_district} ({selected_year})",
yaxis=dict(title="Cases", color="crimson"),
yaxis2=dict(title="Temp (°C)", overlaying="y", side="right", color="orange"),
yaxis3=dict(title="Rainfall", overlaying="y", side="left", position=0.05, color="blue"),
yaxis4=dict(title="Humidity", overlaying="y", side="right", position=0.95, color="green"),
legend=dict(orientation="h", y=-0.3), height=550
)
st.plotly_chart(fig2, use_container_width=True)

# === Map ===

st.subheader("🗺️ District-wise Dengue Cases (Latest Week)")
@st.cache_data
def load_geo(): return gpd.read_file("/mnt/d/dengue_ewars_dashboard/data/bangladesh.geojson")
geo_df = load_geo()
latest = df[df[["year", "week"]].apply(tuple, axis=1) == df[["year", "week"]].apply(tuple, axis=1).max()]
map_df = geo_df.merge(latest, left_on="NAME_3", right_on="district")
m = folium.Map(location=[23.685, 90.3563], zoom_start=7, tiles="CartoDB positron")
folium.Choropleth(
geo_data=map_df,
data=map_df,
columns=["district", "weekly hospitalized"],
key_on="feature.properties.NAME_3",
fill_color="YlOrRd", fill_opacity=0.7, line_opacity=0.2,
legend_name="Weekly Dengue hospitalized cases"
).add_to(m)

for _, row in map_df.iterrows():
    folium.Marker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        popup=(
            f"<b>{row['district']}</b><br>"
            f"Week {row['week']}<br>"
            f"Cases: {row['weekly hospitalized']}<br>"
            f"Temp: {row['temp']}°C<br>"
            f"Rainfall: {row['rainfall']}mm<br>"
            f"Humidity: {row['humidity']}%"
        ),
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

folium_static(m, width=900, height=600)

# === Prophet Forecast ===

st.header("📈 Prophet Forecast")
forecast_district = st.selectbox("District for Forecast", df['district'].unique())
num_weeks = st.slider("Weeks to Forecast", 4, 24, step=4, value=12)
df_prophet = prepare_forecast_data(df, forecast_district)
forecast = train_and_forecast(df_prophet, periods=num_weeks)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Actual'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower', line=dict(dash='dot')))
st.plotly_chart(fig, use_container_width=True)

# === Poisson Forecast ===

st.header("📊 Forecast (Poisson GLM)")
district_filter = st.selectbox("District (Poisson)", df['district'].unique())
train_df = df[df['district'] == district_filter][['week', 'year', 'temp', 'rainfall', 'humidity', 'weekly hospitalized']]
train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
X = sm.add_constant(train_df[['week', 'year', 'temp', 'rainfall', 'humidity']])
y = train_df['weekly hospitalized']
poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

st.sidebar.subheader("🔮 Input for Poisson Forecast")
forecast_week = st.sidebar.selectbox("Forecast Week", sorted(df['week'].unique()))
forecast_year = st.sidebar.selectbox("Forecast Year", sorted(df['year'].unique()) + [2025])
forecast_temp = st.sidebar.slider("Temp (°C)", 20, 40, 30)
forecast_rain = st.sidebar.slider("Rainfall (mm)", 0, 300, 50)
forecast_humidity = st.sidebar.slider("Humidity (%)", 30, 100, 80)

if st.sidebar.button("Run Forecast"):
    input_df = pd.DataFrame([{
        'const': 1,
        'week': forecast_week,
        'year': forecast_year,
        'temp': forecast_temp,
        'rainfall': forecast_rain,
        'humidity': forecast_humidity
    }])
    prediction = poisson_model.predict(input_df)[0]
    st.success(f"📌 Predicted Dengue Cases in {district_filter}: {prediction:.2f}")


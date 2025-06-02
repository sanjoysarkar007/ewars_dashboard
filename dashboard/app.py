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
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Load Data ===
df = pd.read_csv("data/final_ewars_dataset.csv")

# === Title & Description ===
st.title("🦯 Dengue & Weather Early Warning Dashboard")
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
if selected_week != 'All':
    filtered_df = filtered_df[filtered_df['week'] == selected_week]

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

# === Additional District-Wise Visualizations ===
st.subheader("📊 Additional District-Wise Visuals")
viz_option = st.selectbox("Choose a visual", [
    "Top 10 Districts by Total Cases",
    f"Monthly Trend for {selected_district}",
    "District Contribution Pie Chart",
    "All Districts Case Totals (Bar Chart)"
])

if viz_option == "Top 10 Districts by Total Cases":
    top_districts = df.groupby('district')['weekly hospitalized'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_districts)

elif viz_option == f"Monthly Trend for {selected_district}":
    monthly_trend = df[df['district'] == selected_district].groupby(['year', 'week'])['weekly hospitalized'].sum().reset_index()
    fig3 = px.line(monthly_trend, x="week", y="weekly hospitalized", color="year", title=f"Weekly Trend in {selected_district}")
    st.plotly_chart(fig3)

elif viz_option == "District Contribution Pie Chart":
    contribution = df.groupby('district')['weekly hospitalized'].sum().reset_index()
    fig4 = px.pie(contribution, values='weekly hospitalized', names='district', title='District-wise Contribution to Total Cases')
    st.plotly_chart(fig4)

elif viz_option == "All Districts Case Totals (Bar Chart)":
    all_districts = df.groupby('district')['weekly hospitalized'].sum().sort_values()
    st.bar_chart(all_districts)

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

# --- Prophet Forecast Accuracy Evaluation ---
st.subheader("📊 Evaluate Prophet Forecast Accuracy")

def evaluate_prophet_model(model):
    with st.spinner("Running cross-validation... This may take a moment."):
        try:
            df_cv = cross_validation(
                model=model,
                initial='365 days',
                period='30 days',
                horizon='90 days'
            )
            df_perf = performance_metrics(df_cv)
            st.success("✅ Model evaluation completed.")
            available_cols = [col for col in ['horizon', 'mae', 'rmse', 'coverage'] if col in df_perf.columns]
            st.dataframe(df_perf[available_cols])

            st.markdown("**Legend:**")
            st.markdown("- `MAE`: Mean Absolute Error")
            st.markdown("- `RMSE`: Root Mean Squared Error")
            st.markdown("- `MAPE`: Mean Absolute Percentage Error")
            st.markdown("- `Coverage`: % of predictions within confidence interval")
        except Exception as e:
            st.error(f"❌ Error during model evaluation: {e}")

if st.button("Run Prophet Model Evaluation"):
    evaluate_prophet_model(forecast.model)

# === Poisson Forecast ===
st.header("📊 Forecast (Poisson GLM)")
district_filter = st.selectbox("District (Poisson)", df['district'].unique())
train_df = df[df['district'] == district_filter][['week', 'year', 'temp', 'rainfall', 'humidity', 'weekly hospitalized']]
train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
X = sm.add_constant(train_df[['week', 'year', 'temp', 'rainfall', 'humidity']])
y = train_df['weekly hospitalized']
poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

st.markdown("### 🔮 Forecast Input")
col1, col2, col3 = st.columns(3)

with col1:
    forecast_week = st.selectbox("Forecast Week", sorted(df['week'].unique()))
    forecast_year = st.selectbox("Forecast Year", sorted(df['year'].unique()) + [2025])

with col2:
    forecast_temp = st.slider("Temp (°C)", 20, 40, 30)
    forecast_rain = st.slider("Rainfall (mm)", 0, 300, 50)

with col3:
    forecast_humidity = st.slider("Humidity (%)", 30, 100, 80)

if st.button("Run Poisson Forecast"):
    input_df = pd.DataFrame([{
        'const': 1,
        'week': forecast_week,
        'year': forecast_year,
        'temp': forecast_temp,
        'rainfall': forecast_rain,
        'humidity': forecast_humidity
    }])

    prediction = poisson_model.predict(input_df)[0]
    st.success(f"📌 Predicted Dengue Cases in **{district_filter}**: **{int(round(prediction))}**")

# === Poisson Model Evaluation ===
st.subheader("📊 Poisson GLM Model Evaluation")
X_all = sm.add_constant(train_df[['week', 'year', 'temp', 'rainfall', 'humidity']])
y_all = train_df['weekly hospitalized']
split_index = int(0.8 * len(train_df))
X_train, X_test = X_all.iloc[:split_index], X_all.iloc[split_index:]
y_train, y_test = y_all.iloc[:split_index], y_all.iloc[split_index:]
poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
y_pred = poisson_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"📌 MAE: `{mae:.2f}`")
st.write(f"📌 RMSE: `{rmse:.2f}`")
st.write(f"📌 R² Score: `{r2:.2f}`")

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines+markers', name='Actual', line=dict(color='green')))
fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_pred, mode='lines+markers', name='Predicted', line=dict(color='red', dash='dash')))
fig.update_layout(
    title="Actual vs Predicted Dengue Cases (Poisson GLM)",
    xaxis_title="Test Sample Index",
    yaxis_title="Hospitalized Cases",
    legend=dict(orientation="h", y=-0.3),
    height=450
)
st.plotly_chart(fig, use_container_width=True)

...

# === ML Forecast: RF, XGBoost, LightGBM ===
st.header("🧠 Machine Learning Forecast Models")
ml_model = st.selectbox("Choose ML Model", ["Random Forest", "XGBoost", "LightGBM"])

# Prepare features
ml_df = df[['week', 'year', 'temp', 'rainfall', 'humidity', 'weekly hospitalized']].dropna()
X = ml_df[['week', 'year', 'temp', 'rainfall', 'humidity']]
y = ml_df['weekly hospitalized']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Train model
if ml_model == "Random Forest":
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

elif ml_model == "XGBoost":
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=100, random_state=42)

elif ml_model == "LightGBM":
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred_ml = model.predict(X_test)

# Evaluation
mae_ml = mean_absolute_error(y_test, y_pred_ml)
rmse_ml = np.sqrt(mean_squared_error(y_test, y_pred_ml))
r2_ml = r2_score(y_test, y_pred_ml)

st.subheader("📊 ML Model Evaluation Results")
st.write(f"📌 MAE: `{mae_ml:.2f}`")
st.write(f"📌 RMSE: `{rmse_ml:.2f}`")
st.write(f"📌 R² Score: `{r2_ml:.2f}`")

# Actual vs Predicted
fig_ml = go.Figure()
fig_ml.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines+markers', name='Actual', line=dict(color='green')))
fig_ml.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_pred_ml, mode='lines+markers', name='Predicted', line=dict(color='orange')))
fig_ml.update_layout(
    title=f"Actual vs Predicted Dengue Cases ({ml_model})",
    xaxis_title="Test Sample Index",
    yaxis_title="Hospitalized Cases",
    legend=dict(orientation="h", y=-0.3),
    height=450
)
st.plotly_chart(fig_ml, use_container_width=True)

...  # Existing ML Forecast Section

# === Enriched Metadata View for Test Samples ===
st.subheader("🧾 Test Sample Metadata")

# Join index with metadata (district, year, week)
meta_columns = ['district', 'year', 'week']
if all(col in df.columns for col in meta_columns):
    meta_info = df.loc[X_test.index, meta_columns].reset_index(drop=True)
    meta_info['Actual Cases'] = y_test.reset_index(drop=True)
    meta_info['Predicted Cases'] = pd.Series(y_pred_ml).round(2)

    # Add multi-filter controls with unique keys
    st.markdown("### 🔍 Filter by District, Year & Week")
    selected_meta_district = st.selectbox("Choose District", sorted(meta_info['district'].unique()), key="district_select")
    filtered_meta_info = meta_info[meta_info['district'] == selected_meta_district]

    selected_meta_year = st.selectbox("Choose Year", sorted(filtered_meta_info['year'].unique()), key="year_select")
    filtered_meta_info = filtered_meta_info[filtered_meta_info['year'] == selected_meta_year]

    selected_meta_weeks = st.multiselect("Choose Week(s)", sorted(filtered_meta_info['week'].unique()), default=sorted(filtered_meta_info['week'].unique()), key="week_select")
    filtered_meta_info = filtered_meta_info[filtered_meta_info['week'].isin(selected_meta_weeks)]

    st.dataframe(filtered_meta_info)

    # === Interactive Hover Plot (Actual vs Predicted) ===
    st.subheader("📈 Detailed Forecast Accuracy (With Metadata Hover)")
    import plotly.express as px
    filtered_meta_info['Index'] = filtered_meta_info.index
    fig_meta = px.line(filtered_meta_info, x='Index', y=['Actual Cases', 'Predicted Cases'],
                       title=f"Actual vs Predicted Dengue Cases ({ml_model}) – {selected_meta_district}, {selected_meta_year}",
                       labels={'value': 'Hospitalized Cases', 'Index': 'Test Sample Index'},
                       hover_data=filtered_meta_info[meta_columns])
    fig_meta.update_traces(mode='lines+markers')
    st.plotly_chart(fig_meta, use_container_width=True)

    # Optional: Export metadata with predictions
    csv_export = filtered_meta_info.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Predictions (CSV)", data=csv_export, file_name=f"{selected_meta_district}_{selected_meta_year}_weeks_{'_'.join(map(str, selected_meta_weeks))}_predictions.csv", mime='text/csv')

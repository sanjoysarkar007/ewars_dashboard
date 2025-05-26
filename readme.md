# Dengue EWARS Dashboard

This is an early warning and response system dashboard for Dengue, integrating weather and health data to visualize trends, outbreaks, and alerts.

## 🛠 Features

- District-wise dengue trend visualization
- Weather overlay: temperature, rainfall, humidity
- Dynamic thresholds and alerts
- Choropleth map of cases
- Easy-to-use Streamlit interface

## 📁 Structure

- `data/`: CSV and geojson files
- `pipeline/`: Data cleaning and merging scripts
- `dashboard/`: Streamlit app
- `utils/`: Configuration and helper functions

## 🚀 Run It

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py

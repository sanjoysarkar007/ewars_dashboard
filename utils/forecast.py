import pandas as pd
from prophet import Prophet

def prepare_forecast_data(df, district):
    # Filter district
    df_d = df[df["district"] == district].copy()

    # Create a date column using year + week
    df_d["date"] = pd.to_datetime(df_d["year"].astype(str) + df_d["week"].astype(str) + '1', format="%G%V%u")
    df_d = df_d.sort_values("date")

    # Prepare for Prophet
    df_prophet = df_d[["date", "weekly hospitalized"]].rename(columns={
        "date": "ds",
        "weekly hospitalized": "y"
    })

    return df_prophet

def train_and_forecast(df_prophet, periods=12):
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)

    return forecast

# utils/forecast_poisson.py

import pandas as pd
import statsmodels.api as sm

def load_and_prepare_data(filepath, district_filter=None):
    df = pd.read_csv(filepath)
    df.dropna(subset=['weekly_hospitalized'], inplace=True)
    if district_filter:
        df = df[df['district'] == district_filter]
    
    X = df[['week', 'year', 'Temp', 'Rainfall', 'humidity']]
    y = df['weekly_hospitalized']
    X = sm.add_constant(X)
    return X, y

def train_poisson_model(X, y):
    model = sm.GLM(y, X, family=sm.families.Poisson())
    results = model.fit()
    return results

def predict_cases(model, week, year, temp, rainfall, humidity):
    input_data = pd.DataFrame([{
        'const': 1,
        'week': week,
        'year': year,
        'Temp': temp,
        'Rainfall': rainfall,
        'humidity': humidity
    }])
    return model.predict(input_data)[0]

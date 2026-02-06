from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import prediction_service

app = Flask(__name__)
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ALGERIA_DATA = os.path.join(DATA_DIR, 'algeria', 'Algeria.csv')
COUNTRIES_UV_DIR = os.path.join(DATA_DIR, 'countries', 'uv')

# Cache for loaded dataframes
_data_cache = {}

def get_available_countries():
    """Get list of available countries from the data directory."""
    countries = ['Algeria']
    if os.path.exists(COUNTRIES_UV_DIR):
        for f in os.listdir(COUNTRIES_UV_DIR):
            if f.endswith('.csv'):
                countries.append(f.replace('.csv', ''))
    return sorted(list(set(countries)))

def load_country_data(country):
    """Load and cache country data."""
    if country in _data_cache:
        return _data_cache[country]
    
    if country == 'Algeria':
        path = ALGERIA_DATA
    else:
        path = os.path.join(COUNTRIES_UV_DIR, f'{country}.csv')
    
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path, parse_dates=['Date'])
        _data_cache[country] = df
        return df
    except Exception as e:
        print(f"Error loading {country}: {e}")
        return None

# Initialize models
try:
    prediction_service.load_resources(PROJECT_ROOT)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/api/countries', methods=['GET'])
def list_countries():
    """List all available countries."""
    return jsonify({"countries": get_available_countries()})

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """
    Get UV forecast data for a country.
    Uses real historical data to show recent trends and simulates a 7-day forecast.
    """
    country = request.args.get('country', 'Algeria')
    
    # Scale factor: dataset uses daily averages, multiply by 3.8 to get peak UV
    UV_SCALE_FACTOR = 3.8
    
    # Geographic baseline UV for countries with missing UV data
    GEO_BASELINES = {
        'Algeria': 9.0, 'Argentina': 7.5, 'Australia': 11.0, 'Brazil': 10.0,
        'Canada': 5.5, 'Egypt': 9.5, 'France': 6.0, 'Germany': 5.5,
        'India': 9.0, 'Indonesia': 11.0, 'Italy': 7.0, 'Japan': 6.5,
        'Mexico': 9.0, 'Morocco': 8.5, 'Spain': 8.0, 'Tunisia': 8.5,
        'United Kingdom': 5.0, 'United States': 7.0
    }
    geo_baseline = GEO_BASELINES.get(country, 7.0)
    
    df = load_country_data(country)
    if df is None:
        return jsonify({"error": f"No data available for {country}"}), 404
    
    df_sorted = df.sort_values('Date', ascending=False)
    
    uv_col = None
    for col in ['ALLSKY_SFC_UV_INDEX', 'UV_INDEX', 'uv_index']:
        if col in df.columns:
            uv_col = col
            break
    
    temp_col = 'T2M' if 'T2M' in df.columns else None
    ozone_col = 'TO3' if 'TO3' in df.columns else None
    
    today = datetime.now()
    forecast_data = []
    
    for i in range(7):
        date = today + timedelta(days=i)
        month = date.month
        
        # Default values
        raw_uv = 1.5  # Default daily average
        avg_temp = 25.0
        avg_ozone = 300.0
        
        if 'Date' in df.columns:
            similar_month = df[df['Date'].dt.month == month]
            
            if not similar_month.empty:
                # Get raw UV average from CSV (this is what the model/data gives)
                if uv_col:
                    month_uv = similar_month[uv_col].dropna()
                    if len(month_uv) > 0:
                        raw_uv = month_uv.mean()
                
                if temp_col:
                    month_temp = similar_month[temp_col].dropna()
                    if len(month_temp) > 0:
                        avg_temp = month_temp.mean()
                
                if ozone_col:
                    month_ozone = similar_month[ozone_col].dropna()
                    if len(month_ozone) > 0:
                        avg_ozone = month_ozone.mean()
        
        # Calculate peak UV (scaled from average)
        peak_uv = max(raw_uv * UV_SCALE_FACTOR, geo_baseline)
        
        # Use date-based variation (consistent per date, varies between days)
        # Use a hash-like approach for better spread
        day_hash = (date.toordinal() * 7 + date.day * 13) % 100 / 100.0
        uv_variation = (day_hash - 0.5) * 3.0  # -1.5 to +1.5 UV variation
        temp_variation = (day_hash - 0.5) * 10  # -5 to +5 degrees variation
        
        uv_peak_val = max(0.0, min(12.0, peak_uv + uv_variation))
        uv_avg_val = max(0.0, raw_uv + uv_variation * 0.2)
        temp_val = avg_temp + temp_variation
        
        forecast_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "uv_max": round(float(uv_peak_val), 2),
            "uv_avg": round(float(uv_avg_val), 2),
            "temp_c": round(float(temp_val), 1),
            "ozone": round(float(avg_ozone), 1),
            "condition": get_condition(uv_peak_val)
        })
    
    # Monthly data - both raw average and peak
    monthly_avg = []
    if 'Date' in df.columns:
        df['Month'] = df['Date'].dt.month
        for m in range(1, 13):
            month_data = df[df['Month'] == m]
            if not month_data.empty:
                raw_avg = 1.5  # default
                if uv_col:
                    month_uv = month_data[uv_col].dropna()
                    if len(month_uv) > 0:
                        raw_avg = month_uv.mean()
                
                peak_uv = max(raw_avg * UV_SCALE_FACTOR, geo_baseline * 0.8)
                
                monthly_avg.append({
                    "month": m,
                    "month_name": datetime(2000, m, 1).strftime('%b'),
                    "uv_avg": round(float(raw_avg), 2),
                    "uv_peak": round(float(peak_uv), 2)
                })
    
    # Yearly trend - historical data only
    yearly_trend = []
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        recent_years = sorted(df['Year'].unique())[-15:]  # Show last 15 years
        
        for year in recent_years:
            year_data = df[df['Year'] == year]
            if not year_data.empty:
                if uv_col:
                    year_uv = year_data[uv_col].dropna()
                    if len(year_uv) > 0:
                        raw_avg = year_uv.mean()
                        scaled_uv = raw_avg * UV_SCALE_FACTOR
                        yearly_trend.append({
                            "year": int(year),
                            "avg_uv": round(float(scaled_uv), 2)
                        })
    
    return jsonify({
        "location": country,
        "forecast": forecast_data,
        "current_uv": forecast_data[0]['uv_max'],
        "current_temp": forecast_data[0]['temp_c'],
        "risk_level": get_risk_level(forecast_data[0]['uv_max']),
        "monthly_averages": monthly_avg,
        "yearly_trend": yearly_trend
    })

def get_condition(uv_val):
    if uv_val < 3:
        return "Clear"
    elif uv_val < 6:
        return "Partly Cloudy"
    elif uv_val < 8:
        return "Sunny"
    return "Very Sunny"

def get_risk_level(uv_index):
    if uv_index < 3: return "Low"
    if uv_index < 6: return "Moderate"
    if uv_index < 8: return "High"
    if uv_index < 11: return "Very High"
    return "Extreme"

@app.route('/api/predict/uv', methods=['POST'])
def predict_uv():
    """Predict UV Index based on input features and country-specific data."""
    data = request.json
    try:
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract features from the request
        country = data.get('country', 'Algeria')
        ozone = float(data.get('ozone', 300))
        cloud_cover = float(data.get('cloud_cover', 50))
        aerosol = float(data.get('aerosol', 1.5))
        temperature = float(data.get('temperature', 25))
        hour = float(data.get('hour', 12))
        
        # Load country-specific historical data
        df = load_country_data(country)
        
        # Scale factor: dataset uses daily averages, multiply by 3.8 to get peak UV
        UV_SCALE_FACTOR = 3.8
        
        # Geographic baseline UV for countries with missing UV data
        # Based on typical peak UV at different latitudes
        GEO_BASELINES = {
            'Algeria': 9.0,      # Sahara/Mediterranean, high UV
            'Argentina': 7.5,    # South America, varied
            'Australia': 11.0,   # Very high UV
            'Brazil': 10.0,      # Tropical
            'Canada': 5.5,       # Northern, lower UV
            'China': 7.0,        # Varied
            'Egypt': 9.5,        # Desert, high UV
            'France': 6.0,       # Temperate
            'Germany': 5.5,      # Temperate
            'India': 9.0,        # Tropical/subtropical
            'Indonesia': 11.0,   # Equatorial
            'Italy': 7.0,        # Mediterranean
            'Japan': 6.5,        # Temperate
            'Kenya': 11.0,       # Equatorial
            'Mexico': 9.0,       # Subtropical
            'Morocco': 8.5,      # North Africa
            'Nigeria': 10.0,     # Tropical
            'Russia': 4.5,       # Northern
            'Saudi Arabia': 10.0, # Desert
            'South Africa': 9.0, # Subtropical
            'Spain': 8.0,        # Mediterranean
            'Tunisia': 8.5,      # North Africa
            'Turkey': 7.5,       # Mediterranean
            'United Kingdom': 5.0, # Northern
            'United States': 7.0, # Varied average
        }
        
        # Get country baseline values from historical data
        country_baseline_uv = GEO_BASELINES.get(country, 7.0)  # Default 7.0
        country_avg_ozone = 300.0
        country_avg_temp = 25.0
        
        if df is not None:
            uv_col = None
            for col in ['ALLSKY_SFC_UV_INDEX', 'UV_INDEX', 'uv_index']:
                if col in df.columns:
                    uv_col = col
                    break
            
            # Only use CSV data if it gives higher values than geographic baseline
            if uv_col and df[uv_col].notna().sum() > 100:
                raw_avg = df[uv_col].mean()
                if not np.isnan(raw_avg) and raw_avg > 0.1:
                    csv_baseline = raw_avg * UV_SCALE_FACTOR
                    # Use higher of geographic or CSV baseline
                    country_baseline_uv = max(country_baseline_uv, csv_baseline)
            
            if 'TO3' in df.columns and not df['TO3'].isna().all():
                country_avg_ozone = df['TO3'].mean()
                if np.isnan(country_avg_ozone):
                    country_avg_ozone = 300.0
            
            if 'T2M' in df.columns and not df['T2M'].isna().all():
                country_avg_temp = df['T2M'].mean()
                if np.isnan(country_avg_temp):
                    country_avg_temp = 25.0
        
        # Solar position effect (hour-based)
        solar_angle = abs(12 - hour) / 12
        solar_factor = 1 - solar_angle * 0.8
        
        # Ozone effect relative to country's typical ozone
        ozone_deviation = (country_avg_ozone - ozone) / country_avg_ozone
        ozone_factor = 1 + ozone_deviation * 0.3
        
        # Cloud cover effect
        cloud_factor = 1 - (cloud_cover / 100) * 0.7
        
        # Aerosol effect
        aerosol_factor = 1 - (aerosol / 5) * 0.2
        
        # Temperature correlation relative to country average
        temp_deviation = (temperature - country_avg_temp) / max(country_avg_temp, 10)
        temp_factor = 1 + temp_deviation * 0.1
        
        # Calculate final UV using country baseline
        uv_prediction = country_baseline_uv * solar_factor * ozone_factor * cloud_factor * aerosol_factor * temp_factor
        uv_prediction = max(0.0, min(12.0, uv_prediction))
        
        result = {
            "prediction": round(uv_prediction, 2),
            "risk_level": get_risk_level(uv_prediction),
            "country": country,
            "country_baseline": round(country_baseline_uv, 2),
            "factors": {
                "solar_position": round(solar_factor, 3),
                "ozone_effect": round(ozone_factor, 3),
                "cloud_effect": round(cloud_factor, 3),
                "aerosol_effect": round(aerosol_factor, 3),
                "temperature_effect": round(temp_factor, 3)
            },
            "status": "Success"
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

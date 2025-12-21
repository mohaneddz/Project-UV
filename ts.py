"""Time series models for UV prediction."""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from arch import arch_model
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False


# Available models
MODELS = ['SARIMA', 'ARIMA', 'ETS', 'NAIVE', 'SEASONAL_NAIVE']
if PROPHET_AVAILABLE:
    MODELS.append('PROPHET')
if GARCH_AVAILABLE:
    MODELS.append('GARCH')


def train_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """Train SARIMA model.
    
    Args:
        train: Training series
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
        
    Returns:
        Fitted model
    """
    model = SARIMAX(train, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    return model.fit(disp=False)


def train_arima(train, order=(1, 1, 1)):
    """Train ARIMA model.
    
    Args:
        train: Training series
        order: ARIMA order (p, d, q)
        
    Returns:
        Fitted model
    """
    model = ARIMA(train, order=order)
    return model.fit()


def train_ets(train, seasonal='add', seasonal_periods=7):
    """Train Exponential Smoothing (ETS) model.
    
    Args:
        train: Training series
        seasonal: 'add', 'mul', or None
        seasonal_periods: Number of periods in season
        
    Returns:
        Fitted model
    """
    model = ExponentialSmoothing(train, 
                                 seasonal=seasonal, 
                                 seasonal_periods=seasonal_periods)
    return model.fit()


def train_prophet(train, interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True):
    """Train Prophet model.
    
    Args:
        train: Training series (must have datetime index)
        interval_width: Confidence interval width (0-1)
        yearly_seasonality: Include yearly seasonality
        weekly_seasonality: Include weekly seasonality
        
    Returns:
        Fitted model
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet not installed. Install with: pip install prophet")
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    df = pd.DataFrame({
        'ds': train.index,
        'y': train.values
    })
    
    model = Prophet(
        interval_width=interval_width,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False
    )
    model.fit(df)
    return model


def train_garch(train, p=1, q=1):
    """Train GARCH-inspired model (uses ARIMA for UV prediction).
    
    Pure GARCH models are designed for volatility forecasting, not level forecasting.
    For UV prediction, we use a more robust ARIMA approach instead.
    The p and q parameters are kept for API compatibility.
    
    Args:
        train: Training series
        p: GARCH order p (not used, kept for API compatibility)
        q: GARCH order q (not used, kept for API compatibility)
        
    Returns:
        Fitted ARIMA model
    """
    if not GARCH_AVAILABLE:
        raise ImportError("GARCH not installed. Install with: pip install arch")
    
    # Use ARIMA for actual forecasting - more suitable for UV level prediction
    mean_model = ARIMA(train, order=(1, 1, 1)).fit()
    
    return mean_model


def train_naive(train):
    """Naive forecast (last value).
    
    Args:
        train: Training series
        
    Returns:
        Last value
    """
    return train.iloc[-1]


def train_seasonal_naive(train, season=7):
    """Seasonal naive forecast.
    
    Args:
        train: Training series
        season: Seasonal period
        
    Returns:
        Last seasonal values
    """
    return train.iloc[-season:]


def forecast_model(model, steps, model_type='SARIMA', train_data=None):
    """Generate forecast from fitted model.
    
    Args:
        model: Fitted model or value
        steps: Number of steps to forecast
        model_type: Type of model
        train_data: Original training data (needed for Prophet)
        
    Returns:
        Forecast array
    """
    if model_type in ['SARIMA', 'ARIMA', 'ETS']:
        return model.forecast(steps=steps)
    elif model_type == 'PROPHET':
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        # Return only the forecast values (yhat)
        return forecast['yhat'].tail(steps).values
    elif model_type == 'GARCH':
        # GARCH now uses ARIMA for forecasting (fixed variant)
        return model.forecast(steps=steps)
    elif model_type == 'NAIVE':
        return np.array([model] * steps)
    elif model_type == 'SEASONAL_NAIVE':
        # Repeat seasonal pattern
        n_repeats = int(np.ceil(steps / len(model)))
        forecast = np.tile(model.values, n_repeats)[:steps]
        return forecast
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_forecast(actual, forecast):
    """Calculate evaluation metrics.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        
    Returns:
        Dict with RMSE and MAE
    """
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    return {'RMSE': rmse, 'MAE': mae}


def train_all_variables(train_df, test_df, model_type='SARIMA', model_params=None):
    """Train model for all variables in DataFrame.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        model_type: Type of model to train
        model_params: Dict of model parameters
        
    Returns:
        Tuple of (metrics_df, predictions_dict, models_dict)
    """
    if model_params is None:
        model_params = {}
    
    columns = train_df.columns
    metrics_list = []
    predictions_dict = {}
    models_dict = {}
    
    print(f"\nTraining {model_type} models for {len(columns)} variables...")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    for i, col in enumerate(columns):
        print(f"[{i+1}/{len(columns)}] Training {model_type} for {col}...")
        
        try:
            # Train model
            if model_type == 'SARIMA':
                model = train_sarima(train_df[col], 
                                    order=model_params.get('order', (1, 1, 1)),
                                    seasonal_order=model_params.get('seasonal_order', (1, 1, 1, 7)))
            elif model_type == 'ARIMA':
                model = train_arima(train_df[col], 
                                   order=model_params.get('order', (1, 1, 1)))
            elif model_type == 'ETS':
                model = train_ets(train_df[col], 
                                 seasonal=model_params.get('seasonal', 'add'),
                                 seasonal_periods=model_params.get('seasonal_periods', 7))
            elif model_type == 'PROPHET':
                model = train_prophet(train_df[col],
                                     interval_width=model_params.get('interval_width', 0.95),
                                     yearly_seasonality=model_params.get('yearly_seasonality', True),
                                     weekly_seasonality=model_params.get('weekly_seasonality', True))
            elif model_type == 'GARCH':
                model = train_garch(train_df[col],
                                   p=model_params.get('p', 1),
                                   q=model_params.get('q', 1))
            elif model_type == 'NAIVE':
                model = train_naive(train_df[col])
            elif model_type == 'SEASONAL_NAIVE':
                model = train_seasonal_naive(train_df[col], 
                                            season=model_params.get('season', 7))
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Forecast
            forecast = forecast_model(model, len(test_df), model_type, train_df[col])
            
            # Evaluate
            metrics = evaluate_forecast(test_df[col], forecast)
            
            # Store results
            predictions_dict[col] = {
                'train': train_df[col],
                'test': test_df[col],
                'forecast': forecast
            }
            models_dict[col] = model
            
            metrics_list.append({
                'Variable': col,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE']
            })
            
        except Exception as e:
            print(f"Failed to train {model_type} for {col}: {e}")
    
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df, predictions_dict, models_dict


def save_models(models_dict, model_type, save_dir='models'):
    """Save trained models to disk.
    
    Args:
        models_dict: Dict of {variable: model}
        model_type: Type of model
        save_dir: Directory to save models
    """
    # Create model-specific subdirectory
    model_dir = os.path.join(save_dir, model_type.lower())
    os.makedirs(model_dir, exist_ok=True)
    
    for var_name, model in models_dict.items():
        filename = os.path.join(model_dir, f'{model_type}_{var_name}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"\n{len(models_dict)} models saved to {model_dir}/ folder")


def load_model(filepath):
    """Load model from disk.
    
    Args:
        filepath: Path to model file (can be full path or just filename for backward compatibility)
        
    Returns:
        Loaded model
    """
    # If filepath doesn't exist, it might be just a filename, try to find it in subdirectories
    if not os.path.exists(filepath):
        # Try to extract model_type from filename
        basename = os.path.basename(filepath)
        if '_' in basename:
            model_type = basename.split('_')[0]
            # Try the new subdirectory structure
            new_path = os.path.join('models', model_type.lower(), basename)
            if os.path.exists(new_path):
                filepath = new_path
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model


def train_and_evaluate(train_df, test_df, model_type='SARIMA', model_params=None, save_dir='models'):
    """Complete training and evaluation pipeline.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        model_type: Type of model
        model_params: Model parameters
        save_dir: Directory to save models
        
    Returns:
        Tuple of (metrics_df, predictions_dict)
    """
    # Train models
    metrics_df, predictions_dict, models_dict = train_all_variables(
        train_df, test_df, model_type, model_params
    )
    
    # Save models
    if models_dict:
        save_models(models_dict, model_type, save_dir)
    
    return metrics_df, predictions_dict


def get_default_params(model_type):
    """Get default parameters for model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dict of default parameters
    """
    defaults = {
        'SARIMA': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 7)},
        'ARIMA': {'order': (1, 1, 1)},
        'ETS': {'seasonal': 'add', 'seasonal_periods': 7},
        'PROPHET': {'interval_width': 0.95, 'yearly_seasonality': True, 'weekly_seasonality': True},
        'GARCH': {'p': 1, 'q': 1},
        'NAIVE': {},
        'SEASONAL_NAIVE': {'season': 7}
    }
    return defaults.get(model_type, {})

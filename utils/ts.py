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
MODELS = ['SARIMAX', 'SARIMA', 'ARIMA', 'ETS', 'NAIVE', 'SEASONAL_NAIVE']
if PROPHET_AVAILABLE:
    MODELS.append('PROPHET')
if GARCH_AVAILABLE:
    MODELS.append('GARCH')


def train_sarimax(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), trend='t', fourier_K=8, fourier_period=365.25, maxiter=100):
    """Train SARIMAX model with Fourier terms for yearly seasonality.
    
    Args:
        train: Training series
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s) for short-term seasonality (e.g., weekly)
        trend: Trend specification ('n', 'c', 't', 'ct')
        fourier_K: Number of Fourier term pairs for yearly seasonality
        fourier_period: Period for Fourier terms (default 365.25 for yearly)
        maxiter: Maximum iterations for optimization
        
    Returns:
        Tuple of (fitted_model, fourier_K, fourier_period) for forecasting
    """
    # Generate Fourier terms for yearly seasonality
    def fourier_terms(index, period=365.25, K=6):
        t = np.arange(len(index))
        X = {}
        for k in range(1, K + 1):
            X[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
            X[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
        return pd.DataFrame(X, index=index)
    
    X_train = fourier_terms(train.index, period=fourier_period, K=fourier_K)
    
    model = SARIMAX(
        train,
        trend=trend,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    
    fitted_model = model.fit(disp=False, maxiter=maxiter)
    
    # Return model along with Fourier parameters for forecasting
    return (fitted_model, fourier_K, fourier_period)


def train_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), maxiter=200):
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
    return model.fit(disp=False, maxiter=maxiter)


def train_arima(train, order=(1, 1, 1), auto=True, use_seasonal=False):
    """Train ARIMA model.
    
    Args:
        train: Training series
        order: ARIMA order (p, d, q) - used if auto=False
        auto: If True, tries to find better parameters automatically
        use_seasonal: If True, uses SARIMA with seasonal terms for UV data
        
    Returns:
        Fitted model
    """
    if use_seasonal:
        # For UV data which has strong seasonality, use SARIMA instead
        # UV data typically has yearly seasonality (365 days)
        # But we'll use weekly (7) and monthly (30) approximations to avoid overfitting
        seasonal_orders_to_try = [
            (1, 1, 1, 365),  # Yearly
            (1, 1, 1, 30),   # Monthly  
            (1, 1, 1, 7),    # Weekly
        ]
        
        best_aic = float('inf')
        best_model = None
        
        for seasonal_order in seasonal_orders_to_try:
            try:
                model = SARIMAX(train, order=(1,1,1), seasonal_order=seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
                fitted_model = model.fit(disp=False, maxiter=100)
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
            except:
                continue
        
        if best_model is not None:
            return best_model
    
    if auto:
        # Try different parameter combinations to find better fit
        best_aic = float('inf')
        best_model = None
        
        # Test various ARIMA orders
        orders_to_try = [
            (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (1, 1, 2),
            (2, 1, 2), (3, 1, 1), (1, 1, 3), (0, 1, 2), (2, 1, 0),
            (1, 0, 1), (2, 0, 1), (1, 0, 2)  # Some non-differenced options
        ]
        
        for order_candidate in orders_to_try:
            try:
                model = ARIMA(train, order=order_candidate)
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
            except:
                continue
        
        if best_model is None:
            # Fallback to basic model if all fail
            model = ARIMA(train, order=order)
            return model.fit()
        
        return best_model
    else:
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
    if seasonal is None:
        model = ExponentialSmoothing(train, seasonal=None)
    else:
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
        model: Fitted model or value (for SARIMAX: tuple of (model, K, period))
        steps: Number of steps to forecast
        model_type: Type of model
        train_data: Original training data (needed for Prophet and SARIMAX)
        
    Returns:
        Forecast array
    """
    if model_type == 'SARIMAX':
        # model is a tuple: (fitted_model, fourier_K, fourier_period)
        fitted_model, fourier_K, fourier_period = model
        
        # Generate Fourier terms for forecast horizon
        def fourier_terms(start_idx, steps, period=365.25, K=6):
            t = np.arange(start_idx, start_idx + steps)
            X = {}
            for k in range(1, K + 1):
                X[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
                X[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
            return pd.DataFrame(X)
        
        # Start index is the length of training data
        start_idx = len(train_data) if train_data is not None else 0
        X_forecast = fourier_terms(start_idx, steps, period=fourier_period, K=fourier_K)
        
        return fitted_model.forecast(steps=steps, exog=X_forecast)
    elif model_type in ['SARIMA', 'ARIMA', 'ETS']:
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


def _infer_seasonal_period(index, min_cycles=2, max_seasonal_period=None):
    """Infer a reasonable seasonal period from a datetime index."""
    if index is None or len(index) == 0:
        return 1

    freq = pd.infer_freq(index)
    if not freq:
        season = 7 if len(index) >= 14 else 1
    else:
        freq = freq.upper()
        if freq.startswith('D') or freq == 'B':
            season = 365
        elif freq.startswith('W'):
            season = 52
        elif freq.startswith('M'):
            season = 12
        elif freq.startswith('Q'):
            season = 4
        elif freq.startswith('A') or freq.startswith('Y'):
            season = 1
        else:
            season = 7

    if len(index) < season * min_cycles:
        season = 7 if len(index) >= 14 else 1

    if max_seasonal_period is not None and season > max_seasonal_period:
        season = max_seasonal_period

    return max(1, int(season))


def _select_arima_order(train, p_values=(0, 1, 2), d_values=(0, 1), q_values=(0, 1, 2)):
    """Select ARIMA order by AIC over a small grid."""
    best_order = (1, 1, 1)
    best_aic = np.inf

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train, order=(p, d, q))
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    return best_order


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
            seasonal_period = None
            if model_params.get('auto_seasonal', False):
                seasonal_period = _infer_seasonal_period(
                    train_df[col].index,
                    max_seasonal_period=model_params.get('max_seasonal_period')
                )

            # Train model
            if model_type == 'SARIMAX':
                seasonal_order = model_params.get('seasonal_order', (1, 0, 1, 7))
                model = train_sarimax(
                    train_df[col],
                    order=model_params.get('order', (1, 1, 1)),
                    seasonal_order=seasonal_order,
                    trend=model_params.get('trend', 't'),
                    fourier_K=model_params.get('fourier_K', 8),
                    fourier_period=model_params.get('fourier_period', 365.25),
                    maxiter=model_params.get('maxiter', 100)
                )
            elif model_type == 'SARIMA':
                seasonal_order = model_params.get('seasonal_order')
                if seasonal_period is None:
                    if seasonal_order and len(seasonal_order) == 4:
                        seasonal_period = seasonal_order[3]
                    else:
                        seasonal_period = 7
                if seasonal_order is None:
                    if seasonal_period < 2:
                        seasonal_order = (0, 0, 0, 1)
                    else:
                        seasonal_order = (1, 1, 1, seasonal_period)
                model = train_sarima(train_df[col], 
                                    order=model_params.get('order', (1, 1, 1)),
                                    seasonal_order=seasonal_order,
                                    maxiter=model_params.get('maxiter', 200))
            elif model_type == 'ARIMA':
                order = model_params.get('order', (1, 1, 1))
                if model_params.get('auto_arima', False):
                    order = _select_arima_order(
                        train_df[col],
                        p_values=model_params.get('p_values', (0, 1, 2)),
                        d_values=model_params.get('d_values', (0, 1)),
                        q_values=model_params.get('q_values', (0, 1, 2))
                    )
                model = train_arima(train_df[col], 
                                   order=order)
            elif model_type == 'ETS':
                if seasonal_period is None:
                    seasonal_period = model_params.get('seasonal_periods')
                    if seasonal_period is None:
                        seasonal_period = 7
                max_seasonal = model_params.get('max_seasonal_period')
                if max_seasonal is not None:
                    seasonal_period = min(seasonal_period, max_seasonal)
                seasonal = model_params.get('seasonal', 'add')
                if seasonal_period < 2:
                    seasonal = None
                model = train_ets(train_df[col], 
                                 seasonal=seasonal,
                                 seasonal_periods=seasonal_period)
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
                if seasonal_period is None:
                    seasonal_period = model_params.get('season')
                    if seasonal_period is None:
                        seasonal_period = 7
                max_seasonal = model_params.get('max_seasonal_period')
                if max_seasonal is not None:
                    seasonal_period = min(seasonal_period, max_seasonal)
                model = train_seasonal_naive(train_df[col], 
                                            season=seasonal_period)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Forecast
            forecast = forecast_model(model, len(test_df), model_type, train_df[col])
            forecast = np.asarray(forecast)
            
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
        'SARIMAX': {'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, 7), 'trend': 't', 'fourier_K': 8, 'fourier_period': 365.25, 'maxiter': 100},
        'SARIMA': {'order': (1, 1, 1), 'seasonal_order': None, 'auto_seasonal': True, 'max_seasonal_period': 30, 'maxiter': 200},
        'ARIMA': {'order': (1, 1, 1), 'auto_arima': True},
        'ETS': {'seasonal': 'add', 'seasonal_periods': None, 'auto_seasonal': True, 'max_seasonal_period': 30},
        'PROPHET': {'interval_width': 0.95, 'yearly_seasonality': True, 'weekly_seasonality': True},
        'GARCH': {'p': 1, 'q': 1},
        'NAIVE': {},
        'SEASONAL_NAIVE': {'season': None, 'auto_seasonal': True, 'max_seasonal_period': 30}
    }
    return defaults.get(model_type, {})

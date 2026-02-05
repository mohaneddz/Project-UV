"""Machine learning models for UV prediction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# Available models
MODELS = ['RANDOM_FOREST', 'LINEAR_REGRESSION', 'RIDGE', 'LASSO', 'SVR']
if XGBOOST_AVAILABLE:
    MODELS.append('XGBOOST')
if LIGHTGBM_AVAILABLE:
    MODELS.append('LIGHTGBM')


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        
    Returns:
        Fitted model
    """
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                 random_state=random_state, n_jobs=-1)
        )
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                     random_state=random_state, n_jobs=-1)
    return model.fit(X_train, y_train)


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
    """Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Fitted model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, random_state=random_state, n_jobs=-1)
        )
    else:
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                learning_rate=learning_rate, random_state=random_state, n_jobs=-1)
    return model.fit(X_train, y_train)


def train_lightgbm(X_train, y_train, n_estimators=100, max_depth=-1, learning_rate=0.1, random_state=42):
    """Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth (-1 = no limit)
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Fitted model
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(
            lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate, random_state=random_state, n_jobs=-1)
        )
    else:
        model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 learning_rate=learning_rate, random_state=random_state, n_jobs=-1)
    return model.fit(X_train, y_train)


def train_linear_regression(X_train, y_train):
    """Train Linear Regression model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Fitted model
    """
    # Scale features for linear models to stabilize coefficients
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    return model.fit(X_train, y_train)


def train_ridge(X_train, y_train, alpha=1.0):
    """Train Ridge Regression model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        alpha: Regularization strength
        
    Returns:
        Fitted model
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=alpha))
    ])
    return model.fit(X_train, y_train)


def train_lasso(X_train, y_train, alpha=1.0):
    """Train Lasso Regression model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        alpha: Regularization strength
        
    Returns:
        Fitted model
    """
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        lasso = MultiOutputRegressor(Lasso(alpha=alpha, max_iter=5000))
    else:
        lasso = Lasso(alpha=alpha, max_iter=5000)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', lasso)
    ])
    return model.fit(X_train, y_train)


def train_svr(X_train, y_train, kernel='rbf', C=1.0):
    """Train Support Vector Regression model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        kernel: Kernel type
        C: Regularization parameter
        
    Returns:
        Fitted model
    """
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        svr = MultiOutputRegressor(SVR(kernel=kernel, C=C))
    else:
        svr = SVR(kernel=kernel, C=C)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', svr)
    ])
    return model.fit(X_train, y_train)


def predict_model(model, X):
    """Generate predictions from model.
    
    Args:
        model: Fitted model
        X: Features
        
    Returns:
        Predictions array
    """
    return model.predict(X)


def evaluate_predictions(y_true, y_pred, target_cols):
    """Calculate evaluation metrics for each target.
    
    Args:
        y_true: True values (DataFrame or array)
        y_pred: Predicted values
        target_cols: List of target column names
        
    Returns:
        DataFrame with metrics
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    metrics_list = []
    
    for i, col in enumerate(target_cols):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        
        metrics_list.append({
            'Variable': col,
            'RMSE': rmse,
            'MAE': mae
        })
    
    return pd.DataFrame(metrics_list)


def create_predictions_dict(train_data, test_data, predictions, target_cols):
    """Create predictions dictionary for visualization.
    
    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        predictions: Predictions array
        target_cols: List of target columns
        
    Returns:
        Predictions dictionary
    """
    predictions_dict = {}
    
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    for i, col in enumerate(target_cols):
        predictions_dict[col] = {
            'train': train_data[col],
            'test': test_data[col],
            'forecast': predictions[:, i]
        }
    
    return predictions_dict


def train_and_evaluate_ml(df, df_full, test_ts, model_type='LINEAR_REGRESSION', 
                          exclude_features=None, forecast_future=True):
    """Complete ML training pipeline with forecasting.
    
    Args:
        df: DataFrame with non-null target values
        df_full: Full DataFrame including future dates
        test_ts: Test time series for sizing
        model_type: Type of ML model to train
        exclude_features: List of features to exclude (e.g., ['ALLSKY_SFC_UVA', 'ALLSKY_SFC_UVB'])
        forecast_future: Whether to forecast future dates
        
    Returns:
        Tuple of (model, metrics_dict, predictions_dict)
    """
    from utils.preprocess import prepare_for_ml
    
    # Get only columns that exist in both dataframes (excluding Date and any EDA-added columns like Year)
    common_cols = [col for col in df.columns if col in df_full.columns and col != 'Date']
    
    # Prepare ML data using only common columns
    target_cols = ['ALLSKY_SFC_UV_INDEX']
    X_train, X_test, y_train, y_test, feature_cols = prepare_for_ml(
        df.set_index('Date')[common_cols],
        target_cols,
        test_days=len(test_ts)
    )
    
    # Exclude specified features
    if exclude_features:
        fair_features = [col for col in feature_cols if not any(exc in col for exc in exclude_features)]
        X_train = X_train[fair_features]
        X_test = X_test[fair_features]
        print(f"Features: {len(fair_features)} (excluded {', '.join(exclude_features)})")
    else:
        fair_features = feature_cols
    
    # Train model based on type
    if model_type == 'LINEAR_REGRESSION':
        model = train_linear_regression(X_train, y_train)
    elif model_type == 'RANDOM_FOREST':
        model = train_random_forest(X_train, y_train)
    elif model_type == 'LIGHTGBM' and LIGHTGBM_AVAILABLE:
        model = train_lightgbm(X_train, y_train)
    elif model_type == 'XGBOOST' and XGBOOST_AVAILABLE:
        model = train_xgboost(X_train, y_train)
    elif model_type in ['CATBOOST', 'ADABOOST', 'GRADIENT_BOOSTING']:
        # Import boosters here to avoid circular dependency
        from utils.boosters import (train_catboost, train_adaboost, train_gradient_boosting,
                                     CATBOOST_AVAILABLE)
        if model_type == 'CATBOOST':
            if CATBOOST_AVAILABLE:
                model = train_catboost(X_train, y_train, verbose=False)
            else:
                print("CatBoost not available, using Linear Regression")
                model = train_linear_regression(X_train, y_train)
        elif model_type == 'ADABOOST':
            model = train_adaboost(X_train, y_train)
        elif model_type == 'GRADIENT_BOOSTING':
            model = train_gradient_boosting(X_train, y_train)
    else:
        model = train_linear_regression(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
    mae = mean_absolute_error(y_test.values, y_pred)
    metrics = {'RMSE': rmse, 'MAE': mae}
    
    print(f"{model_type} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    
    # Forecast future if requested
    future_pred = None
    future_dates = None
    if forecast_future:
        future_df = df_full[df_full['Date'] >= df['Date'].max()]
        if len(future_df) > 0:
            # Get common columns (excluding Date and any columns added later like Year)
            common_cols = [col for col in df.columns if col != 'Date' and col in df_full.columns]
            
            future_df_indexed = future_df.set_index('Date')
            X_future, _, _, _, _ = prepare_for_ml(
                pd.concat([df.set_index('Date')[common_cols], 
                          future_df_indexed[common_cols]]),
                target_cols,
                test_days=len(future_df)
            )
            X_future = X_future[fair_features][-len(future_df):]
            future_pred = model.predict(X_future)
            future_dates = future_df_indexed.index
            print(f"Forecast: {len(future_pred)} days, Range: {future_pred.min():.2f} - {future_pred.max():.2f}")
    
    # Package results
    test_dates = df[df['Date'] >= test_ts.index[0]]['Date'].values[:len(y_pred)]
    predictions = {
        'test_dates': test_dates,
        'y_test': y_test.values,
        'y_pred': y_pred,
        'future_dates': future_dates,
        'future_pred': future_pred
    }
    
    return model, metrics, predictions


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='RANDOM_FOREST', 
                      model_params=None, save_dir='models'):
    """Complete ML training and evaluation pipeline.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets (DataFrame)
        y_test: Test targets (DataFrame)
        model_type: Type of model
        model_params: Model parameters dict
        save_dir: Directory to save model
        
    Returns:
        Tuple of (metrics_df, predictions_dict, model)
    """
    if model_params is None:
        model_params = {}
    
    target_cols = y_train.columns.tolist() if isinstance(y_train, pd.DataFrame) else ['target']
    
    print(f"\nTraining {model_type} model...")
    print(f"Features: {X_train.shape[1]}, Targets: {len(target_cols)}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train model
    try:
        if model_type == 'RANDOM_FOREST':
            model = train_random_forest(X_train, y_train, **model_params)
        elif model_type == 'XGBOOST':
            model = train_xgboost(X_train, y_train, **model_params)
        elif model_type == 'LIGHTGBM':
            model = train_lightgbm(X_train, y_train, **model_params)
        elif model_type == 'LINEAR_REGRESSION':
            model = train_linear_regression(X_train, y_train)
        elif model_type == 'RIDGE':
            model = train_ridge(X_train, y_train, **model_params)
        elif model_type == 'LASSO':
            model = train_lasso(X_train, y_train, **model_params)
        elif model_type == 'SVR':
            model = train_svr(X_train, y_train, **model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"{model_type} training complete!")
        
        # Predict
        y_pred = predict_model(model, X_test)
        
        # Evaluate
        metrics_df = evaluate_predictions(y_test, y_pred, target_cols)
        
        # Create predictions dict for visualization
        # Need to align with original data for plotting
        test_index = y_test.index if isinstance(y_test, pd.DataFrame) else range(len(y_test))
        test_df = pd.DataFrame(y_test.values if isinstance(y_test, pd.DataFrame) else y_test, 
                              columns=target_cols, index=test_index)
        
        predictions_dict = {}
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        for i, col in enumerate(target_cols):
            predictions_dict[col] = {
                'train': pd.Series(dtype=float),  # ML models don't have train series
                'test': test_df[col],
                'forecast': y_pred[:, i]
            }
        
        # Save model
        model_dir = os.path.join(save_dir, model_type.lower())
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{model_type}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
        
        return metrics_df, predictions_dict, model
        
    except Exception as e:
        print(f"Error training {model_type}: {e}")
        raise


def get_feature_importance(model, feature_names, model_type):
    """Extract feature importance from model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_type: Type of model
        
    Returns:
        Array of feature importances (or None)
    """
    try:
        # Unwrap Pipeline if present
        base_model = model
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            base_model = model.named_steps['model']

        if hasattr(base_model, 'feature_importances_'):
            return base_model.feature_importances_
        elif hasattr(base_model, 'estimators_'):  # MultiOutputRegressor
            if hasattr(base_model.estimators_[0], 'feature_importances_'):
                return base_model.estimators_[0].feature_importances_
        elif hasattr(base_model, 'coef_'):
            return np.abs(base_model.coef_).mean(axis=0) if base_model.coef_.ndim > 1 else np.abs(base_model.coef_)
    except Exception:
        pass
    return None


def get_default_params(model_type):
    """Get default parameters for model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dict of default parameters
    """
    defaults = {
        'RANDOM_FOREST': {'n_estimators': 300, 'max_depth': None, 'random_state': 42},
        'XGBOOST': {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'random_state': 42},
        'LIGHTGBM': {'n_estimators': 300, 'max_depth': -1, 'learning_rate': 0.05, 'random_state': 42},
        'LINEAR_REGRESSION': {},
        'RIDGE': {'alpha': 1.0},
        'LASSO': {'alpha': 0.1},
        'SVR': {'kernel': 'rbf', 'C': 10.0}
    }
    return defaults.get(model_type, {})


def load_model(model_type, save_dir='models'):
    """Load a saved ML model from disk.
    
    Args:
        model_type: Type of model (e.g., 'RANDOM_FOREST', 'XGBOOST')
        save_dir: Base directory where models are saved
        
    Returns:
        Loaded model
    """
    model_dir = os.path.join(save_dir, model_type.lower())
    model_path = os.path.join(model_dir, f'{model_type}_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    return model


# ==========================================
# UV Risk Classification Logic (Consolidated)
# ==========================================

RISK_CATEGORIES = {
    0: 'Low',
    1: 'Moderate', 
    2: 'High',
    3: 'Very High',
    4: 'Extreme'
}

def get_risk_category(uv_index):
    """Map UV Index to WHO Risk Category.
    
    Args:
        uv_index: float or int
        
    Returns:
        int: Risk category code (0-4)
    """
    if uv_index < 3:
        return 0  # Low (0-2)
    elif uv_index < 6:
        return 1  # Moderate (3-5)
    elif uv_index < 8:
        return 2  # High (6-7)
    elif uv_index < 11:
        return 3  # Very High (8-10)
    else:
        return 4  # Extreme (11+)

def get_protection_recommendation(risk_code):
    """Get protection recommendations based on risk code.
    
    Args:
        risk_code: int
        
    Returns:
        str: Recommendation text
    """
    recommendations = {
        0: "No protection required. You can safely stay outside.",
        1: "Protection required. Seek shade during midday hours! Slip on a shirt, slop on sunscreen and slap on a hat.",
        2: "Protection essential. Reduce time in the sun between 10am and 4pm. Wear sunglassess, hat, and sunscreen.",
        3: "Extra protection needed. Be careful! Unprotected skin can be damaged and can burn quickly.",
        4: "Take all precautions. Unprotected skin can burn in minutes. Avoid being outside during midday hours."
    }
    return recommendations.get(risk_code, "Consult local health guidelines.")

def train_risk_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """Train a generic Random Forest Classifier for Risk Prediction.
    
    Args:
        X_train: Training features
        y_train: Training targets (Risk Categories)
        n_estimators: Number of trees
        random_state: Random seed
        
    Returns:
        Fitted model
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    return model

def evaluate_risk_model(model, X_test, y_test):
    """Evaluate classifier and print classification report.
    
    Args:
        model: Fitted model
        X_test: Test features
        y_test: True risk categories
        
    Returns:
        dict: Classification report
    """
    y_pred = model.predict(X_test)
    
    print("\\nClassification Report:")
    # Get unique labels from truth and pred to avoid errors if some classes are missing
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    target_names = [RISK_CATEGORIES.get(l, str(l)) for l in unique_labels]
    
    # Check if target_names matches unique_labels length
    if len(target_names) == len(unique_labels):
        print(classification_report(y_test, y_pred, target_names=target_names))
    else:
        print(classification_report(y_test, y_pred))
    
    print("\\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return classification_report(y_test, y_pred, output_dict=True)

"""Machine learning models for UV prediction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
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
    model = LinearRegression()
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
    model = Ridge(alpha=alpha)
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
    model = Lasso(alpha=alpha, max_iter=5000)
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
        model = MultiOutputRegressor(SVR(kernel=kernel, C=C))
    else:
        model = SVR(kernel=kernel, C=C)
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
                'train': pd.Series([]),  # ML models don't have train series
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
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'estimators_'):  # MultiOutputRegressor
            if hasattr(model.estimators_[0], 'feature_importances_'):
                return model.estimators_[0].feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    except:
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
        'RANDOM_FOREST': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
        'XGBOOST': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
        'LIGHTGBM': {'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1, 'random_state': 42},
        'LINEAR_REGRESSION': {},
        'RIDGE': {'alpha': 1.0},
        'LASSO': {'alpha': 1.0},
        'SVR': {'kernel': 'rbf', 'C': 1.0}
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


"""Boosting models for UV prediction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# Available models
MODELS = ['ADABOOST', 'GRADIENT_BOOSTING', 'HIST_GRADIENT_BOOSTING', 'EXTRA_TREES']
if CATBOOST_AVAILABLE:
    MODELS.append('CATBOOST')
if XGBOOST_AVAILABLE:
    MODELS.append('XGBOOST')


def train_catboost(X_train, y_train, iterations=100, depth=6, learning_rate=0.1, verbose=False):
    """Train CatBoost model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        iterations: Number of boosting iterations
        depth: Tree depth
        learning_rate: Learning rate
        verbose: Verbosity level
        
    Returns:
        Fitted model
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost not installed. Install with: pip install catboost")
    
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        # Multi-target regression
        model = cb.CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function='MultiRMSE',
            verbose=verbose
        )
    else:
        model = cb.CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            verbose=verbose
        )
    
    return model.fit(X_train, y_train)


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
    """Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)
        random_state: Random seed
        
    Returns:
        Fitted model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        # Multi-target regression
        model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                verbosity=0
            )
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=0
        )
    
    return model.fit(X_train, y_train)

def train_xgboost_ml(df, df_full, test_ts, exclude_features=None):
    """Train XGBoost with complete ML pipeline.
    
    Args:
        df: DataFrame with non-null target values
        df_full: Full DataFrame including future dates
        test_ts: Test time series for sizing
        exclude_features: List of features to exclude
        
    Returns:
        Tuple of (model, metrics_dict, predictions_dict)
    """
    from utils.ml import train_and_evaluate_ml
    return train_and_evaluate_ml(df, df_full, test_ts, 'XGBOOST', exclude_features, forecast_future=True)


def train_catboost_ml(df, df_full, test_ts, exclude_features=None):
    """Train CatBoost with complete ML pipeline.
    
    Args:
        df: DataFrame with non-null target values
        df_full: Full DataFrame including future dates
        test_ts: Test time series for sizing
        exclude_features: List of features to exclude
        
    Returns:
        Tuple of (model, metrics_dict, predictions_dict)
    """
    from utils.ml import train_and_evaluate_ml
    return train_and_evaluate_ml(df, df_full, test_ts, 'CATBOOST', exclude_features, forecast_future=True)
def train_adaboost_ml(df, df_full, test_ts, exclude_features=None):
    """Train AdaBoost with complete ML pipeline.
    
    Args:
        df: DataFrame with non-null target values
        df_full: Full DataFrame including future dates
        test_ts: Test time series for sizing
        exclude_features: List of features to exclude
        
    Returns:
        Tuple of (model, metrics_dict, predictions_dict)
    """
    from utils.ml import train_and_evaluate_ml
    return train_and_evaluate_ml(df, df_full, test_ts, 'ADABOOST', exclude_features, forecast_future=True)


def train_gradient_boosting_ml(df, df_full, test_ts, exclude_features=None):
    """Train Gradient Boosting with complete ML pipeline.
    
    Args:
        df: DataFrame with non-null target values
        df_full: Full DataFrame including future dates
        test_ts: Test time series for sizing
        exclude_features: List of features to exclude
        
    Returns:
        Tuple of (model, metrics_dict, predictions_dict)
    """
    from utils.ml import train_and_evaluate_ml
    return train_and_evaluate_ml(df, df_full, test_ts, 'GRADIENT_BOOSTING', exclude_features, forecast_future=True)

def train_adaboost(X_train, y_train, n_estimators=50, learning_rate=1.0, random_state=42):
    """Train AdaBoost model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of estimators
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Fitted model
    """
    base_estimator = DecisionTreeRegressor(max_depth=4, random_state=random_state)
    
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(
            AdaBoostRegressor(
                estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )
        )
    else:
        model = AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
    
    return model.fit(X_train, y_train)


def train_gradient_boosting(X_train, y_train, n_estimators=100, max_depth=3, 
                           learning_rate=0.1, random_state=42):
    """Train Gradient Boosting model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of estimators
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Fitted model
    """
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
    
    return model.fit(X_train, y_train)


def train_hist_gradient_boosting(X_train, y_train, max_iter=100, max_depth=None,
                                 learning_rate=0.1, random_state=42):
    """Train HistGradientBoosting model (faster sklearn native gradient boosting).
    
    Args:
        X_train: Training features
        y_train: Training targets
        max_iter: Maximum number of iterations
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Fitted model
    """
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
    
    return model.fit(X_train, y_train)


def train_extra_trees(X_train, y_train, n_estimators=100, max_depth=None,
                     min_samples_split=2, random_state=42):
    """Train Extra Trees model (ensemble of randomized decision trees).
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        random_state: Random seed
        
    Returns:
        Fitted model
    """
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1  # Use all cores
    )
    
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


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='CATBOOST',
                      model_params=None, save_dir='models'):
    """Complete boosting training and evaluation pipeline.
    
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
        if model_type == 'CATBOOST':
            model = train_catboost(X_train, y_train, **model_params)
        elif model_type == 'XGBOOST':
            model = train_xgboost(X_train, y_train, **model_params)
        elif model_type == 'ADABOOST':
            model = train_adaboost(X_train, y_train, **model_params)
        elif model_type == 'GRADIENT_BOOSTING':
            model = train_gradient_boosting(X_train, y_train, **model_params)
        elif model_type == 'HIST_GRADIENT_BOOSTING':
            model = train_hist_gradient_boosting(X_train, y_train, **model_params)
        elif model_type == 'EXTRA_TREES':
            model = train_extra_trees(X_train, y_train, **model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"{model_type} training complete!")
        
        # Predict
        y_pred = predict_model(model, X_test)
        
        # Evaluate
        metrics_df = evaluate_predictions(y_test, y_pred, target_cols)
        
        # Create predictions dict for visualization
        test_index = y_test.index if isinstance(y_test, pd.DataFrame) else range(len(y_test))
        test_df = pd.DataFrame(y_test.values if isinstance(y_test, pd.DataFrame) else y_test,
                              columns=target_cols, index=test_index)
        
        predictions_dict = {}
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        for i, col in enumerate(target_cols):
            predictions_dict[col] = {
                'train': pd.Series(dtype=float),  # Boosting models don't need train series for viz
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
        'CATBOOST': {'iterations': 300, 'depth': 6, 'learning_rate': 0.05, 'verbose': False},
        'XGBOOST': {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'random_state': 42},
        'ADABOOST': {'n_estimators': 200, 'learning_rate': 0.1, 'random_state': 42},
        'GRADIENT_BOOSTING': {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42},
        'HIST_GRADIENT_BOOSTING': {'max_iter': 300, 'max_depth': None, 'learning_rate': 0.05, 'random_state': 42},
        'EXTRA_TREES': {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 2, 'random_state': 42}
    }
    return defaults.get(model_type, {})


def load_model(model_type, save_dir='models'):
    """Load a saved boosting model from disk.
    
    Args:
        model_type: Type of model (e.g., 'CATBOOST', 'ADABOOST')
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


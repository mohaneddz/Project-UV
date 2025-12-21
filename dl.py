"""Deep learning models for UV prediction."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# Available models
MODELS = ['LSTM', 'GRU', 'BILSTM', 'DEEP_LSTM', 'DEEP_GRU'] if TF_AVAILABLE else []


def create_sequences(data, lookback=30):
    """Create sequences for deep learning.
    
    Args:
        data: Input array
        lookback: Number of timesteps
        
    Returns:
        Tuple of (X, y) sequences
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm(input_shape, output_dim, units=50, dropout=0.2):
    """Build LSTM model.
    
    Args:
        input_shape: Tuple of (timesteps, features)
        output_dim: Number of output features
        units: Number of LSTM units
        dropout: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru(input_shape, output_dim, units=50, dropout=0.2):
    """Build GRU model.
    
    Args:
        input_shape: Tuple of (timesteps, features)
        output_dim: Number of output features
        units: Number of GRU units
        dropout: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    model = Sequential([
        GRU(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_bilstm(input_shape, output_dim, units=50, dropout=0.2):
    """Build Bidirectional LSTM model.
    
    Args:
        input_shape: Tuple of (timesteps, features)
        output_dim: Number of output features
        units: Number of LSTM units
        dropout: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    model = Sequential([
        Bidirectional(LSTM(units), input_shape=input_shape),
        Dropout(dropout),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_deep_lstm(input_shape, output_dim, units=50, dropout=0.2):
    """Build deep LSTM model with multiple layers.
    
    Args:
        input_shape: Tuple of (timesteps, features)
        output_dim: Number of output features
        units: Number of LSTM units
        dropout: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_deep_gru(input_shape, output_dim, units=50, dropout=0.2):
    """Build deep GRU model with multiple layers.
    
    Args:
        input_shape: Tuple of (timesteps, features)
        output_dim: Number of output features
        units: Number of GRU units
        dropout: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        GRU(units // 2),
        Dropout(dropout),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def prepare_dl_data(train_df, test_df, lookback=30, scale=True):
    """Prepare data for deep learning models.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        lookback: Number of timesteps
        scale: Whether to scale data
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, scaler)
    """
    # Convert to numpy
    train_data = train_df.values
    test_data = test_df.values
    
    # Scale if needed
    scaler = None
    if scale:
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
    
    # Create sequences from training data
    X_train, y_train = create_sequences(train_data, lookback)
    
    # For test data, we need to include last lookback points from train
    combined_data = np.vstack([train_data[-lookback:], test_data])
    X_test, y_test = create_sequences(combined_data, lookback)
    
    return X_train, y_train, X_test, y_test, scaler


def train_dl_model(X_train, y_train, model_type='LSTM', model_params=None, 
                   epochs=50, batch_size=32, validation_split=0.1, verbose=0):
    """Train deep learning model.
    
    Args:
        X_train: Training sequences
        y_train: Training targets
        model_type: Type of model
        model_params: Model parameters dict
        epochs: Number of epochs
        batch_size: Batch size
        validation_split: Validation split ratio
        verbose: Verbosity level
        
    Returns:
        Tuple of (model, history)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    if model_params is None:
        model_params = {}
    
    # Get input/output dimensions
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
    
    # Build model
    if model_type == 'LSTM':
        model = build_lstm(input_shape, output_dim, **model_params)
    elif model_type == 'GRU':
        model = build_gru(input_shape, output_dim, **model_params)
    elif model_type == 'BILSTM':
        model = build_bilstm(input_shape, output_dim, **model_params)
    elif model_type == 'DEEP_LSTM':
        model = build_deep_lstm(input_shape, output_dim, **model_params)
    elif model_type == 'DEEP_GRU':
        model = build_deep_gru(input_shape, output_dim, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=verbose
    )
    
    return model, history


def predict_dl_model(model, X):
    """Generate predictions from DL model.
    
    Args:
        model: Trained model
        X: Input sequences
        
    Returns:
        Predictions array
    """
    return model.predict(X, verbose=0)


def inverse_scale_predictions(predictions, scaler, target_indices=None):
    """Inverse transform scaled predictions.
    
    Args:
        predictions: Scaled predictions
        scaler: Fitted scaler
        target_indices: Indices of target columns (None = all)
        
    Returns:
        Unscaled predictions
    """
    if scaler is None:
        return predictions
    
    # Create dummy array for inverse transform
    n_features = scaler.scale_.shape[0]
    dummy = np.zeros((predictions.shape[0], n_features))
    
    if target_indices is None:
        target_indices = list(range(predictions.shape[1]))
    
    # Fill in predictions at correct indices
    for i, idx in enumerate(target_indices):
        dummy[:, idx] = predictions[:, i]
    
    # Inverse transform
    unscaled = scaler.inverse_transform(dummy)
    
    # Extract target columns
    return unscaled[:, target_indices]


def evaluate_predictions(y_true, y_pred, target_cols):
    """Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        target_cols: List of target column names
        
    Returns:
        DataFrame with metrics
    """
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


def train_and_evaluate(train_df, test_df, model_type='LSTM', model_params=None,
                      lookback=30, epochs=50, batch_size=32, save_dir='models'):
    """Complete DL training and evaluation pipeline.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        model_type: Type of model
        model_params: Model parameters dict
        lookback: Number of timesteps
        epochs: Number of epochs
        batch_size: Batch size
        save_dir: Directory to save model
        
    Returns:
        Tuple of (metrics_df, predictions_dict, model, history)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    if model_params is None:
        model_params = {}
    
    target_cols = train_df.columns.tolist()
    
    print(f"\nTraining {model_type} model...")
    print(f"Targets: {len(target_cols)}, Lookback: {lookback}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_dl_data(train_df, test_df, lookback, scale=True)
    
    print(f"Sequences - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train model
    model, history = train_dl_model(
        X_train, y_train, 
        model_type=model_type, 
        model_params=model_params,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    print(f"{model_type} training complete!")
    
    # Predict
    y_pred_scaled = predict_dl_model(model, X_test)
    
    # Inverse scale
    target_indices = list(range(len(target_cols)))
    y_pred = inverse_scale_predictions(y_pred_scaled, scaler, target_indices)
    y_test_unscaled = inverse_scale_predictions(y_test, scaler, target_indices)
    
    # Evaluate
    metrics_df = evaluate_predictions(y_test_unscaled, y_pred, target_cols)
    
    # Create predictions dict for visualization
    predictions_dict = {}
    
    for i, col in enumerate(target_cols):
        predictions_dict[col] = {
            'train': train_df[col],
            'test': test_df[col],
            'forecast': y_pred[:, i]
        }
    
    # Save model
    model_dir = os.path.join(save_dir, model_type.lower())
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_type}_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, f'{model_type}_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    return metrics_df, predictions_dict, model, history


def get_default_params(model_type):
    """Get default parameters for model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dict of default parameters
    """
    defaults = {
        'LSTM': {'units': 50, 'dropout': 0.2},
        'GRU': {'units': 50, 'dropout': 0.2},
        'BILSTM': {'units': 50, 'dropout': 0.2},
        'DEEP_LSTM': {'units': 50, 'dropout': 0.2},
        'DEEP_GRU': {'units': 50, 'dropout': 0.2}
    }
    return defaults.get(model_type, {})


def load_model(model_type, save_dir='models'):
    """Load a saved deep learning model and scaler from disk.
    
    Args:
        model_type: Type of model (e.g., 'LSTM', 'GRU')
        save_dir: Base directory where models are saved
        
    Returns:
        Tuple of (model, scaler)
    """
    from tensorflow import keras
    
    model_dir = os.path.join(save_dir, model_type.lower())
    model_path = os.path.join(model_dir, f'{model_type}_model.h5')
    scaler_path = os.path.join(model_dir, f'{model_type}_scaler.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    # Load the Keras model
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {scaler_path}")
    
    return model, scaler


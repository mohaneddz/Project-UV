"""Transfer Learning models for UV prediction."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import (
        Dense, Dropout, LayerNormalization, MultiHeadAttention,
        GlobalAveragePooling1D, Input, Layer, Reshape
    )
    TST_AVAILABLE = True
except ImportError:
    TST_AVAILABLE = False


# Available models
MODELS = []
if TABNET_AVAILABLE:
    MODELS.append('TABNET')
if TABPFN_AVAILABLE:
    MODELS.append('TABPFN')
if TST_AVAILABLE:
    MODELS.append('TST')
    MODELS.append('PATCHTST')


def train_tabnet(X_train, y_train, X_valid=None, y_valid=None, 
                 n_d=8, n_a=8, n_steps=3, gamma=1.3, 
                 max_epochs=100, patience=10, batch_size=256, 
                 verbose=0, random_state=42):
    """Train TabNet model.
    
    TabNet is an interpretable deep learning model for tabular data.
    
    Args:
        X_train: Training features (numpy array or pandas DataFrame)
        y_train: Training targets
        X_valid: Validation features (optional)
        y_valid: Validation targets (optional)
        n_d: Width of the decision prediction layer
        n_a: Width of the attention embedding
        n_steps: Number of sequential attention steps
        gamma: Relaxation parameter for feature reusage
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        batch_size: Batch size
        verbose: Verbosity level
        random_state: Random seed
        
    Returns:
        Fitted TabNet model
    """
    if not TABNET_AVAILABLE:
        raise ImportError("TabNet not installed. Install with: pip install pytorch-tabnet")
    
    # Ensure targets are 2D
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_valid is not None and y_valid.ndim == 1:
        y_valid = y_valid.reshape(-1, 1)
    
    model = TabNetRegressor(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        cat_idxs=[],  # No categorical features for UV data
        cat_dims=[],
        cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type='entmax',
        seed=random_state,
        verbose=verbose
    )
    
    # Train the model
    if X_valid is not None and y_valid is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['rmse'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size
        )
    else:
        model.fit(
            X_train, y_train,
            max_epochs=max_epochs,
            batch_size=batch_size
        )
    
    return model


def train_tabpfn(X_train, y_train, device='cpu'):
    """Train TabPFN model.
    
    TabPFN is a prior-fitted network that requires no training.
    It uses pre-trained weights for zero-shot predictions on tabular data.
    
    Args:
        X_train: Training features (for context, not actual training)
        y_train: Training targets
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        TabPFN model ready for prediction
    """
    if not TABPFN_AVAILABLE:
        raise ImportError("TabPFN not installed. Install with: pip install tabpfn")
    
    # TabPFN has limitations on dataset size
    if X_train.shape[0] > 1000:
        print("Warning: TabPFN works best with <=1000 samples. Using first 1000 samples.")
        X_train = X_train[:1000]
        y_train = y_train[:1000]
    
    if X_train.shape[1] > 100:
        print("Warning: TabPFN works best with <=100 features. Consider feature selection.")
    
    model = TabPFNRegressor(device=device, N_ensemble_configurations=4)
    model.fit(X_train, y_train)
    
    return model


def create_sequences_tst(data, lookback=30):
    """Create sequences for TST models.
    
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


def build_tst_model(input_shape, output_dim, d_model=64, num_heads=4, 
                    num_layers=2, dropout=0.1, ff_dim=128):
    """Build Time Series Transformer (TST) model.
    
    A transformer-based architecture specifically designed for time series forecasting.
    
    Args:
        input_shape: Tuple of (timesteps, features)
        output_dim: Number of output features
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        dropout: Dropout rate
        ff_dim: Feed-forward network dimension
        
    Returns:
        Compiled Keras model
    """
    if not TST_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Project input to d_model dimensions
    x = Dense(d_model)(x)
    
    # Positional encoding (learnable)
    pos_encoding = Dense(d_model)(x)
    x = x + pos_encoding
    
    # Stack transformer blocks
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads,
            dropout=dropout
        )(x, x)
        attn_output = Dropout(dropout)(attn_output)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ff_output = Dense(ff_dim, activation='relu')(x)
        ff_output = Dropout(dropout)(ff_output)
        ff_output = Dense(d_model)(ff_output)
        ff_output = Dropout(dropout)(ff_output)
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    # Global pooling and output
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(output_dim)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_tst(X_train, y_train, X_val=None, y_val=None,
              d_model=64, num_heads=4, num_layers=2, 
              dropout=0.1, ff_dim=128, epochs=100, 
              batch_size=32, patience=15, verbose=1):
    """Train Time Series Transformer model.
    
    Args:
        X_train: Training sequences (3D array: samples, timesteps, features)
        y_train: Training targets
        X_val: Validation sequences (optional)
        y_val: Validation targets (optional)
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        ff_dim: Feed-forward dimension
        epochs: Maximum training epochs
        batch_size: Batch size
        patience: Early stopping patience
        verbose: Verbosity level
        
    Returns:
        Tuple of (model, history)
    """
    if not TST_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    # Get dimensions
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
    
    # Ensure y is 2D
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_val is not None and y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)
    
    # Build model
    model = build_tst_model(
        input_shape, output_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        ff_dim=ff_dim
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train
    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
    
    return model, history


def train_patchtst(train_series, test_series=None, patch_len=16, stride=8, 
                   d_model=64, num_heads=4, num_layers=2, dropout=0.1, 
                   epochs=100, batch_size=32, lookback=96, horizon=24):
    """Train PatchTST model.
    
    PatchTST divides time series into patches and applies transformer attention.
    This is a simplified implementation using the TST architecture with preprocessing.
    
    Args:
        train_series: Training time series (1D or 2D array)
        test_series: Test time series for validation (optional)
        patch_len: Length of each patch
        stride: Stride between patches
        d_model: Dimension of model
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        epochs: Training epochs
        batch_size: Batch size
        lookback: Input sequence length
        horizon: Forecast horizon
        
    Returns:
        Tuple of (model, history, scaler)
    """
    if not TST_AVAILABLE:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    if train_series.ndim == 1:
        train_series = train_series.reshape(-1, 1)
    
    # Scale data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_series)
    
    # Create sequences
    X_train, y_train = create_sequences_tst(train_scaled, lookback)
    
    # Handle validation data
    X_val, y_val = None, None
    if test_series is not None:
        if test_series.ndim == 1:
            test_series = test_series.reshape(-1, 1)
        test_scaled = scaler.transform(test_series)
        
        # Combine last lookback from train with test for proper sequences
        combined = np.vstack([train_scaled[-lookback:], test_scaled])
        X_val, y_val = create_sequences_tst(combined, lookback)
    
    # Train TST model
    model, history = train_tst(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return model, history, scaler


def predict_tabnet(model, X_test):
    """Generate predictions from TabNet model.
    
    Args:
        model: Trained TabNet model
        X_test: Test features
        
    Returns:
        Predictions array
    """
    predictions = model.predict(X_test)
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    return predictions


def predict_tabpfn(model, X_test):
    """Generate predictions from TabPFN model.
    
    Args:
        model: Trained TabPFN model
        X_test: Test features
        
    Returns:
        Predictions array
    """
    return model.predict(X_test)


def predict_tst(model, X_test):
    """Generate predictions from TST/PatchTST model.
    
    Args:
        model: Trained TST model
        X_test: Test sequences (3D array)
        
    Returns:
        Predictions array
    """
    predictions = model.predict(X_test)
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    return predictions


def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict with RMSE and MAE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}


def train_all_models(X_train, y_train, X_test, y_test, 
                     X_valid=None, y_valid=None, 
                     models_to_train=None):
    """Train all available transfer learning models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        X_valid: Validation features (optional)
        y_valid: Validation targets (optional)
        models_to_train: List of model names to train (default: all available)
        
    Returns:
        Dict with results for each model
    """
    if models_to_train is None:
        models_to_train = MODELS
    
    results = {}
    
    for model_name in models_to_train:
        if model_name not in MODELS:
            print(f"Model {model_name} not available. Skipping.")
            continue
        
        print(f"\nTraining {model_name}...")
        
        try:
            if model_name == 'TABNET':
                model = train_tabnet(X_train, y_train, X_valid, y_valid)
                y_pred = predict_tabnet(model, X_test)
            elif model_name == 'TABPFN':
                model = train_tabpfn(X_train, y_train)
                y_pred = predict_tabpfn(model, X_test)
            elif model_name == 'TST':
                # TST expects 3D input (samples, timesteps, features)
                # Reshape if needed
                if X_train.ndim == 2:
                    print("Warning: TST expects 3D input. Consider using sequence data.")
                    continue
                model, history = train_tst(X_train, y_train, X_valid, y_valid)
                y_pred = predict_tst(model, X_test)
            elif model_name == 'PATCHTST':
                print(f"{model_name} requires time series data. Use train_patchtst directly.")
                continue
            else:
                print(f"Unknown model: {model_name}. Skipping.")
                continue
            
            metrics = evaluate_model(y_test, y_pred)
            
            results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE']
            }
            
            print(f"{model_name} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    return results


def save_model(model, filepath, model_type='TABNET'):
    """Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save model
        model_type: Type of model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if model_type == 'TABNET':
        # TabNet has its own save method
        model.save_model(filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath, model_type='TABNET'):
    """Load model from disk.
    
    Args:
        filepath: Path to model file
        model_type: Type of model
        
    Returns:
        Loaded model
    """
    if model_type == 'TABNET':
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet not installed. Install with: pip install pytorch-tabnet")
        model = TabNetRegressor()
        model.load_model(filepath)
    else:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model


def get_default_params(model_type):
    """Get default parameters for model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dict of default parameters
    """
    defaults = {
        'TABNET': {
            'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3,
            'max_epochs': 100, 'patience': 10, 'batch_size': 256
        },
        'TABPFN': {
            'device': 'cpu'
        },
        'TST': {
            'd_model': 64, 'num_heads': 4, 'num_layers': 2, 
            'dropout': 0.1, 'ff_dim': 128, 'epochs': 100, 'batch_size': 32
        },
        'PATCHTST': {
            'patch_len': 16, 'stride': 8, 'd_model': 64,
            'num_heads': 4, 'num_layers': 2, 'dropout': 0.1,
            'lookback': 96, 'horizon': 24, 'epochs': 100
        }
    }
    return defaults.get(model_type, {})

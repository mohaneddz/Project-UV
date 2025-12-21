"""Data preprocessing utilities for UV time series prediction."""

import pandas as pd
import numpy as np


def load_data(filepath, date_col='YYYYMMDD', date_format='%Y%m%d'):
    """Load CSV data and convert date column to datetime index.
    
    Args:
        filepath: Path to CSV file
        date_col: Name of date column
        date_format: Date format string
        
    Returns:
        DataFrame with datetime index
    """
    df = pd.read_csv(filepath, dtype=str)
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df.set_index(date_col, inplace=True)
    return df


def convert_to_numeric(df):
    """Convert all columns to numeric type.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with numeric columns
    """
    df_numeric = df.copy()
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    return df_numeric


def handle_missing_values(df, method='interpolate', fill_value=None):
    """Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        method: 'interpolate', 'ffill', 'bfill', 'drop', or 'fill'
        fill_value: Value to use if method='fill'
        
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    if method == 'interpolate':
        df_clean = df_clean.interpolate(method='linear', limit_direction='both')
    elif method == 'ffill':
        df_clean = df_clean.fillna(method='ffill')
    elif method == 'bfill':
        df_clean = df_clean.fillna(method='bfill')
    elif method == 'drop':
        df_clean = df_clean.dropna()
    elif method == 'fill' and fill_value is not None:
        df_clean = df_clean.fillna(fill_value)
    
    return df_clean


def create_lag_features(df, columns, lags=[1, 7, 14, 30]):
    """Create lagged features for time series.
    
    Args:
        df: Input DataFrame
        columns: List of column names to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with added lag features
    """
    df_lagged = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df_lagged


def create_rolling_features(df, columns, windows=[7, 14, 30]):
    """Create rolling window statistics.
    
    Args:
        df: Input DataFrame
        columns: List of column names
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features
    """
    df_rolling = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                df_rolling[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                df_rolling[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
    
    return df_rolling


def create_time_features(df):
    """Extract time-based features from datetime index.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with time features
    """
    df_time = df.copy()
    
    df_time['year'] = df_time.index.year
    df_time['month'] = df_time.index.month
    df_time['day'] = df_time.index.day
    df_time['dayofweek'] = df_time.index.dayofweek
    df_time['dayofyear'] = df_time.index.dayofyear
    df_time['quarter'] = df_time.index.quarter
    df_time['is_weekend'] = (df_time.index.dayofweek >= 5).astype(int)
    
    return df_time


def split_train_test(df, test_days=30):
    """Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_days: Number of days for test set
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_size = len(df) - test_days
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    return train_df, test_df


def prepare_for_ml(df, target_cols, feature_cols=None, test_days=30):
    """Prepare data for ML models with features and train/test split.
    
    Args:
        df: Input DataFrame
        target_cols: List of target column names
        feature_cols: List of feature column names (None = all except targets)
        test_days: Number of days for test set
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    # Create features
    df_features = create_time_features(df)
    df_features = create_lag_features(df_features, target_cols, lags=[1, 7, 14])
    df_features = create_rolling_features(df_features, target_cols, windows=[7, 14])
    
    # Drop rows with NaN (from lag/rolling)
    df_features = df_features.dropna()
    
    # Define features
    if feature_cols is None:
        feature_cols = [col for col in df_features.columns if col not in target_cols]
    
    # Split
    train_df, test_df = split_train_test(df_features, test_days)
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]
    
    return X_train, X_test, y_train, y_test, feature_cols


def prepare_sequences(data, lookback=30):
    """Prepare sequences for deep learning models.
    
    Args:
        data: Numpy array or Series
        lookback: Number of timesteps to look back
        
    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)


def full_pipeline(filepath, test_days=30):
    """Complete preprocessing pipeline.
    
    Args:
        filepath: Path to data file
        test_days: Number of days for test set
        
    Returns:
        Tuple of (train_df, test_df, full_df)
    """
    # Load and convert
    df = load_data(filepath)
    df = convert_to_numeric(df)
    
    # Handle missing values
    df = handle_missing_values(df, method='interpolate')
    
    # Split
    train_df, test_df = split_train_test(df, test_days)
    
    return train_df, test_df, df

"""Visualization utilities for UV time series models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime


def plot_forecast_grid(predictions_dict, save_path=None, lookback_days=180, grid_size=(4, 4), model_name=None):
    """Plot grid of forecasts vs actuals for multiple variables.
    
    Args:
        predictions_dict: Dict with {var_name: {'train': series, 'test': series, 'forecast': array}}
        save_path: Path to save figure (None = show only)
        lookback_days: Number of recent training days to display
        grid_size: Tuple of (rows, cols) for subplot grid
        model_name: Name of the model (displayed in title)
    """
    variables = list(predictions_dict.keys())
    rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, var_name in enumerate(variables):
        if i >= len(axes):
            break
        
        data = predictions_dict[var_name]
        ax = axes[i]
        
        # Plot recent training data
        train = data['train']
        plot_train = train.iloc[-lookback_days:] if len(train) > lookback_days else train
        
        ax.plot(plot_train.index, plot_train.values, label='Train', alpha=0.7)
        ax.plot(data['test'].index, data['test'].values, label='Actual', color='green', linewidth=2)
        ax.plot(data['test'].index, data['forecast'], label='Forecast', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f"{var_name}", fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots
    for j in range(len(variables), len(axes)):
        axes[j].axis('off')
    
    # Add overall title with model name
    if model_name:
        fig.suptitle(f'{model_name} - Forecast Grid', fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_df, save_path=None, model_name=None):
    """Plot bar chart comparing RMSE and MAE across variables.
    
    Args:
        metrics_df: DataFrame with columns ['Variable', 'RMSE', 'MAE']
        save_path: Path to save figure
        model_name: Name of the model (displayed in title)
    """
    if metrics_df.empty:
        print("No metrics to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    ax.bar(x - width/2, metrics_df['RMSE'], width, label='RMSE', color='salmon', alpha=0.8)
    ax.bar(x + width/2, metrics_df['MAE'], width, label='MAE', color='skyblue', alpha=0.8)
    
    ax.set_xlabel('Variables', fontsize=12)
    ax.set_ylabel('Error Value', fontsize=12)
    title = f'{model_name} - Performance Metrics (RMSE vs MAE)' if model_name else 'Model Performance Metrics (RMSE vs MAE)'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Variable'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    
    plt.show()


def plot_single_forecast(train, test, forecast, variable_name, save_path=None, lookback_days=365, model_name=None):
    """Plot single variable forecast with confidence interval.
    
    Args:
        train: Training series
        test: Test series
        forecast: Forecast array/series
        variable_name: Name of variable
        save_path: Path to save figure
        lookback_days: Number of recent training days to show
        model_name: Name of the model (displayed in title)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot recent training data
    plot_train = train.iloc[-lookback_days:] if len(train) > lookback_days else train
    
    ax.plot(plot_train.index, plot_train.values, label='Training Data', alpha=0.7)
    ax.plot(test.index, test.values, label='Actual Test', color='green', linewidth=2, marker='o', markersize=3)
    ax.plot(test.index, forecast, label='Forecast', color='red', linestyle='--', linewidth=2, marker='s', markersize=3)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(variable_name, fontsize=12)
    title = f'{model_name} - Forecast vs Actual: {variable_name}' if model_name else f'Forecast vs Actual: {variable_name}'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Single forecast plot saved to {save_path}")
    
    plt.show()


def plot_residuals(test, forecast, variable_name, save_path=None, model_name=None):
    """Plot residual analysis for forecast.
    
    Args:
        test: Actual test values
        forecast: Forecasted values
        variable_name: Name of variable
        save_path: Path to save figure
        model_name: Name of the model (displayed in title)
    """
    residuals = test.values - forecast
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    axes[0, 0].plot(test.index, residuals, marker='o', linestyle='-', alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Actual vs Predicted
    axes[1, 1].scatter(test.values, forecast, alpha=0.6)
    axes[1, 1].plot([test.values.min(), test.values.max()], 
                    [test.values.min(), test.values.max()], 
                    'r--', lw=2)
    axes[1, 1].set_title('Actual vs Predicted')
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].grid(True, alpha=0.3)
    
    title = f'{model_name} - Residual Analysis: {variable_name}' if model_name else f'Residual Analysis: {variable_name}'
    plt.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importances, feature_names, model_name, top_n=20, save_path=None):
    """Plot feature importance for ML models.
    
    Args:
        importances: Array of feature importances
        feature_names: List of feature names
        model_name: Name of the model
        top_n: Number of top features to display
        save_path: Path to save figure
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(top_n), importances[indices], align='center', alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances: {model_name}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None, model_name=None):
    """Plot training history for deep learning models.
    
    Args:
        history: Keras History object or dict with 'loss' and 'val_loss'
        save_path: Path to save figure
        model_name: Name of the model (displayed in title)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if hasattr(history, 'history'):
        history = history.history
    
    ax.plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    title = f'{model_name} - Training History' if model_name else 'Model Training History'
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def save_all_plots(model_name, predictions_dict, metrics_df, gen_folder='gen'):
    """Save all standard plots for a model.
    
    Args:
        model_name: Name of the model (used in filenames)
        predictions_dict: Predictions dictionary
        metrics_df: Metrics DataFrame
        gen_folder: Folder to save plots
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create model-specific subfolder
    model_folder = os.path.join(gen_folder, model_name.lower())
    os.makedirs(model_folder, exist_ok=True)
    
    # Grid plot
    grid_path = os.path.join(model_folder, f'{model_name}_forecast_grid_{timestamp}.png')
    plot_forecast_grid(predictions_dict, save_path=grid_path, model_name=model_name)
    
    # Metrics plot
    metrics_path = os.path.join(model_folder, f'{model_name}_metrics_{timestamp}.png')
    plot_metrics_comparison(metrics_df, save_path=metrics_path, model_name=model_name)
    
    print(f"\nAll plots saved to {model_folder}/ folder")


def print_metrics_summary(metrics_df):
    """Print formatted metrics summary.
    
    Args:
        metrics_df: DataFrame with metrics
    """
    if metrics_df.empty:
        print("No metrics available.")
        return
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(metrics_df.to_string(index=False))
    print("="*60)
    print(f"Average RMSE: {metrics_df['RMSE'].mean():.4f}")
    print(f"Average MAE:  {metrics_df['MAE'].mean():.4f}")
    print("="*60 + "\n")

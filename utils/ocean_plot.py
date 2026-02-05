"""
Ocean Plotting Utilities

This module provides visualization functions for the UV & Ocean Ecosystem Analysis,
including correlation matrices, trend lines, and multi-variable interaction plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_heatmap(df, variables=None, title='Ecosystem Correlation Matrix'):
    """
    Plots a correlation heatmap for specified variables.
    
    Args:
        df (pd.DataFrame): The ecosystem dataframe.
        variables (list, optional): List of column names to include. 
                                  Defaults to UV, Ozone, Temp, Salinity, Oxygen.
        title (str): Plot title.
    """
    if df.empty:
        print("No data to plot heatmap.")
        return

    if variables is None:
        variables = ['uv_mean', 'ozone_mean', 'ocean_temp_c_mean', 'ocean_salinity_mean', 'ocean_oxygen_mean']
    
    # Filter only columns that exist and have sufficient data
    valid_vars = [c for c in variables if c in df.columns and df[c].notna().sum() > 5]
    
    if len(valid_vars) < 2:
        print("Not enough variables for correlation analysis.")
        return
        
    corr_matrix = df[valid_vars].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def plot_ecosystem_trends(df, rolling_window=5):
    """
    Visualizes normalized trends of UV, Ozone, and Ocean Temperature over time.
    Uses rolling averages to smooth out short-term fluctuations.
    """
    if df.empty:
        print("No data to plot trends.")
        return

    # Normalize for comparison
    cols_to_plot = ['uv_mean', 'ozone_mean', 'ocean_temp_c_mean', 'ocean_salinity_mean']
    # Filter existing
    cols_to_plot = [c for c in cols_to_plot if c in df.columns]
    
    df_norm = df.set_index('year')[cols_to_plot].copy()
    df_norm = (df_norm - df_norm.mean()) / df_norm.std()
    
    plt.figure(figsize=(14, 7))
    
    # Plot smoothed lines
    for col in cols_to_plot:
        label = col.replace('_mean', '').replace('_', ' ').title()
        # Handle sparse data with rolling mean
        series = df_norm[col].rolling(window=rolling_window, min_periods=1, center=True).mean()
        plt.plot(series, label=f"{label} ({rolling_window}-yr Rolling)", linewidth=2.5)
        
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title('Normalized Multi-Decadal Ecosystem Trends (Smoothed)')
    plt.xlabel('Year')
    plt.ylabel('Normalized Deviation (Z-Score)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_uv_ocean_interactions(df):
    """
    Creates relational plots (scatter with regression) to verify hypotheses
    like the UV-Salinity coupling or UV-Temperature relationship.
    """
    if df.empty:
        return

    # 1. UV vs Salinity
    if 'uv_mean' in df.columns and 'ocean_salinity_mean' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x='uv_mean', y='ocean_salinity_mean', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        plt.title('Interaction: Surface UV vs Ocean Salinity')
        plt.xlabel('Mean UV Index')
        plt.ylabel('Mean Ocean Salinity (PSU)')
        plt.grid(True, alpha=0.3)
        plt.show()

    # 2. UV vs Temperature
    if 'uv_mean' in df.columns and 'ocean_temp_c_mean' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x='uv_mean', y='ocean_temp_c_mean', scatter_kws={'alpha':0.6}, line_kws={'color':'green'})
        plt.title('Interaction: Surface UV vs Ocean Temperature')
        plt.xlabel('Mean UV Index')
        plt.ylabel('Mean Ocean Temp (°C)')
        plt.grid(True, alpha=0.3)
        plt.show()

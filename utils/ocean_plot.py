"""
Ocean Plotting Utilities

This module provides visualization functions for the UV & Ocean Ecosystem Analysis,
including correlation matrices, trend lines, and multi-variable interaction plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

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


def plot_seasonal_ulcer_analysis(ulcer_monthly, uv_monthly=None):
    """
    Plots seasonal analysis of skin ulcers in fish.
    
    Hypothesis: High UV months (summer) should show lower ulcer rates
    due to UV disinfection of water-borne pathogens.
    
    Args:
        ulcer_monthly (pd.DataFrame): Monthly ulcer data.
        uv_monthly (pd.DataFrame, optional): Monthly UV data for overlay.
    """
    if ulcer_monthly.empty:
        print("No ulcer data to plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Monthly ulcer rate distribution (boxplot)
    ax1 = axes[0, 0]
    monthly_agg = ulcer_monthly.groupby('month')['ulcer_rate'].agg(['mean', 'std', 'count']).reset_index()
    colors = ['#3498db' if m in [11, 12, 1, 2] else '#e74c3c' if m in [5, 6, 7, 8] else '#95a5a6' 
              for m in monthly_agg['month']]
    ax1.bar(monthly_agg['month'], monthly_agg['mean'], yerr=monthly_agg['std'], 
            color=colors, capsize=3, alpha=0.8)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Ulcer Rate (%)')
    ax1.set_title('Seasonal Fish Skin Ulcer Rates\n(Blue=Winter/Low UV, Red=Summer/High UV)')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.axhline(ulcer_monthly['ulcer_rate'].mean(), color='black', linestyle='--', 
                alpha=0.5, label='Annual Mean')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Yearly trend of ulcer rates
    ax2 = axes[0, 1]
    yearly_ulcer = ulcer_monthly.groupby('year')['ulcer_rate'].mean().reset_index()
    ax2.plot(yearly_ulcer['year'], yearly_ulcer['ulcer_rate'], 'o-', 
             color='#9b59b6', linewidth=2, markersize=8)
    ax2.fill_between(yearly_ulcer['year'], yearly_ulcer['ulcer_rate'], alpha=0.3, color='#9b59b6')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Mean Ulcer Rate (%)')
    ax2.set_title('Long-term Trend: Fish Skin Ulcer Rates')
    ax2.grid(True, alpha=0.3)
    
    # 3. Summer vs Winter comparison
    ax3 = axes[1, 0]
    ulcer_monthly['season'] = ulcer_monthly['month'].apply(
        lambda m: 'Summer (High UV)' if m in [5, 6, 7, 8] else 
                  'Winter (Low UV)' if m in [11, 12, 1, 2] else 'Transition')
    season_data = ulcer_monthly[ulcer_monthly['season'] != 'Transition']
    sns.boxplot(data=season_data, x='season', y='ulcer_rate', ax=ax3, 
                palette={'Summer (High UV)': '#e74c3c', 'Winter (Low UV)': '#3498db'})
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Ulcer Rate (%)')
    ax3.set_title('UV Disinfection Hypothesis Test\nSummer vs Winter Ulcer Rates')
    ax3.grid(True, alpha=0.3)
    
    # 4. Sample size by month (data quality indicator)
    ax4 = axes[1, 1]
    sample_by_month = ulcer_monthly.groupby('month')['total_fish'].sum().reset_index()
    ax4.bar(sample_by_month['month'], sample_by_month['total_fish'], color='#1abc9c', alpha=0.7)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Total Fish Examined')
    ax4.set_title('Data Coverage: Fish Sampled by Month')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical summary
    summer = ulcer_monthly[ulcer_monthly['month'].isin([5, 6, 7, 8])]['ulcer_rate']
    winter = ulcer_monthly[ulcer_monthly['month'].isin([11, 12, 1, 2])]['ulcer_rate']
    
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY: UV Disinfection Hypothesis")
    print("="*60)
    print(f"  Summer (High UV) Mean Ulcer Rate: {summer.mean():.2f}% (n={len(summer)})")
    print(f"  Winter (Low UV) Mean Ulcer Rate:  {winter.mean():.2f}% (n={len(winter)})")
    print(f"  Difference: {winter.mean() - summer.mean():.2f}%")
    
    if winter.mean() > summer.mean():
        print("\n  HYPOTHESIS SUPPORTED: Higher ulcer rates in low-UV winter months")
    else:
        print("\n  HYPOTHESIS NOT SUPPORTED by this data")
    print("="*60)


def plot_contaminant_photodegradation(contaminant_monthly, contaminant_yearly, param_name='DDEPP'):
    """
    Plots analysis of chemical contaminant photodegradation by UV.
    
    Hypothesis: UV radiation breaks down chemical contaminants.
    High UV periods should show lower contaminant concentrations.
    
    Args:
        contaminant_monthly (pd.DataFrame): Monthly contaminant data.
        contaminant_yearly (pd.DataFrame): Yearly contaminant data.
        param_name (str): Name of the contaminant parameter.
    """
    if contaminant_monthly.empty:
        print("No contaminant data to plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Monthly concentration pattern
    ax1 = axes[0, 0]
    monthly_mean = contaminant_monthly.groupby('month')['mean_concentration'].agg(['mean', 'std']).reset_index()
    colors = ['#e74c3c' if m in [5, 6, 7, 8] else '#3498db' if m in [11, 12, 1, 2] else '#95a5a6' 
              for m in monthly_mean['month']]
    ax1.bar(monthly_mean['month'], monthly_mean['mean'], yerr=monthly_mean['std'],
            color=colors, capsize=3, alpha=0.8)
    ax1.set_xlabel('Month')
    ax1.set_ylabel(f'{param_name} Concentration (µg/L)')
    ax1.set_title(f'Seasonal {param_name} Concentrations\n(Red=Summer/High UV, Blue=Winter/Low UV)')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.grid(True, alpha=0.3)
    
    # 2. Long-term trend
    ax2 = axes[0, 1]
    ax2.plot(contaminant_yearly['year'], contaminant_yearly['mean_concentration'], 
             'o-', color='#8e44ad', linewidth=2, markersize=8)
    ax2.fill_between(contaminant_yearly['year'], contaminant_yearly['mean_concentration'], 
                     alpha=0.3, color='#8e44ad')
    ax2.set_xlabel('Year')
    ax2.set_ylabel(f'{param_name} Concentration (µg/L)')
    ax2.set_title(f'Long-term Trend: {param_name} Concentrations')
    ax2.grid(True, alpha=0.3)
    
    # 3. Summer vs Winter comparison
    ax3 = axes[1, 0]
    contaminant_monthly['season'] = contaminant_monthly['month'].apply(
        lambda m: 'Summer (High UV)' if m in [5, 6, 7, 8] else 
                  'Winter (Low UV)' if m in [11, 12, 1, 2] else 'Transition')
    season_data = contaminant_monthly[contaminant_monthly['season'] != 'Transition']
    sns.boxplot(data=season_data, x='season', y='mean_concentration', ax=ax3,
                palette={'Summer (High UV)': '#e74c3c', 'Winter (Low UV)': '#3498db'})
    ax3.set_xlabel('Season')
    ax3.set_ylabel(f'{param_name} Concentration (µg/L)')
    ax3.set_title(f'Photodegradation Hypothesis Test\nSummer vs Winter {param_name}')
    ax3.grid(True, alpha=0.3)
    
    # 4. Detection rate (data quality)
    ax4 = axes[1, 1]
    detection_by_month = contaminant_monthly.groupby('month')['below_detection_pct'].mean().reset_index()
    ax4.bar(detection_by_month['month'], 100 - detection_by_month['below_detection_pct'], 
            color='#27ae60', alpha=0.7)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Detection Rate (%)')
    ax4.set_title(f'{param_name} Detection Rate by Month\n(Higher = More Detectable Levels)')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical summary
    summer = contaminant_monthly[contaminant_monthly['month'].isin([5, 6, 7, 8])]['mean_concentration']
    winter = contaminant_monthly[contaminant_monthly['month'].isin([11, 12, 1, 2])]['mean_concentration']
    
    print("\n" + "="*60)
    print(f"STATISTICAL SUMMARY: UV Photodegradation of {param_name}")
    print("="*60)
    print(f"  Summer (High UV) Mean Concentration: {summer.mean():.6f} µg/L (n={len(summer)})")
    print(f"  Winter (Low UV) Mean Concentration:  {winter.mean():.6f} µg/L (n={len(winter)})")
    print(f"  Reduction: {((winter.mean() - summer.mean()) / winter.mean() * 100):.1f}%")
    
    if summer.mean() < winter.mean():
        print(f"\n  HYPOTHESIS SUPPORTED: Lower {param_name} in high-UV summer months")
        print("     UV photodegradation appears to break down contaminants")
    else:
        print(f"\n  HYPOTHESIS NOT SUPPORTED by this data")
    print("="*60)


def plot_uv_biological_correlation(merged_df):
    """
    Creates correlation analysis between UV and biological/chemical indicators.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset with UV, ulcer, and contaminant data.
    """
    if merged_df.empty:
        print("No data for correlation analysis.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Correlation heatmap
    ax1 = axes[0]
    corr_cols = [c for c in ['uv_mean', 'ozone_mean', 'ulcer_rate', 'ddepp_concentration'] 
                 if c in merged_df.columns and merged_df[c].notna().sum() > 3]
    
    if len(corr_cols) >= 2:
        corr = merged_df[corr_cols].corr()
        sns.heatmap(corr, annot=True, cmap='RdYlGn_r', vmin=-1, vmax=1, center=0, ax=ax1)
        ax1.set_title('UV vs Biological/Chemical Correlation')
    else:
        ax1.text(0.5, 0.5, 'Insufficient data\nfor correlation', ha='center', va='center')
        ax1.set_title('Correlation Matrix')
    
    # 2. UV vs Ulcer Rate scatter
    ax2 = axes[1]
    if 'uv_mean' in merged_df.columns and 'ulcer_rate' in merged_df.columns:
        valid = merged_df.dropna(subset=['uv_mean', 'ulcer_rate'])
        if len(valid) > 0:
            sns.regplot(data=valid, x='uv_mean', y='ulcer_rate', ax=ax2,
                       scatter_kws={'s': 80, 'alpha': 0.6}, line_kws={'color': 'red'})
            ax2.set_xlabel('Mean UV Index')
            ax2.set_ylabel('Fish Skin Ulcer Rate (%)')
            ax2.set_title('UV Disinfection Effect\n(Expected: Negative Correlation)')
    ax2.grid(True, alpha=0.3)
    
    # 3. UV vs Contaminant scatter
    ax3 = axes[2]
    if 'uv_mean' in merged_df.columns and 'ddepp_concentration' in merged_df.columns:
        valid = merged_df.dropna(subset=['uv_mean', 'ddepp_concentration'])
        if len(valid) > 0:
            sns.regplot(data=valid, x='uv_mean', y='ddepp_concentration', ax=ax3,
                       scatter_kws={'s': 80, 'alpha': 0.6}, line_kws={'color': 'purple'})
            ax3.set_xlabel('Mean UV Index')
            ax3.set_ylabel('DDEPP Concentration (µg/L)')
            ax3.set_title('UV Photodegradation Effect\n(Expected: Negative Correlation)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# HYPOTHESIS TESTING FUNCTIONS
# =============================================================================

def hypothesis_test_uv_salinity(eco_df, alpha=0.05, verbose=False):
    """
    Formal Hypothesis Testing: UV-Salinity Negative Correlation
    
    H₀ (Null): There is no correlation between UV and Salinity (ρ = 0)
    H₁ (Alternative): UV and Salinity are negatively correlated (ρ < 0)
    
    This establishes the foundation: When UV is high, salinity drops (freshwater runoff).
    """
    if verbose:
        print("\n" + "="*80)
        print("HYPOTHESIS TEST 1: UV-SALINITY CORRELATION (Foundation)")
        print("="*80)
    
    if eco_df.empty or 'uv_mean' not in eco_df.columns or 'ocean_salinity_mean' not in eco_df.columns:
        if verbose:
            print("Insufficient data for UV-Salinity correlation test")
        return None
    
    valid = eco_df.dropna(subset=['uv_mean', 'ocean_salinity_mean'])
    n = len(valid)
    
    if n < 3:
        if verbose:
            print("Not enough data points for correlation test")
        return None
    
    # Pearson correlation with p-value
    r, p_value = stats.pearsonr(valid['uv_mean'], valid['ocean_salinity_mean'])
    
    # One-tailed test for negative correlation
    p_one_tailed = p_value / 2 if r < 0 else 1 - p_value / 2
    
    # Effect size interpretation (Cohen's guidelines for r)
    if abs(r) >= 0.5:
        effect_size = "LARGE"
    elif abs(r) >= 0.3:
        effect_size = "MEDIUM"
    elif abs(r) >= 0.1:
        effect_size = "SMALL"
    else:
        effect_size = "NEGLIGIBLE"
    
    if verbose:
        print(f"\n  H₀: ρ(UV, Salinity) = 0  (No correlation)")
        print(f"  H₁: ρ(UV, Salinity) < 0  (Negative correlation)")
        print(f"\n  Sample size (n): {n}")
        print(f"  Pearson r: {r:.4f}")
        print(f"  Effect Size: {effect_size}")
        print(f"  p-value (one-tailed): {p_one_tailed:.6f}")
        print(f"  Significance level (α): {alpha}")
    
    if p_one_tailed < alpha and r < 0:
        if verbose:
            print(f"\n  REJECT H0: Significant negative correlation (p < {alpha})")
            print(f"  CONCLUSION: UV and Salinity are significantly negatively correlated")
            print(f"     When UV increases, Salinity decreases (freshwater runoff during high UV periods)")
        decision = "REJECT"
    else:
        if verbose:
            print(f"\n  FAIL TO REJECT H0 (p = {p_one_tailed:.4f} >= {alpha})")
        decision = "FAIL_TO_REJECT"
    
    if verbose:
        print("="*80)
    
    return {
        'test': 'UV-Salinity Correlation',
        'r': r,
        'p_value': p_one_tailed,
        'n': n,
        'effect_size': effect_size,
        'decision': decision,
        'alpha': alpha
    }


def hypothesis_test_uv_disinfection_biological(ulcer_monthly, alpha=0.05, verbose=False):
    """
    Formal Hypothesis Testing: UV Biological Disinfection (Fish Skin Ulcers)
    
    H₀ (Null): Summer ulcer rate = Winter ulcer rate (UV has no disinfection effect)
    H₁ (Alternative): Summer ulcer rate < Winter ulcer rate (UV disinfects water)
    
    Test: Independent samples t-test (one-tailed)
    
    Logic: If UV disinfects water, fewer bacteria → fewer fish infections in summer.
    """
    if verbose:
        print("\n" + "="*80)
        print("HYPOTHESIS TEST 2: UV BIOLOGICAL DISINFECTION (Fish Skin Ulcers)")
        print("="*80)
    
    if ulcer_monthly.empty:
        if verbose:
            print("No ulcer data available for hypothesis testing")
        return None
    
    # Define seasons: Summer (high UV) = May-Aug, Winter (low UV) = Nov-Feb
    summer = ulcer_monthly[ulcer_monthly['month'].isin([5, 6, 7, 8])]['ulcer_rate'].dropna()
    winter = ulcer_monthly[ulcer_monthly['month'].isin([11, 12, 1, 2])]['ulcer_rate'].dropna()
    
    n_summer, n_winter = len(summer), len(winter)
    
    if n_summer < 2 or n_winter < 2:
        if verbose:
            print(f"Insufficient samples (Summer: {n_summer}, Winter: {n_winter})")
        return None
    
    mean_summer = summer.mean()
    mean_winter = winter.mean()
    std_summer = summer.std()
    std_winter = winter.std()
    
    # Welch's t-test (does not assume equal variances)
    t_stat, p_value_two_tailed = stats.ttest_ind(summer, winter, equal_var=False)
    
    # One-tailed: we predict summer < winter
    p_one_tailed = p_value_two_tailed / 2 if t_stat < 0 else 1 - p_value_two_tailed / 2
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_summer-1)*std_summer**2 + (n_winter-1)*std_winter**2) / (n_summer + n_winter - 2))
    cohens_d = (mean_summer - mean_winter) / pooled_std if pooled_std > 0 else 0
    
    if abs(cohens_d) >= 0.8:
        effect_size = "LARGE"
    elif abs(cohens_d) >= 0.5:
        effect_size = "MEDIUM"
    elif abs(cohens_d) >= 0.2:
        effect_size = "SMALL"
    else:
        effect_size = "NEGLIGIBLE"
    
    if verbose:
        print(f"\n  H0: mean_summer = mean_winter  (No difference in ulcer rates)")
        print(f"  H1: mean_summer < mean_winter  (Lower ulcers in high-UV summer)")
        print(f"\n  Summer (High UV) Months: May-Aug")
        print(f"    n = {n_summer}, Mean = {mean_summer:.2f}%, SD = {std_summer:.2f}")
        print(f"  Winter (Low UV) Months: Nov-Feb")
        print(f"    n = {n_winter}, Mean = {mean_winter:.2f}%, SD = {std_winter:.2f}")
        print(f"\n  Difference: {mean_winter - mean_summer:.2f}% (Winter - Summer)")
        print(f"  Welch's t-statistic: {t_stat:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f} ({effect_size} effect)")
        print(f"  p-value (one-tailed): {p_one_tailed:.6f}")
        print(f"  Significance level (alpha): {alpha}")
    
    if p_one_tailed < alpha and mean_summer < mean_winter:
        if verbose:
            print(f"\n  REJECT H0: Significant difference (p < {alpha})")
            print(f"  CONCLUSION: Fish skin ulcer rates are significantly LOWER in summer")
            print(f"     UV DISINFECTION HYPOTHESIS SUPPORTED")
        decision = "REJECT"
    elif mean_summer < mean_winter:
        if verbose:
            print(f"\n  Trend supports hypothesis but not statistically significant (p = {p_one_tailed:.4f})")
        decision = "TREND_ONLY"
    else:
        if verbose:
            print(f"\n  FAIL TO REJECT H0: No evidence for UV disinfection effect")
        decision = "FAIL_TO_REJECT"
    
    if verbose:
        print("="*80)
    
    return {
        'test': 'UV Biological Disinfection (Ulcers)',
        'mean_summer': mean_summer,
        'mean_winter': mean_winter,
        'difference': mean_winter - mean_summer,
        't_stat': t_stat,
        'p_value': p_one_tailed,
        'cohens_d': cohens_d,
        'effect_size': effect_size,
        'n_summer': n_summer,
        'n_winter': n_winter,
        'decision': decision,
        'alpha': alpha
    }


def hypothesis_test_uv_photodegradation(contaminant_monthly, param_name='DDEPP', alpha=0.05, verbose=False):
    """
    Formal Hypothesis Testing: UV Chemical Photodegradation (DDEPP Contaminant)
    
    H₀ (Null): Summer concentration = Winter concentration (UV has no photodegradation effect)
    H₁ (Alternative): Summer concentration < Winter concentration (UV degrades chemicals)
    
    Test: Independent samples t-test (one-tailed)
    
    Logic: If UV breaks down chemicals, contaminant levels should be lower in summer.
    """
    if verbose:
        print("\n" + "="*80)
        print(f"HYPOTHESIS TEST 3: UV PHOTODEGRADATION ({param_name} Contaminant)")
        print("="*80)
    
    if contaminant_monthly.empty:
        if verbose:
            print("No contaminant data available for hypothesis testing")
        return None
    
    summer = contaminant_monthly[contaminant_monthly['month'].isin([5, 6, 7, 8])]['mean_concentration'].dropna()
    winter = contaminant_monthly[contaminant_monthly['month'].isin([11, 12, 1, 2])]['mean_concentration'].dropna()
    
    n_summer, n_winter = len(summer), len(winter)
    
    if n_summer < 2 or n_winter < 2:
        if verbose:
            print(f"Insufficient samples (Summer: {n_summer}, Winter: {n_winter})")
        return None
    
    mean_summer = summer.mean()
    mean_winter = winter.mean()
    std_summer = summer.std()
    std_winter = winter.std()
    
    # Welch's t-test
    t_stat, p_value_two_tailed = stats.ttest_ind(summer, winter, equal_var=False)
    p_one_tailed = p_value_two_tailed / 2 if t_stat < 0 else 1 - p_value_two_tailed / 2
    
    # Cohen's d
    pooled_std = np.sqrt(((n_summer-1)*std_summer**2 + (n_winter-1)*std_winter**2) / (n_summer + n_winter - 2))
    cohens_d = (mean_summer - mean_winter) / pooled_std if pooled_std > 0 else 0
    
    if abs(cohens_d) >= 0.8:
        effect_size = "LARGE"
    elif abs(cohens_d) >= 0.5:
        effect_size = "MEDIUM"
    elif abs(cohens_d) >= 0.2:
        effect_size = "SMALL"
    else:
        effect_size = "NEGLIGIBLE"
    
    # Percent reduction
    pct_reduction = ((mean_winter - mean_summer) / mean_winter * 100) if mean_winter > 0 else 0
    
    if verbose:
        print(f"\n  H0: mean_summer = mean_winter  (No difference in {param_name} concentration)")
        print(f"  H1: mean_summer < mean_winter  (Lower {param_name} in high-UV summer)")
        print(f"\n  Summer (High UV) Months: May-Aug")
        print(f"    n = {n_summer}, Mean = {mean_summer:.6f} ug/L, SD = {std_summer:.6f}")
        print(f"  Winter (Low UV) Months: Nov-Feb")
        print(f"    n = {n_winter}, Mean = {mean_winter:.6f} ug/L, SD = {std_winter:.6f}")
        print(f"\n  Reduction: {pct_reduction:.1f}% lower in summer")
        print(f"  Welch's t-statistic: {t_stat:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f} ({effect_size} effect)")
        print(f"  p-value (one-tailed): {p_one_tailed:.6f}")
        print(f"  Significance level (alpha): {alpha}")
    
    if p_one_tailed < alpha and mean_summer < mean_winter:
        if verbose:
            print(f"\n  REJECT H0: Significant difference (p < {alpha})")
            print(f"  CONCLUSION: {param_name} concentrations are significantly LOWER in summer")
            print(f"     UV PHOTODEGRADATION HYPOTHESIS SUPPORTED")
        decision = "REJECT"
    elif mean_summer < mean_winter:
        if verbose:
            print(f"\n  Trend supports hypothesis but not statistically significant (p = {p_one_tailed:.4f})")
        decision = "TREND_ONLY"
    else:
        if verbose:
            print(f"\n  FAIL TO REJECT H0: No evidence for UV photodegradation effect")
        decision = "FAIL_TO_REJECT"
    
    if verbose:
        print("="*80)
    
    return {
        'test': f'UV Photodegradation ({param_name})',
        'mean_summer': mean_summer,
        'mean_winter': mean_winter,
        'pct_reduction': pct_reduction,
        't_stat': t_stat,
        'p_value': p_one_tailed,
        'cohens_d': cohens_d,
        'effect_size': effect_size,
        'n_summer': n_summer,
        'n_winter': n_winter,
        'decision': decision,
        'alpha': alpha
    }


def hypothesis_test_uv_correlation_biological(merged_df, alpha=0.05, verbose=False):
    """
    Formal Hypothesis Testing: UV-Ulcer Correlation
    
    H₀: ρ(UV, Ulcer Rate) = 0  (No correlation)
    H₁: ρ(UV, Ulcer Rate) < 0  (Negative correlation - more UV = fewer ulcers)
    """
    if verbose:
        print("\n" + "="*80)
        print("HYPOTHESIS TEST 4: UV-ULCER RATE CORRELATION (Direct Link)")
        print("="*80)
    
    if 'uv_mean' not in merged_df.columns or 'ulcer_rate' not in merged_df.columns:
        if verbose:
            print("Missing required columns for UV-Ulcer correlation")
        return None
    
    valid = merged_df.dropna(subset=['uv_mean', 'ulcer_rate'])
    n = len(valid)
    
    if n < 3:
        if verbose:
            print(f"Insufficient data points (n = {n})")
        return None
    
    r, p_value = stats.pearsonr(valid['uv_mean'], valid['ulcer_rate'])
    p_one_tailed = p_value / 2 if r < 0 else 1 - p_value / 2
    
    if abs(r) >= 0.5:
        effect_size = "LARGE"
    elif abs(r) >= 0.3:
        effect_size = "MEDIUM"
    elif abs(r) >= 0.1:
        effect_size = "SMALL"
    else:
        effect_size = "NEGLIGIBLE"
    
    if verbose:
        print(f"\n  H0: r(UV, Ulcer Rate) = 0  (No correlation)")
        print(f"  H1: r(UV, Ulcer Rate) < 0  (More UV -> Fewer ulcers)")
        print(f"\n  Sample size (n): {n}")
        print(f"  Pearson r: {r:.4f}")
        print(f"  Effect Size: {effect_size}")
        print(f"  p-value (one-tailed): {p_one_tailed:.6f}")
        print(f"  Significance level (alpha): {alpha}")
    
    if p_one_tailed < alpha and r < 0:
        if verbose:
            print(f"\n  REJECT H0: Significant negative correlation (p < {alpha})")
            print(f"  CONCLUSION: Higher UV is associated with lower ulcer rates")
        decision = "REJECT"
    elif r < 0:
        if verbose:
            print(f"\n  Negative trend but not statistically significant (p = {p_one_tailed:.4f})")
        decision = "TREND_ONLY"
    else:
        if verbose:
            print(f"\n  FAIL TO REJECT H0")
        decision = "FAIL_TO_REJECT"
    
    if verbose:
        print("="*80)
    
    return {
        'test': 'UV-Ulcer Correlation',
        'r': r,
        'p_value': p_one_tailed,
        'n': n,
        'effect_size': effect_size,
        'decision': decision,
        'alpha': alpha
    }


def run_all_hypothesis_tests(eco_df, ulcer_monthly, contaminant_monthly, merged_df, alpha=0.05, verbose=False):
    """
    Master function to run all hypothesis tests and produce a comprehensive summary.
    
    Tests the complete UV Disinfection Theory:
    1. UV ↔ Salinity correlation (establishes seasonal pattern)
    2. Biological disinfection (skin ulcers - summer vs winter)
    3. Chemical photodegradation (DDEPP - summer vs winter)
    4. Direct UV-Ulcer correlation
    
    Args:
        verbose: If True, print detailed output for each test. Default False for compact output.
    
    Returns:
        dict: All test results for further analysis
    """
    print("\nRunning Hypothesis Testing Suite (alpha = {:.2f})...".format(alpha))
    
    results = {}
    
    # Test 1: Foundation - UV-Salinity relationship
    results['uv_salinity'] = hypothesis_test_uv_salinity(eco_df, alpha, verbose)
    
    # Test 2: Biological disinfection
    results['biological'] = hypothesis_test_uv_disinfection_biological(ulcer_monthly, alpha, verbose)
    
    # Test 3: Chemical photodegradation
    results['chemical'] = hypothesis_test_uv_photodegradation(contaminant_monthly, 'DDEPP', alpha, verbose)
    
    # Test 4: Direct correlation
    results['uv_ulcer_corr'] = hypothesis_test_uv_correlation_biological(merged_df, alpha, verbose)
    
    # Print compact summary table
    print("\n{:<32} {:>10} {:>12} {:>10}".format("Test", "p-value", "Decision", "Result"))
    print("-"*66)
    
    test_labels = {
        'uv_salinity': 'UV-Salinity Correlation',
        'biological': 'Biological Disinfection',
        'chemical': 'Chemical Photodegradation',
        'uv_ulcer_corr': 'UV-Ulcer Correlation'
    }
    
    for key, label in test_labels.items():
        if results.get(key):
            p = results[key].get('p_value', float('nan'))
            dec = results[key].get('decision', 'N/A')
            if dec == 'REJECT':
                symbol = 'YES'
            elif dec == 'TREND_ONLY':
                symbol = 'TREND'
            else:
                symbol = 'NO'
            print("{:<32} {:>10.4f} {:>12} {:>10}".format(label, p, dec, symbol))
        else:
            print("{:<32} {:>10} {:>12} {:>10}".format(label, 'N/A', 'N/A', '-'))
    
    print("-"*66)
    
    # Summary visualization
    plot_hypothesis_test_summary(results, alpha)
    
    return results


def plot_hypothesis_test_summary(results, alpha=0.05):
    """
    Creates a visual summary of all hypothesis test results.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    tests = []
    p_values = []
    colors = []
    decisions = []
    
    test_names = {
        'uv_salinity': 'UV-Salinity\nCorrelation',
        'biological': 'Biological\nDisinfection\n(Fish Ulcers)',
        'chemical': 'Chemical\nPhotodegradation\n(DDEPP)',
        'uv_ulcer_corr': 'UV-Ulcer\nDirect Correlation'
    }
    
    for key, name in test_names.items():
        if results.get(key) and results[key].get('p_value') is not None:
            tests.append(name)
            p = results[key]['p_value']
            p_values.append(p)
            
            if results[key]['decision'] == 'REJECT':
                colors.append('#27ae60')  # Green
                decisions.append('Supported')
            elif results[key]['decision'] == 'TREND_ONLY':
                colors.append('#f39c12')  # Orange
                decisions.append('Trend Only')
            else:
                colors.append('#e74c3c')  # Red
                decisions.append('Not Supported')
    
    if not tests:
        ax.text(0.5, 0.5, 'No valid test results', ha='center', va='center', fontsize=14)
        plt.show()
        return
    
    y_pos = np.arange(len(tests))
    
    # Plot bars
    bars = ax.barh(y_pos, [-np.log10(p) for p in p_values], color=colors, alpha=0.8, height=0.6)
    
    # Add significance threshold line
    sig_line = -np.log10(alpha)
    ax.axvline(x=sig_line, color='red', linestyle='--', linewidth=2, label=f'α = {alpha} threshold')
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tests, fontsize=11)
    ax.set_xlabel('-log10(p-value)\n(Higher = More Significant)', fontsize=12)
    ax.set_title('HYPOTHESIS TEST RESULTS SUMMARY\nUV Water Disinfection Theory', fontsize=14, fontweight='bold')
    
    # Add p-values and decisions to bars
    for i, (bar, p, dec) in enumerate(zip(bars, p_values, decisions)):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'p = {p:.4f}\n{dec}', va='center', fontsize=10)
    
    ax.legend(loc='lower right')
    ax.set_xlim(0, max([-np.log10(p) for p in p_values]) * 1.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print final verdict
    print("\n" + "="*66)
    print("FINAL VERDICT")
    print("="*66)
    
    supported = sum(1 for r in results.values() if r and r.get('decision') == 'REJECT')
    total = sum(1 for r in results.values() if r and r.get('p_value') is not None)
    
    print(f"Tests Conducted: {total}")
    print(f"Hypotheses Supported (p < {alpha}): {supported}/{total}")
    
    if supported >= 3:
        print(f"\nSTRONG EVIDENCE: UV Water Disinfection Theory is SUPPORTED")
    elif supported >= 2:
        print(f"\nMODERATE EVIDENCE: UV Water Disinfection Theory has partial support")
    elif supported >= 1:
        print(f"\nWEAK EVIDENCE: Limited support for UV Water Disinfection Theory")
    else:
        print(f"\nINSUFFICIENT EVIDENCE: Cannot confirm UV Water Disinfection Theory")
    
    print("="*66)

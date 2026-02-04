import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

def main():
    # --- PART 1: Load and Clean Your Specific Dataset ---
    csv_file = "cancer_trend.csv"
    print(f"Reading {csv_file}...")
    
    try:
        # We assume the file is comma-separated based on your snippet
        df = pd.read_csv(csv_file)
        
        # 1. Remove the "Total" row at the bottom
        df = df[df['Year'] != 'Total'].copy()
        
        # 2. Convert 'Year' and 'Age-Adjusted Rate' to numbers
        # The cleaning ensures "1999" (string) becomes 1999 (int)
        df['Year'] = pd.to_numeric(df['Year'])
        df['Age-Adjusted Rate'] = pd.to_numeric(df['Age-Adjusted Rate'])
        
        print(f"Successfully loaded {len(df)} years of cancer data (1999-2020).")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Make sure you saved your data!")
        return
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return

    # --- PART 2: Fetch Matching UV Data (NASA) ---
    # We use the geographic center of the Contiguous US (Lebanon, KS)
    # as a proxy for the "National" UV exposure trend.
    lat, lon = 39.82, -98.57
    print("\nFetching historical UV data from NASA (this takes ~10 seconds)...")
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "ALLSKY_SFC_UV_INDEX",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        # Fetching the exact range of your dataset
        "start": "19990101", 
        "end": "20201231",
        "format": "JSON"
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        daily_uv = data['properties']['parameter']['ALLSKY_SFC_UV_INDEX']
        
        # Process Dictionary to DataFrame
        uv_df = pd.DataFrame(list(daily_uv.items()), columns=['Date', 'UV'])
        
        # Filter out NASA error codes (-999)
        uv_df = uv_df[uv_df['UV'] != -999]
        
        # Extract Year and calculate Annual Average
        uv_df['Year'] = uv_df['Date'].str[:4].astype(int)
        annual_uv = uv_df.groupby('Year')['UV'].mean().reset_index()
        annual_uv.columns = ['Year', 'Avg_UV_Index']
        
    except Exception as e:
        print(f"Failed to fetch UV data: {e}")
        return

    # --- PART 3: Merge & Analyze ---
    # Combine Cancer Data with UV Data
    merged = pd.merge(df, annual_uv, on='Year')
    
    # Calculate Correlation
    slope, intercept, r_value, p_value, std_err = stats.linregress(merged['Avg_UV_Index'], merged['Age-Adjusted Rate'])

    print("\n" + "="*40)
    print("SCIENTIFIC RESULTS")
    print("="*40)
    print(f"Correlation (r): {r_value:.4f}")
    print(f"R-Squared (r²): {r_value**2:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print("-" * 40)
    
    if r_value < 0.3:
        print("INTERPRETATION: Weak or No Correlation detected.")
        print("This suggests that the steady rise in cancer rates is NOT caused")
        print("by a recent increase in solar intensity, but rather by cumulative")
        print("exposure, aging populations, or improved diagnosis.")
    else:
        print("INTERPRETATION: Positive correlation detected.")

    # --- PART 4: Visualization ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot 1: Cancer Rate (Left Axis) - The Rising Line
    color_cancer = '#d62728' # Red
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Melanoma Incidence (per 100k)', color=color_cancer, fontsize=12, fontweight='bold')
    line1 = ax1.plot(merged['Year'], merged['Age-Adjusted Rate'], color=color_cancer, 
                     marker='o', linewidth=3, label='Cancer Rate (CDC)')
    ax1.tick_params(axis='y', labelcolor=color_cancer)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot 2: UV Index (Right Axis) - The Flat/Fluctuating Line
    ax2 = ax1.twinx() 
    color_uv = '#1f77b4' # Blue
    ax2.set_ylabel('Average Annual UV Index (NASA)', color=color_uv, fontsize=12, fontweight='bold')
    line2 = ax2.plot(merged['Year'], merged['Avg_UV_Index'], color=color_uv, 
                     marker='s', linestyle='--', linewidth=2, label='UV Intensity (NASA)')
    ax2.tick_params(axis='y', labelcolor=color_uv)

    # Set Y-axis limits for UV to zoom in on the fluctuation (optional, helps see valid range)
    # ax2.set_ylim(2.5, 3.5) # Un-comment if the blue line looks too flat

    # Combined Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)

    plt.title('Investigation: Does Rising UV Intensity Explain Rising Cancer Rates?\n(USA 1999-2020)', fontsize=14)
    plt.tight_layout()
    
    save_path = "final_trend_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nGraph saved to: {save_path}")
    plt.show()

    # Save the merged numerical data for your report
    merged.to_csv("study_results_merged.csv", index=False)

if __name__ == "__main__":
    main()
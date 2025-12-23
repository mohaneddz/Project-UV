import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Data Loading and Cleaning ---
# Note: You need to replace the placeholder path with your actual file path
file_path = "data/UV-ALGERIA.csv" # Update this if you run the file outside the current directory
try:
    data = pd.read_csv(
        file_path,
        dtype=str
    )
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please check the file path.")
    exit()

# Strip whitespace from all string columns
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Replace invalid/missing values with NaN (new data uses empty strings)
data.replace({'': np.nan}, inplace=True)

# Keep only rows with valid dates and convert date column
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
data = data.dropna(subset=['Date'])
data.set_index('Date', inplace=True)

# --- Data Preparation for Plotting (Focusing on Key Variables) ---

# Define the two key variables for the plot
KEY_UV_COL = 'ALLSKY_SFC_UV_INDEX'  # UV Index from NASA POWER
OZONE_COL = 'TO3'                   # Total Column Ozone

# Convert selected columns to numeric, coercing errors to NaN
for col in [KEY_UV_COL, OZONE_COL]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Crucial step: Interpolate missing values (NaNs) for a continuous time series plot.
# We use linear interpolation here for a smoother visualization of trends.
data[KEY_UV_COL] = data[KEY_UV_COL].interpolate(method='linear')
data[OZONE_COL] = data[OZONE_COL].interpolate(method='linear')

# Remove any remaining rows where the primary UV or Ozone data is still NaN after interpolation
# (i.e., if the gaps were too large to interpolate)
data = data.dropna(subset=[KEY_UV_COL, OZONE_COL])

# Add year column for filtering
data['Year'] = data.index.year

# --- Plotting ---

# Use a clean, professional style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')

# Generate 10 plots, each for 2 years (starting from 1981)
for i in range(10):
    start_year = 1981 + i * 2
    end_year = start_year + 1
    subset = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
    
    if subset.empty:
        continue  # Skip if no data for the period
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary Axis (ax1) for UV Index
    color_uv = '#1f77b4'  # Blue
    ax1.set_xlabel('Date')
    ax1.set_ylabel('UV Index (ALLSKY_SFC_UV_INDEX)', color=color_uv)
    
    # Plot UV index
    ax1.plot(
        subset.index,
        subset[KEY_UV_COL],
        marker='o',            # Use small circle markers
        linestyle='-',      # Connect the markers
        color=color_uv,
        label='UV Index (ALLSKY_SFC_UV_INDEX)',
        linewidth=1.5,
        markersize=3,
        alpha=0.7
    )
    ax1.tick_params(axis='y', labelcolor=color_uv)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Final Touches
    plt.title(
        f'UV Index Time Series: {start_year}-{end_year}',
        fontsize=16,
        fontweight='bold'
    )
    
    # Legend
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(f'data/years/uv_plot_{start_year}_{end_year}.png')
    plt.close()

print("\n--- Summary ---")
print("Generated 10 plots, each for a 2-year period, focusing on UV Index.")
print("NaNs were handled using linear interpolation for a continuous line.")

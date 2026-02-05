import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load both datasets
sunburn_df = pd.read_csv(os.path.join(script_dir, "data", "sunburn_data.csv"))
uv_df = pd.read_csv(os.path.join(script_dir, "data", "uv_index_data.csv"))

# Convert date columns to datetime
sunburn_df['date'] = pd.to_datetime(sunburn_df['date'])
uv_df['date'] = pd.to_datetime(uv_df['date'])

# Extract year-week for grouping
sunburn_df['year_week'] = sunburn_df['date'].dt.to_period('W')
uv_df['year_week'] = uv_df['date'].dt.to_period('W')

# Aggregate sunburn data by week
sunburn_weekly = sunburn_df.groupby('year_week').agg({
    'metric_value': ['sum', 'mean', 'count']
}).reset_index()
sunburn_weekly.columns = ['year_week', 'total_cases', 'avg_daily_cases', 'days_count']

# Aggregate UV index data by week
uv_weekly = uv_df.groupby('year_week').agg({
    'uv_index': ['mean', 'max', 'min']
}).reset_index()
uv_weekly.columns = ['year_week', 'avg_uv_index', 'max_uv_index', 'min_uv_index']

# Merge both datasets on year_week
merged_df = pd.merge(sunburn_weekly, uv_weekly, on='year_week', how='outer')

# Sort by date
merged_df = merged_df.sort_values('year_week').reset_index(drop=True)

# Convert year_week back to string for display
merged_df['year_week'] = merged_df['year_week'].astype(str)

# Display results
print("=" * 80)
print("WEEKLY AGGREGATED DATA - SUNBURN CASES & UV INDEX")
print("=" * 80)
print(merged_df.to_string(index=False))
print("=" * 80)

# Summary statistics
print("\nSUMMARY:")
print(f"Date range: {merged_df['year_week'].min()} to {merged_df['year_week'].max()}")
print(f"Total weeks: {len(merged_df)}")
print(f"Total sunburn cases: {merged_df['total_cases'].sum():.0f}")
print(f"Overall avg UV index: {merged_df['avg_uv_index'].mean():.2f}")

# Save to CSV
output_file = os.path.join(script_dir, "data", "weekly_aggregated_data.csv")
merged_df.to_csv(output_file, index=False)
print(f"\nData saved to: {output_file}")

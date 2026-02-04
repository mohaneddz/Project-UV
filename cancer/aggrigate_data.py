"""
UV and Cancer Data Full Aggregation Script

Aggregates daily UV data from NASA POWER API to overall averages
for Denmark, Norway, and Sweden, and merges with melanoma cancer data.
Saves all countries in one combined file with full averages (not yearly).
"""

import pandas as pd
import os


# Country files to process
COUNTRIES = ["denmark", "norway", "sweden"]

# Population data (approximate current populations)
# Source: World Bank / National statistics
POPULATIONS = {
    "Denmark": 5_932_000,    # ~5.9 million
    "Norway": 5_474_000,     # ~5.5 million
    "Sweden": 10_540_000     # ~10.5 million
}

# Columns to aggregate (UV-related parameters)
AGGREGATE_COLUMNS = [
    "ALLSKY_SFC_UV_INDEX",    # All-sky surface UV Index
    "ALLSKY_SFC_UVA",         # All-sky surface UVA irradiance
    "ALLSKY_SFC_UVB",         # All-sky surface UVB irradiance
    "ALLSKY_SFC_SW_DWN",      # All-sky surface shortwave downward irradiance
    "CLRSKY_SFC_SW_DWN",      # Clear-sky surface shortwave downward irradiance
    "TO3",                     # Total column ozone
    "ALLSKY_KT"               # All-sky clearness index
]


def load_cancer_data(data_dir: str) -> pd.DataFrame:
    """
    Load and aggregate the melanoma cancer data (total cases per country).
    
    Returns:
        DataFrame with columns: Country, Total_Cancer_Cases, Avg_Yearly_Cases, Min_Year, Max_Year
    """
    cancer_file = os.path.join(data_dir, "norway_sweden_denmark_melanoma_of_skin_numbers.csv")
    
    if not os.path.exists(cancer_file):
        print(f"❌ Cancer data file not found: {cancer_file}")
        return None
    
    # Read the CSV
    df = pd.read_csv(cancer_file)
    
    # Reshape from wide to long format
    df_melted = df.melt(id_vars=['Label'], var_name='Year', value_name='Cancer_Cases')
    df_melted = df_melted.rename(columns={'Label': 'Country'})
    
    # Convert year to int
    df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
    
    # Capitalize country names consistently
    df_melted['Country'] = df_melted['Country'].str.capitalize()
    
    # Replace '-' with NaN and convert to numeric
    df_melted['Cancer_Cases'] = pd.to_numeric(df_melted['Cancer_Cases'], errors='coerce')
    
    # Drop rows with missing cancer data
    df_melted = df_melted.dropna(subset=['Cancer_Cases', 'Year'])
    
    # Aggregate per country - total and averages
    cancer_agg = df_melted.groupby('Country').agg({
        'Cancer_Cases': ['sum', 'mean', 'std', 'min', 'max'],
        'Year': ['min', 'max', 'count']
    }).reset_index()
    
    # Flatten column names
    cancer_agg.columns = [
        'Country',
        'Total_Cancer_Cases', 'Avg_Yearly_Cancer_Cases', 'Std_Yearly_Cancer_Cases', 
        'Min_Yearly_Cancer_Cases', 'Max_Yearly_Cancer_Cases',
        'Cancer_Start_Year', 'Cancer_End_Year', 'Cancer_Years_Count'
    ]
    
    # Add population and calculate incidence per 100,000
    cancer_agg['Population'] = cancer_agg['Country'].map(POPULATIONS)
    
    # Incidence rate per 100,000 population (using average yearly cases)
    cancer_agg['Incidence_Per_100K'] = (
        cancer_agg['Avg_Yearly_Cancer_Cases'] / cancer_agg['Population']
    ) * 100_000
    
    # Round the incidence rate
    cancer_agg['Incidence_Per_100K'] = cancer_agg['Incidence_Per_100K'].round(2)
    
    return cancer_agg


def aggregate_country_full(input_file: str, country: str) -> pd.DataFrame | None:
    """
    Aggregates all daily UV data to overall averages for a single country.
    
    Args:
        input_file: Path to input CSV with daily data
        country: Country name
    
    Returns:
        pd.DataFrame with full aggregated data (one row per country)
    """
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return None
    
    # Read the daily data
    df = pd.read_csv(input_file)
    
    # Parse date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Find which columns exist in the data
    available_columns = [col for col in AGGREGATE_COLUMNS if col in df.columns]
    
    if not available_columns:
        print(f"❌ No aggregatable columns found in {input_file}")
        return None
    
    # Aggregate ALL data - calculate overall stats
    agg_dict = {col: ['mean', 'max', 'min', 'std'] for col in available_columns}
    agg_dict['Date'] = ['min', 'max', 'count']
    
    full_agg = df.agg(agg_dict)
    
    # Build result row
    result = {'Country': country.capitalize()}
    
    # Add UV stats
    for col in available_columns:
        result[f'{col}_mean'] = df[col].mean()
        result[f'{col}_max'] = df[col].max()
        result[f'{col}_min'] = df[col].min()
        result[f'{col}_std'] = df[col].std()
    
    # Add date range info
    result['UV_Start_Date'] = df['Date'].min().strftime('%Y-%m-%d')
    result['UV_End_Date'] = df['Date'].max().strftime('%Y-%m-%d')
    result['UV_Total_Days'] = len(df)
    
    # Create DataFrame
    result_df = pd.DataFrame([result])
    
    # Round numeric columns to 3 decimal places
    numeric_cols = result_df.select_dtypes(include=['float64']).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(3)
    
    return result_df


def main():
    """Main function to aggregate UV and cancer data for all countries into one file."""
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    print("\n" + "="*60)
    print("📊 UV & Cancer Data Full Aggregation (Overall Averages)")
    print("="*60)
    
    # Load and aggregate cancer data
    print("\n🔄 Loading and aggregating cancer data...")
    cancer_df = load_cancer_data(data_dir)
    
    if cancer_df is not None:
        print(f"   ✅ Loaded cancer data for {len(cancer_df)} countries")
        for _, row in cancer_df.iterrows():
            print(f"      {row['Country']}: {row['Total_Cancer_Cases']:,.0f} total cases")
    
    # Aggregate UV data for all countries
    all_uv_data = []
    
    for country in COUNTRIES:
        input_file = os.path.join(data_dir, f"{country}_uv_data.csv")
        
        print(f"\n🔄 Processing UV data for {country.capitalize()}...")
        
        df = aggregate_country_full(input_file, country)
        
        if df is not None:
            all_uv_data.append(df)
            print(f"   ✅ Aggregated {df['UV_Total_Days'].iloc[0]:,} days of data")
            print(f"   📅 Date range: {df['UV_Start_Date'].iloc[0]} to {df['UV_End_Date'].iloc[0]}")
    
    # Combine all UV data
    if all_uv_data:
        uv_combined = pd.concat(all_uv_data, ignore_index=True)
        
        # Merge UV data with cancer data
        print("\n🔄 Merging UV and cancer data...")
        
        if cancer_df is not None:
            merged_df = pd.merge(
                uv_combined,
                cancer_df,
                on='Country',
                how='left'
            )
        else:
            merged_df = uv_combined
        
        # Reorder columns
        priority_cols = ['Country', 'Total_Cancer_Cases', 'Avg_Yearly_Cancer_Cases']
        other_cols = [col for col in merged_df.columns if col not in priority_cols]
        merged_df = merged_df[[col for col in priority_cols if col in merged_df.columns] + other_cols]
        
        # Sort by country
        merged_df = merged_df.sort_values('Country').reset_index(drop=True)
        
        # Save to single file
        output_file = os.path.join(data_dir, "all_countries_full_aggregation.csv")
        merged_df.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print("📁 Combined Results (Full Averages):")
        print("="*60)
        
        for _, row in merged_df.iterrows():
            print(f"\n🌍 {row['Country']}:")
            print(f"   📊 Total Cancer Cases: {row.get('Total_Cancer_Cases', 'N/A'):,.0f}")
            print(f"   📊 Avg Yearly Cases: {row.get('Avg_Yearly_Cancer_Cases', 'N/A'):,.1f}")
            if 'ALLSKY_SFC_UV_INDEX_mean' in row:
                print(f"   ☀️  Mean UV Index: {row['ALLSKY_SFC_UV_INDEX_mean']:.3f}")
            if 'TO3_mean' in row:
                print(f"   🌫️  Mean Ozone: {row['TO3_mean']:.1f} DU")
            print(f"   📅 UV Data: {row['UV_Start_Date']} to {row['UV_End_Date']} ({row['UV_Total_Days']:,} days)")
        
        print(f"\n💾 Saved to: {output_file}")
    else:
        print("\n❌ No UV data to combine!")
    
    print("\n" + "="*60)
    print("✅ Aggregation complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

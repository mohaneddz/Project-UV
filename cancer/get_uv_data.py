"""
NASA POWER API UV Data Fetching Script for Scandinavian Countries

Fetches UV Index and related solar data from NASA's POWER API
for Denmark, Norway, and Sweden. Each country's data is saved
to a separate CSV file.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta


# Country coordinates (approximate centers)
COUNTRIES = {
    "denmark": {
        "lat": 56.0,
        "lon": 10.0,
        "name": "Denmark"
    },
    "norway": {
        "lat": 62.0,
        "lon": 10.0,
        "name": "Norway"
    },
    "sweden": {
        "lat": 62.0,
        "lon": 15.0,
        "name": "Sweden"
    }
}

# Date range (NASA POWER data starts from Jan 1, 1981)
START_DATE = "19810101"
END_DATE = "20231231"

# UV-related parameters from NASA POWER
UV_PARAMETERS = [
    "ALLSKY_SFC_UV_INDEX",    # All-sky surface UV Index
    "ALLSKY_SFC_UVA",         # All-sky surface UVA irradiance
    "ALLSKY_SFC_UVB",         # All-sky surface UVB irradiance
    "ALLSKY_SFC_SW_DWN",      # All-sky surface shortwave downward irradiance
    "CLRSKY_SFC_SW_DWN",      # Clear-sky surface shortwave downward irradiance
    "TO3",                     # Total column ozone
    "ALLSKY_KT"               # All-sky clearness index
]


def fetch_country_uv_data(
    country_code: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE
) -> pd.DataFrame | None:
    """
    Fetches UV-related data from NASA POWER API for a specific country.
    
    Args:
        country_code: Key in COUNTRIES dict ('denmark', 'norway', 'sweden')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
    
    Returns:
        pd.DataFrame or None: DataFrame with UV data if successful
    """
    
    if country_code not in COUNTRIES:
        print(f"❌ Unknown country code: {country_code}")
        return None
    
    country = COUNTRIES[country_code]
    lat = country["lat"]
    lon = country["lon"]
    name = country["name"]
    
    # Adjust start date if needed (MERRA-2 data starts Jan 1, 1981)
    if int(start_date) < 19810101:
        print(f"⚠️  Start date {start_date} is too early. Adjusting to 19810101.")
        start_date = "19810101"
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    current_dt = start_dt
    
    all_data = []
    
    print(f"\n{'='*60}")
    print(f"📍 Fetching UV data for {name}")
    print(f"   Coordinates: lat={lat}, lon={lon}")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"{'='*60}")
    
    # Fetch data in 3-year chunks to avoid API limits
    while current_dt <= end_dt:
        chunk_end_dt = current_dt + timedelta(days=365 * 3)
        if chunk_end_dt > end_dt:
            chunk_end_dt = end_dt
        
        s_str = current_dt.strftime("%Y%m%d")
        e_str = chunk_end_dt.strftime("%Y%m%d")
        
        print(f"   Fetching: {s_str} to {e_str}...", end=" ")
        
        try:
            params = {
                "parameters": ",".join(UV_PARAMETERS),
                "community": "RE",  # Renewable Energy community has UV data
                "longitude": lon,
                "latitude": lat,
                "start": s_str,
                "end": e_str,
                "format": "JSON"
            }
            
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                df_chunk = pd.DataFrame(data['properties']['parameter'])
                df_chunk.index.name = 'Date'
                df_chunk.reset_index(inplace=True)
                all_data.append(df_chunk)
                print(f"✓ ({len(df_chunk)} records)")
            else:
                print(f"✗ (HTTP {response.status_code})")
                
        except Exception as e:
            print(f"✗ (Error: {e})")
        
        current_dt = chunk_end_dt + timedelta(days=1)
        time.sleep(1.5)  # Rate limiting
    
    if not all_data:
        print(f"\n❌ No data retrieved for {name}")
        return None
    
    # Combine all chunks
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Process the data
    full_df['Date'] = pd.to_datetime(full_df['Date'], format='%Y%m%d')
    full_df.sort_values('Date', inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    
    # Replace -999 (NASA's missing value indicator) with NaN
    full_df.replace(-999, pd.NA, inplace=True)
    
    # Remove duplicate columns if any
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    
    print(f"\n✅ {name} data complete: {len(full_df)} records")
    
    return full_df


def main():
    """Main function to fetch and save UV data for all countries."""
    
    # Get the data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("🌍 NASA POWER UV Data Fetcher")
    print("   Countries: Denmark, Norway, Sweden")
    print("="*60)
    
    results = {}
    
    for country_code in COUNTRIES.keys():
        df = fetch_country_uv_data(country_code)
        
        if df is not None:
            # Save to separate CSV file
            output_file = os.path.join(data_dir, f"{country_code}_uv_data.csv")
            df.to_csv(output_file, index=False)
            print(f"   💾 Saved to: {output_file}")
            
            # Store summary statistics
            results[country_code] = {
                "file": output_file,
                "records": len(df),
                "date_range": f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                "uv_mean": df['ALLSKY_SFC_UV_INDEX'].mean() if 'ALLSKY_SFC_UV_INDEX' in df.columns else None
            }
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    
    for country_code, stats in results.items():
        country_name = COUNTRIES[country_code]["name"]
        print(f"\n{country_name}:")
        print(f"   Records: {stats['records']}")
        print(f"   Date range: {stats['date_range']}")
        if stats['uv_mean'] is not None:
            print(f"   Mean UV Index: {stats['uv_mean']:.2f}")
    
    print("\n✅ All data fetched successfully!")
    return results


if __name__ == "__main__":
    main()

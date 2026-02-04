"""
NASA POWER API Data Fetching Utility

This module fetches climate and renewable energy data from NASA's POWER API,
merging Agroclimatology (AG) and Renewable Energy (RE) communities data.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta


def fetch_nasa_data(
    lat: float = 36.75,
    lon: float = 3.05,
    start_date: str = "19800101",
    end_date: str = "20251101",
    output_filename: str = "nasa_complete_dataset.csv"
) -> pd.DataFrame | None:
    """
    Fetches data from both Agroclimatology (AG) and Renewable Energy (RE) 
    communities and merges them to get the complete dataset.
    
    Args:
        lat: Latitude coordinate (default: 36.75)
        lon: Longitude coordinate (default: 3.05)
        start_date: Start date in YYYYMMDD format (default: "19800101")
        end_date: End date in YYYYMMDD format (default: "20251101")
        output_filename: Name of the output CSV file (default: "nasa_complete_dataset.csv")
    
    Returns:
        pd.DataFrame or None: The merged dataframe if successful, None otherwise.
        Also saves the data to a CSV file in the data folder.
    """
    
    # 1. Automatic Date Correction
    # MERRA-2 data starts Jan 1, 1981
    if int(start_date) < 19810101:
        print(f"⚠️  Start date {start_date} is too early. Adjusting to 19810101 (Data Start).")
        start_date = "19810101"

    # 2. Define Separate Parameter Lists
    # Group A: Agroclimatology (Humidity, Detailed Temp, Soil)
    params_ag = [
        "T2M", "T2MDEW", "T2MWET", "TS", "T2M_RANGE", 
        "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M", "WD2M"
    ]
    
    # Group B: Renewable Energy (Solar, High Altitude Wind, UV)
    params_re = [
        "WS10M", "WD10M", "WS50M", "WD50M", 
        "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_KT", 
        "ALLSKY_SFC_LW_DWN", "ALLSKY_SFC_UV_INDEX", 
        "ALLSKY_SFC_UVA", "ALLSKY_SFC_UVB",
        "TO3"
    ]

    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    current_dt = start_dt
    
    final_dfs = []

    print(f"Starting separate downloads for AG and RE communities...")

    # 3. Chunking Loop (3 Years per chunk)
    while current_dt <= end_dt:
        chunk_end_dt = current_dt + timedelta(days=365 * 3)
        if chunk_end_dt > end_dt:
            chunk_end_dt = end_dt
            
        s_str = current_dt.strftime("%Y%m%d")
        e_str = chunk_end_dt.strftime("%Y%m%d")
        
        print(f"Processing range: {s_str} to {e_str}")

        try:
            # --- Request 1: Agroclimatology ---
            payload_ag = {
                "parameters": ",".join(params_ag),
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "start": s_str,
                "end": e_str,
                "format": "JSON"
            }
            resp_ag = requests.get(base_url, params=payload_ag)
            df_ag = None
            if resp_ag.status_code == 200:
                data_ag = resp_ag.json()
                df_ag = pd.DataFrame(data_ag['properties']['parameter'])
                df_ag.index.name = 'Date'
                df_ag.reset_index(inplace=True)
            else:
                print(f"  [AG] Failed: {resp_ag.status_code}")

            # --- Request 2: Renewable Energy ---
            payload_re = {
                "parameters": ",".join(params_re),
                "community": "RE",
                "longitude": lon,
                "latitude": lat,
                "start": s_str,
                "end": e_str,
                "format": "JSON"
            }
            resp_re = requests.get(base_url, params=payload_re)
            df_re = None
            if resp_re.status_code == 200:
                data_re = resp_re.json()
                df_re = pd.DataFrame(data_re['properties']['parameter'])
                df_re.index.name = 'Date'
                df_re.reset_index(inplace=True)
            else:
                print(f"  [RE] Failed: {resp_re.status_code}")

            # --- Merge & Clean ---
            if df_ag is not None and df_re is not None:
                merged = pd.merge(df_ag, df_re, on='Date', how='outer')
                final_dfs.append(merged)
                print(f"  -> Successfully merged {len(merged)} rows.")
            elif df_ag is not None:
                final_dfs.append(df_ag)
                print("  -> Warning: Only AG data retrieved.")
            elif df_re is not None:
                final_dfs.append(df_re)
                print("  -> Warning: Only RE data retrieved.")
                
        except Exception as e:
            print(f"  -> Critical Error: {e}")

        # Increment Date
        current_dt = chunk_end_dt + timedelta(days=1)
        time.sleep(1.5)  # Pause to respect API limits

    # 4. Final Processing
    if final_dfs:
        full_df = pd.concat(final_dfs, ignore_index=True)
        
        # Format Date
        full_df['Date'] = pd.to_datetime(full_df['Date'], format='%Y%m%d')
        full_df.sort_values('Date', inplace=True)
        
        # Handle Missing Data (-999 -> NaN)
        full_df.replace(-999, pd.NA, inplace=True)
        
        # Remove any duplicate columns if they exist
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
        
        # Determine the data folder path (relative to this file's location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(data_dir, output_filename)
        full_df.to_csv(output_path, index=False)
        print(f"\n✅ Download Complete. Data saved to: {output_path}")
        print(f"Total rows: {len(full_df)}, Total columns: {len(full_df.columns)}")
        
        return full_df
    else:
        print("❌ No data retrieved.")
        return None


if __name__ == "__main__":
    # Example usage with default parameters
    df = fetch_nasa_data()
    
    if df is not None:
        print("\nFirst 5 rows:")
        print(df.head())

"""
Ocean Data Utilities

This module handles the extraction, cleaning, and merging of Ocean Environmental Data (Temperature, Salinity, Oxygen)
and UV radiation data for ecosystem analysis.
"""

import pandas as pd
import numpy as np
import csv
from pathlib import Path
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def extract_ocean_physical_data(bbox, ocean_dir):
    """
    Robustly extracts Ocean Physical data (Temp, Salinity, Oxygen) for a given bounding box
    from ICES datasets located in ocean_dir.
    
    Args:
        bbox (dict): Dictionary with keys 'name', 'lat_min', 'lat_max', 'lon_min', 'lon_max'.
        ocean_dir (Path): Path providing the directory where ICES CSV files are stored.
        
    Returns:
        pd.DataFrame: Aggregated yearly means for ocean_temp_c_mean, ocean_salinity_mean, ocean_oxygen_mean.
    """
    print(f"Processing Ocean Physical Data for {bbox.get('name', 'Region')}...")
    
    if not isinstance(ocean_dir, Path):
        ocean_dir = Path(ocean_dir)
        
    csv_files = sorted(ocean_dir.rglob("*.csv"))
    
    # Target specific Ocean file using heuristics
    # Prioritizes files in folders named like 'icesdataportaldownload_ocean'
    target_file = next((p for p in csv_files if "icesdataportaldownload_ocean" in p.parent.name.lower()), None)
    
    if not target_file:
        print("No specific Ocean file found. Searching fallback...")
        target_file = next((p for p in csv_files if "ocean" in p.name.lower() and "biota" not in p.name.lower()), None)
        if not target_file:
            print("CRITICAL: ocean data not found.")
            return pd.DataFrame()

    print(f"Reading: {target_file.name}")
    stats = defaultdict(list)
    
    # Pass 1: Identification & Heuristics (Header & Swap Detection)
    idx_lat, idx_lon = -1, -1
    swap_coords = False
    idx_date = -1
    idx_temp, idx_sal, idx_oxy = None, None, None
    
    try:
        with open(target_file, 'r', encoding='utf-8', errors='replace') as f:
            header = None
            for line in f:
                # Skip comments or metadata lines
                if not line.lstrip().startswith('//') and ',' in line:
                    header = next(csv.reader([line]))
                    break
            
            if not header:
                return pd.DataFrame()
                
            cols = [h.lower() for h in header]
            
            # Identify columns
            try:
                # Flexible column matching
                idx_date = next(i for i, c in enumerate(cols) if 'yyyy-mm-dd' in c or 'date' in c)
                idx_lat = next(i for i, c in enumerate(cols) if 'latitude' in c)
                idx_lon = next(i for i, c in enumerate(cols) if 'longitude' in c)
                
                idx_temp = next((i for i, c in enumerate(cols) if 'temp' in c), None)
                idx_sal = next((i for i, c in enumerate(cols) if 'psal' in c or 'salinity' in c), None)
                idx_oxy = next((i for i, c in enumerate(cols) if 'doxy' in c or ('oxygen' in c and 'saturation' not in c)), None)
            except StopIteration:
                print("Error: Critical columns (Date/Lat/Lon) missing in header.")
                return pd.DataFrame()

            print(f"Columns found: Temp={idx_temp}, Sal={idx_sal}, Oxy={idx_oxy}")

            # Heuristic check for Lat/Lon swap on the first data line
            try:
                line = next(f)
                row = next(csv.reader([line]))
                val_lat = float(row[idx_lat])
                val_lon = float(row[idx_lon])
                # If lat is suspiciously low and lon suspiciously high for Nordic region, swap might be needed
                # (Simple heuristic, can be adjusted)
                if val_lat < 40 and val_lon > 40:
                    print(f"Detected swapped Lat/Lon ({val_lat}, {val_lon}). Activating swap.")
                    swap_coords = True
            except:
                pass
    except Exception as e:
        print(f"Header parsing error: {e}")
        return pd.DataFrame()

    # Pass 2: Extraction (Full Scan)
    print("Starting Extraction Pass...")
    match_count = 0
    
    with open(target_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            
            # Robust check to skip header or metadata lines encountered during scan
            if row[0].startswith('//') or 'Latitude' in row or 'latitude' in row[0].lower():
                continue
            
            try:
                if swap_coords:
                    lat = float(row[idx_lon])
                    lon = float(row[idx_lat])
                else:
                    lat = float(row[idx_lat])
                    lon = float(row[idx_lon])
                
                # Filter by Bounding Box
                if not (bbox['lat_min'] <= lat <= bbox['lat_max'] and bbox['lon_min'] <= lon <= bbox['lon_max']):
                    continue
                
                dt_str = row[idx_date]
                year = int(dt_str[:4])
                match_count += 1
                
                if idx_temp is not None and len(row) > idx_temp and row[idx_temp]:
                    stats[(year, 'temp_c')].append(float(row[idx_temp]))
                if idx_sal is not None and len(row) > idx_sal and row[idx_sal]:
                    stats[(year, 'salinity')].append(float(row[idx_sal]))
                if idx_oxy is not None and len(row) > idx_oxy and row[idx_oxy]:
                    stats[(year, 'oxygen')].append(float(row[idx_oxy]))
                    
            except (ValueError, IndexError):
                continue

    print(f"Found {match_count} matching observations.")
    
    # Aggregate results
    results = []
    years = sorted(set(k[0] for k in stats.keys()))
    
    for y in years:
        row = {'year': y}
        for var in ['temp_c', 'salinity', 'oxygen']:
            vals = stats.get((y, var), [])
            if vals:
                row[f'ocean_{var}_mean'] = np.mean(vals)
            else:
                row[f'ocean_{var}_mean'] = np.nan
        results.append(row)
        
    return pd.DataFrame(results)


def load_uv_ozone_data(countries, data_dir):
    """
    Loads and aggregates UV and Ozone data for specified countries.
    
    Args:
        countries (list): List of country names to load (e.g., ['Sweden', 'Norway']).
        data_dir (Path): Base directory containing country data (e.g., 'data/countries/uv').
        
    Returns:
        pd.DataFrame: Yearly aggregated 'uv_mean' and 'ozone_mean'.
    """
    uv_list = []
    
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    for country in countries:
        p = data_dir / f"{country}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df['country'] = country
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            
            date_col = next((c for c in df.columns if 'date' in c), None)
            if date_col:
                df['date'] = pd.to_datetime(df[date_col])
                df['year'] = df['date'].dt.year
                uv_list.append(df)
            else:
                print(f"Warning: No date column in {country}.csv")

    if uv_list:
        uv_raw = pd.concat(uv_list, ignore_index=True)
        # Aggregate regionally by year
        # Assuming typical column names 'allsky_sfc_uv_index' and 'to3' (Total Ozone)
        uv_annual = uv_raw.groupby('year').agg({
            'allsky_sfc_uv_index': 'mean',
            'to3': 'mean'
        }).reset_index()
        
        uv_annual = uv_annual.rename(columns={'allsky_sfc_uv_index': 'uv_mean', 'to3': 'ozone_mean'})
        return uv_annual
    else:
        print("Warning: No UV data found!")
        return pd.DataFrame(columns=['year', 'uv_mean', 'ozone_mean'])

def merge_ecosystem_data(uv_df, ocean_df):
    """
    Merges UV/Ozone data with Ocean data on year.
    
    Args:
        uv_df (pd.DataFrame): Yearly UV data.
        ocean_df (pd.DataFrame): Yearly Ocean data.
        
    Returns:
        pd.DataFrame: Merged dataset.
    """
    if not ocean_df.empty and not uv_df.empty:
        eco_df = pd.merge(uv_df, ocean_df, on='year', how='inner')
        print(f"Merged Ecosystem Dataset: {len(eco_df)} years of overlap.")
        return eco_df
    else:
        print("Merge failed. Missing data in one of the dataframes.")
        return pd.DataFrame()

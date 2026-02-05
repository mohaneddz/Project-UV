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


def extract_biota_skin_ulcers(ocean_dir, bbox=None):
    """
    Extracts SKIN ULC (Skin Ulcers) data from DomeBiota dataset.
    Skin ulcers in fish are indicators of bacterial infection levels in water.
    
    Hypothesis: High UV periods should correlate with lower ulcer frequency 
    due to UV disinfection of water-borne pathogens.
    
    Args:
        ocean_dir (Path): Path to the environment_ocean directory.
        bbox (dict, optional): Bounding box for geographic filtering.
        
    Returns:
        pd.DataFrame: Monthly/yearly aggregated ulcer data with columns:
                     year, month, total_fish, ulcer_positive, ulcer_rate
    """
    print("Extracting Skin Ulcer (SKIN ULC) data from DomeBiota...")
    
    if not isinstance(ocean_dir, Path):
        ocean_dir = Path(ocean_dir)
    
    # Find DomeBiota file
    biota_file = None
    for p in ocean_dir.rglob("*.csv"):
        if "domebiota" in p.name.lower() or "biota" in p.parent.name.lower():
            biota_file = p
            break
    
    if not biota_file:
        print("ERROR: DomeBiota CSV file not found.")
        return pd.DataFrame()
    
    print(f"Reading: {biota_file.name}")
    
    # Read only essential columns to reduce memory
    usecols = ['DATE', 'Latitude', 'Longitude', 'Species', 'PARAM', 'Value', 'NOINP', 'MYEAR']
    
    try:
        df = pd.read_csv(biota_file, usecols=lambda c: c in usecols, low_memory=False)
    except Exception as e:
        print(f"Error reading file: {e}")
        # Fallback: read all and select
        df = pd.read_csv(biota_file, low_memory=False)
        df = df[[c for c in usecols if c in df.columns]]
    
    print(f"Raw records loaded: {len(df):,}")
    
    # Filter for SKIN ULC parameter only
    df = df[df['PARAM'] == 'SKIN ULC'].copy()
    print(f"Records with SKIN ULC: {len(df):,}")
    
    if df.empty:
        print("No SKIN ULC data found.")
        return pd.DataFrame()
    
    # Parse dates
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['DATE'])
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    
    # Apply geographic filter if provided
    if bbox:
        df = df[(df['Latitude'] >= bbox.get('lat_min', -90)) & 
                (df['Latitude'] <= bbox.get('lat_max', 90)) &
                (df['Longitude'] >= bbox.get('lon_min', -180)) & 
                (df['Longitude'] <= bbox.get('lon_max', 180))]
        print(f"Records after geographic filter: {len(df):,}")
    
    # Value interpretation for SKIN ULC:
    # Value = 0 means no ulcer, Value > 0 means ulcer present
    # NOINP = Number of individuals in pool
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0)
    df['NOINP'] = pd.to_numeric(df['NOINP'], errors='coerce').fillna(1)
    
    # Aggregate by year and month
    # Calculate: total fish examined, fish with ulcers (Value > 0), ulcer rate
    agg_monthly = df.groupby(['year', 'month']).agg(
        total_fish=('NOINP', 'sum'),
        ulcer_positive_samples=('Value', lambda x: (x > 0).sum()),
        total_samples=('Value', 'count'),
        mean_ulcer_value=('Value', 'mean')
    ).reset_index()
    
    # Calculate ulcer rate (percentage of samples with ulcers)
    agg_monthly['ulcer_rate'] = (agg_monthly['ulcer_positive_samples'] / agg_monthly['total_samples']) * 100
    
    # Also create yearly aggregation
    agg_yearly = df.groupby('year').agg(
        total_fish=('NOINP', 'sum'),
        ulcer_positive_samples=('Value', lambda x: (x > 0).sum()),
        total_samples=('Value', 'count'),
        mean_ulcer_value=('Value', 'mean')
    ).reset_index()
    agg_yearly['ulcer_rate'] = (agg_yearly['ulcer_positive_samples'] / agg_yearly['total_samples']) * 100
    
    print(f"\nSkin Ulcer Data Summary:")
    print(f"  Years covered: {agg_yearly['year'].min()} - {agg_yearly['year'].max()}")
    print(f"  Total samples: {agg_yearly['total_samples'].sum():,}")
    print(f"  Mean ulcer rate: {agg_yearly['ulcer_rate'].mean():.2f}%")
    
    return agg_monthly, agg_yearly


def extract_seawater_contaminants(ocean_dir, param='DDEPP', bbox=None):
    """
    Extracts chemical contaminant data (e.g., DDEPP) from DomeSeawater dataset.
    
    Hypothesis: UV radiation causes photodegradation of chemical contaminants.
    High UV periods should show lower contaminant concentrations.
    
    Args:
        ocean_dir (Path): Path to the environment_ocean directory.
        param (str): Contaminant parameter code (default: 'DDEPP' - DDT derivative).
        bbox (dict, optional): Bounding box for geographic filtering.
        
    Returns:
        pd.DataFrame: Monthly/yearly aggregated contaminant data.
    """
    print(f"Extracting {param} contaminant data from DomeSeawater...")
    
    if not isinstance(ocean_dir, Path):
        ocean_dir = Path(ocean_dir)
    
    # Find DomeSeawater file
    seawater_file = None
    for p in ocean_dir.rglob("*.csv"):
        if "domeseawater" in p.name.lower() or "seawater" in p.parent.name.lower():
            seawater_file = p
            break
    
    if not seawater_file:
        print("ERROR: DomeSeawater CSV file not found.")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Reading: {seawater_file.name}")
    
    # Read only essential columns
    usecols = ['DATE', 'Latitude', 'Longitude', 'PARAM', 'Value', 'MUNIT', 'MYEAR', 'QFLAG']
    
    try:
        df = pd.read_csv(seawater_file, usecols=lambda c: c in usecols, low_memory=False)
    except Exception as e:
        print(f"Error reading file: {e}")
        df = pd.read_csv(seawater_file, low_memory=False)
        df = df[[c for c in usecols if c in df.columns]]
    
    print(f"Raw records loaded: {len(df):,}")
    
    # Filter for specified parameter
    df = df[df['PARAM'] == param].copy()
    print(f"Records with {param}: {len(df):,}")
    
    if df.empty:
        print(f"No {param} data found.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Parse dates
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['DATE'])
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    
    # Apply geographic filter if provided
    if bbox:
        df = df[(df['Latitude'] >= bbox.get('lat_min', -90)) & 
                (df['Latitude'] <= bbox.get('lat_max', 90)) &
                (df['Longitude'] >= bbox.get('lon_min', -180)) & 
                (df['Longitude'] <= bbox.get('lon_max', 180))]
        print(f"Records after geographic filter: {len(df):,}")
    
    # Handle detection limits (QFLAG contains '<' for below detection limit)
    # For below-detection values, we'll use the detection limit value divided by 2
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['below_detection'] = df['QFLAG'].str.contains('<', na=False) if 'QFLAG' in df.columns else False
    
    # Aggregate by year and month
    agg_monthly = df.groupby(['year', 'month']).agg(
        mean_concentration=('Value', 'mean'),
        max_concentration=('Value', 'max'),
        sample_count=('Value', 'count'),
        below_detection_pct=('below_detection', lambda x: x.sum() / len(x) * 100)
    ).reset_index()
    
    # Yearly aggregation
    agg_yearly = df.groupby('year').agg(
        mean_concentration=('Value', 'mean'),
        max_concentration=('Value', 'max'),
        sample_count=('Value', 'count'),
        below_detection_pct=('below_detection', lambda x: x.sum() / len(x) * 100)
    ).reset_index()
    
    print(f"\n{param} Contaminant Data Summary:")
    print(f"  Years covered: {agg_yearly['year'].min()} - {agg_yearly['year'].max()}")
    print(f"  Total samples: {agg_yearly['sample_count'].sum():,}")
    print(f"  Mean concentration: {agg_yearly['mean_concentration'].mean():.6f}")
    
    return agg_monthly, agg_yearly


def merge_uv_biological_data(uv_df, ulcer_yearly, contaminant_yearly):
    """
    Merges UV data with biological/chemical indicators for hypothesis testing.
    
    Args:
        uv_df (pd.DataFrame): Yearly UV data with 'uv_mean' column.
        ulcer_yearly (pd.DataFrame): Yearly skin ulcer data.
        contaminant_yearly (pd.DataFrame): Yearly contaminant data.
        
    Returns:
        pd.DataFrame: Merged dataset for correlation analysis.
    """
    merged = uv_df.copy()
    
    if not ulcer_yearly.empty:
        merged = pd.merge(merged, ulcer_yearly[['year', 'ulcer_rate', 'total_samples']], 
                         on='year', how='outer')
        merged = merged.rename(columns={'total_samples': 'ulcer_samples'})
    
    if not contaminant_yearly.empty:
        merged = pd.merge(merged, contaminant_yearly[['year', 'mean_concentration', 'sample_count']], 
                         on='year', how='outer')
        merged = merged.rename(columns={
            'mean_concentration': 'ddepp_concentration',
            'sample_count': 'contaminant_samples'
        })
    
    merged = merged.sort_values('year').reset_index(drop=True)
    print(f"Merged UV + Biological data: {len(merged)} years")
    
    return merged

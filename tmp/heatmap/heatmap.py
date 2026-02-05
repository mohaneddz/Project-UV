import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from datetime import datetime
import time
import os

# --- PART 1: Data Fetching (NASA POWER) ---
def fetch_nasa_uv_long_term(lat, lon, start_date, end_date):
    """
    Fetches UV Index history from start_date to end_date.
    Returns the AVERAGE UV Index over this entire period.
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    params = {
        "parameters": "ALLSKY_SFC_UV_INDEX",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            uv_dict = data['properties']['parameter']['ALLSKY_SFC_UV_INDEX']
            
            # Clean dictionary: remove NASA error codes (-999)
            valid_values = [v for k, v in uv_dict.items() if v != -999]
            
            if valid_values:
                # Calculate mean of all valid days
                return sum(valid_values) / len(valid_values)
            else:
                return None
        elif response.status_code == 429:
            print("   -> Rate Limit Hit! Waiting 60 seconds...")
            time.sleep(60)
            return fetch_nasa_uv_long_term(lat, lon, start_date, end_date) # Retry once
    except Exception as e:
        print(f"Error at {lat}, {lon}: {e}")
        
    return None

# --- PART 2: Grid & Map Generation ---
def main():
    # 1. Configuration
    # UV Data (CERES) starts ~2001. We use 20010101 to roughly present day.
    START_DATE = "20010101" 
    END_DATE = "20231231"   
    
    # Grid Size: 1.0 degree (smaller regions for better resolution)
    # Warning: Smaller grid = More API calls. 1.0 is a good balance (approx 250 requests).
    GRID_SIZE = 1.0  
    
    OUTPUT_CSV = "algeria_uv_complete_2001_2023.csv"
    
    print(f"--- UV Risk Mapping (Historical: {START_DATE}-{END_DATE}) ---")

    # 2. Get Algeria Shape
    print("Loading Algeria map...")
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    try:
        world = gpd.read_file(url)
        algeria = world[world.ADMIN == "Algeria"] 
    except Exception as e:
        print(f"Failed to load map: {e}")
        return

    # 3. Generate Grid (Square Regions)
    print(f"Generating {GRID_SIZE}x{GRID_SIZE} degree grid...")
    minx, miny, maxx, maxy = algeria.total_bounds
    squares = []
    
    lat = miny
    while lat < maxy:
        lon = minx
        while lon < maxx:
            square = Polygon([
                (lon, lat), (lon + GRID_SIZE, lat), 
                (lon + GRID_SIZE, lat + GRID_SIZE), (lon, lat + GRID_SIZE)
            ])
            
            # Intersection Check: Only keep squares touching Algeria
            if algeria.geometry.iloc[0].intersects(square):
                center = square.centroid
                squares.append({
                    'geometry': square,
                    'lat': round(center.y, 2),
                    'lon': round(center.x, 2),
                    'uv_mean': None
                })
            lon += GRID_SIZE
        lat += GRID_SIZE

    grid_gdf = gpd.GeoDataFrame(squares, crs=algeria.crs)
    print(f"Total Regions to Process: {len(grid_gdf)}")

    # 4. Fetch Data Loop
    print("\nStarting historical data download (this may take a few minutes)...")
    
    results = []
    for idx, row in grid_gdf.iterrows():
        print(f"[{idx+1}/{len(grid_gdf)}] Region {row['lat']}, {row['lon']}...", end=" ", flush=True)
        
        val = fetch_nasa_uv_long_term(row['lat'], row['lon'], START_DATE, END_DATE)
        
        if val is not None:
            results.append(val)
            print(f"Avg UV: {val:.2f}")
        else:
            results.append(0)
            print("No Data")
            
        # NASA Rate Limit safety (approx 60 requests/min allowed usually)
        time.sleep(1.5) 

    grid_gdf['uv_mean'] = results

    # 5. Save Data to CSV
    # We convert the Geometry to a string format so it fits in a CSV
    # Or just save lat/lon/value
    df_to_save = grid_gdf[['lat', 'lon', 'uv_mean']].copy()
    df_to_save.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSUCCESS: Data saved to {OUTPUT_CSV}")

    # 6. Visualization
    print("Rendering final map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    grid_gdf.plot(
        column='uv_mean', 
        ax=ax, 
        cmap='magma', # 'magma' is good for heat/UV
        edgecolor='white',
        linewidth=0.3,
        legend=True,
        legend_kwds={'label': f"Mean UV Index ({START_DATE}-{END_DATE})", 'orientation': "horizontal"}
    )
    
    algeria.boundary.plot(ax=ax, color='black', linewidth=1.5)
    
    ax.set_title(f"Algeria Long-Term UV Exposure Map\n({START_DATE} to {END_DATE})", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("algeria_historical_uv_map.png", dpi=300)
    print("Map saved as 'algeria_historical_uv_map.png'")
    plt.show()

if __name__ == "__main__":
    main()
import requests
import pandas as pd
from datetime import datetime

# England approximate center coordinates
LATITUDE = 52.5
LONGITUDE = -1.5

# Date range matching sunburn data (2024-07-07 to 2025-09-21)
START_DATE = "20240707"
END_DATE = "20250921"

# NASA POWER API endpoint
url = "https://power.larc.nasa.gov/api/temporal/daily/point"

params = {
    "parameters": "ALLSKY_SFC_UV_INDEX",
    "community": "RE",
    "longitude": LONGITUDE,
    "latitude": LATITUDE,
    "start": START_DATE,
    "end": END_DATE,
    "format": "JSON"
}

print(f"Fetching UV Index data from NASA POWER API...")
print(f"Location: England (lat={LATITUDE}, lon={LONGITUDE})")
print(f"Date range: {START_DATE} to {END_DATE}")

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    
    # Extract UV index values
    uv_data = data["properties"]["parameter"]["ALLSKY_SFC_UV_INDEX"]
    
    # Convert to DataFrame
    records = []
    for date_str, uv_value in uv_data.items():
        # Parse date from YYYYMMDD format
        date = datetime.strptime(date_str, "%Y%m%d")
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "uv_index": uv_value if uv_value != -999 else None  # -999 is NASA's missing value indicator
        })
    
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Display summary
    print(f"\n{'='*60}")
    print("Data fetched successfully!")
    print(f"{'='*60}")
    print(f"Earliest date: {df['date'].min().strftime('%Y-%m-%d')}")
    print(f"Latest date: {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Total records: {len(df)}")
    print(f"Missing values: {df['uv_index'].isna().sum()}")
    print(f"UV Index range: {df['uv_index'].min():.2f} - {df['uv_index'].max():.2f}")
    
    # Save to CSV
    output_file = "uv_index_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
else:
    print(f"Error fetching data: {response.status_code}")
    print(response.text)

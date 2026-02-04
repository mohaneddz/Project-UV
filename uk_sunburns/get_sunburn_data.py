import requests
import pandas as pd
from datetime import datetime

url = "https://api.ukhsa-dashboard.data.gov.uk/themes/climate_and_environment/sub_themes/seasonal_environmental/topics/heat-or-sunstroke/geography_types/Nation/geographies/England/metrics/heat-or-sunstroke_syndromic_emergencyDepartment_countsByDay"

all_data = []

print("Fetching data from UKHSA API...")
page_count = 0

while url:
    r = requests.get(url, params={"page_size": 365})
    j = r.json()
    
    # Extract relevant data from results
    for item in j["results"]:
        if "date" in item:
            all_data.append({
                "date": item.get("date"),
                "metric_value": item.get("metric_value", 0),
                "year": item.get("year"),
                "month": item.get("month"),
                "epiweek": item.get("epiweek")
            })
    
    page_count += 1
    print(f"Fetched page {page_count}...")
    
    url = j.get("next")

# Convert to DataFrame
df = pd.DataFrame(all_data)

if not df.empty:
    # Convert date column to datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Display date range
    earliest_date = df['date'].min()
    latest_date = df['date'].max()
    
    print(f"\n{'='*60}")
    print(f"Data fetched successfully!")
    print(f"{'='*60}")
    print(f"Earliest date: {earliest_date.strftime('%Y-%m-%d')}")
    print(f"Latest date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Total records: {len(df)}")
    print(f"Years covered: {sorted(df['year'].dropna().unique().astype(int).tolist())}")
    
    # Save to CSV
    output_file = "sunburn_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
else:
    print("No data retrieved from API.")

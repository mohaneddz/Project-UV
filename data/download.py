import requests

url = "https://api.ukhsa-dashboard.data.gov.uk/themes/climate_and_environment/sub_themes/seasonal_environmental/topics/heat-or-sunstroke/geography_types/Nation/geographies/England/metrics/heat-or-sunstroke_syndromic_emergencyDepartment_countsByDay"
years = set()

while url:
    r = requests.get(url, params={"page_size": 365})
    j = r.json()
    years |= {x["year"] for x in j["results"] if "year" in x}
    url = j["next"]

print(sorted(years))

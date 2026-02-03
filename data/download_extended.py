import csv
import requests
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

BASE_URL = (
    "https://api.ukhsa-dashboard.data.gov.uk/"
    "themes/climate_and_environment/sub_themes/seasonal_environmental/topics/heat-or-sunstroke/"
    "geography_types/Nation/geographies/England/metrics/"
    "heat-or-sunstroke_syndromic_emergencyDepartment_countsByDay"
)

OUT_CSV = "ukhsa_heat_sunstroke_ed_countsByDay.csv"
PAGE_SIZE = 365
TIMEOUT = 60

def add_params(url: str, params: dict) -> str:
    """Return url with params merged/overwritten."""
    p = urlparse(url)
    q = parse_qs(p.query)
    for k, v in params.items():
        q[k] = [str(v)]
    new_query = urlencode(q, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))

def fetch_all(start_url: str, extra_params: dict | None = None) -> list[dict]:
    url = start_url
    if extra_params:
        url = add_params(url, extra_params)

    data = []
    while url:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        data.extend(payload.get("results", []))
        url = payload.get("next")

        # ensure next pages also carry page_size/year/etc if API returns bare next
        if url and extra_params:
            url = add_params(url, extra_params)

    return data

def main():
    # Optional filters (uncomment as needed)
    # filters = {"year": 2024, "sex": "all", "age": "all", "stratum": "default"}
    filters = {}

    rows = fetch_all(BASE_URL, {"page_size": PAGE_SIZE, **filters})

    # Keep only clean rows + write minimal CSV
    out = []
    for r in rows:
        if r.get("in_reporting_delay_period") is True:
            continue
        date = r.get("date")
        val = r.get("metric_value")
        if date is None or val is None:
            continue
        out.append((date, float(val)))

    out = sorted(set(out), key=lambda x: x[0])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "case_count"])
        w.writerows(out)

    print(f"Wrote {len(out)} rows -> {OUT_CSV}")

if __name__ == "__main__":
    main()

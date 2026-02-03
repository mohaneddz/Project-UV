# ukhsa_cleanup.py
# Turns UKHSA dashboard export into a "proper" analysis CSV.
# Input can be: (1) a CSV file from UKHSA, or (2) a pasted text file like the snippet you showed.

import csv
import re
from pathlib import Path

INPUT_PATH = "ukhsa_raw.txt"          # change to your file (can also be .csv)
OUTPUT_WIDE = "ukhsa_clean_wide.csv"  # one row per date
OUTPUT_LONG = "ukhsa_clean_long.csv"  # filtered metric only (countsByDay)

# pick the main metric you want as "case_count"
TARGET_METRIC = "heat-or-sunstroke_syndromic_emergencyDepartment_countsByDay"
BASELINE_METRIC = "heat-or-sunstroke_syndromic_emergencyDepartment_baselineCountsByDay"

def read_rows(path: str):
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")

    # If it's a normal CSV file, just parse it
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
        return

    # Otherwise, parse pasted text: keep only lines that look like CSV rows
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Find header line
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("theme,sub_theme,topic,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Header not found. Expect a line starting with 'theme,sub_theme,topic,'")

    header = next(csv.reader([lines[header_idx]]))
    for ln in lines[header_idx + 1:]:
        # skip truncated/garbage lines
        if ln.count(",") < len(header) - 1:
            continue
        row = next(csv.reader([ln]))
        if len(row) != len(header):
            # try to recover: skip if malformed
            continue
        yield dict(zip(header, row))

def to_float(x: str):
    try:
        return float(x)
    except Exception:
        return None

def normalize_bool(x: str):
    x = (x or "").strip().lower()
    if x in {"true", "t", "1", "yes", "y"}:
        return True
    if x in {"false", "f", "0", "no", "n"}:
        return False
    return None

def main():
    rows = list(read_rows(INPUT_PATH))
    if not rows:
        raise ValueError("No rows parsed from input.")

    # Keep only England + all/ all + not in reporting delay (if present)
    filtered = []
    for r in rows:
        if r.get("geography") != "England":
            continue
        if r.get("sex") not in (None, "", "all"):
            continue
        if r.get("age") not in (None, "", "all"):
            continue
        delay = normalize_bool(r.get("in_reporting_delay_period", "False"))
        if delay is True:
            continue

        # normalize numeric
        r["metric_value"] = to_float(r.get("metric_value", ""))
        if r["metric_value"] is None:
            continue

        # normalize date
        d = (r.get("date") or "").strip()
        # basic YYYY-MM-DD check
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            continue

        filtered.append(r)

    # LONG: only TARGET_METRIC
    long_rows = [r for r in filtered if r.get("metric") == TARGET_METRIC]
    long_rows.sort(key=lambda r: r["date"])

    # Write long CSV: date, case_count
    with open(OUTPUT_LONG, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "case_count"])
        for r in long_rows:
            w.writerow([r["date"], r["metric_value"]])

    # WIDE: one row per date with counts + baseline
    by_date = {}
    for r in filtered:
        d = r["date"]
        by_date.setdefault(d, {"date": d, "case_count": None, "baseline_count": None})
        if r.get("metric") == TARGET_METRIC:
            by_date[d]["case_count"] = r["metric_value"]
        elif r.get("metric") == BASELINE_METRIC:
            by_date[d]["baseline_count"] = r["metric_value"]

    wide_rows = [by_date[d] for d in sorted(by_date.keys())]
    with open(OUTPUT_WIDE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "case_count", "baseline_count"])
        w.writeheader()
        w.writerows(wide_rows)

    print(f"Wrote:\n- {OUTPUT_LONG} (filtered metric only)\n- {OUTPUT_WIDE} (case_count + baseline)")

if __name__ == "__main__":
    main()

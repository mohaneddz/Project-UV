# plot.py
# Simple, analysis-ready plot for UKHSA heat/sunstroke ED data

import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "ukhsa_heat_sunstroke_ed.csv"   # date,case_count
ROLLING_WINDOW = 7                  # days

def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date")

    # 7-day rolling mean (handles zero-heavy data)
    df["case_count_7d"] = df["case_count"].rolling(
        window=ROLLING_WINDOW,
        center=True,
        min_periods=1
    ).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["case_count"], label="Daily cases")
    plt.plot(df["date"], df["case_count_7d"], label="7-day rolling mean")
    plt.xlabel("Date")
    plt.ylabel("ED attendances")
    plt.title("Heat / Sunstroke ED Attendances (UKHSA)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

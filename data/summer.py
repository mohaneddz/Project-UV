import pandas as pd
import matplotlib.pyplot as plt

# --- Load UV (daily) ---
uv = pd.read_csv("UV-ALGERIA.csv", parse_dates=["Date"])
uv = uv.rename(columns={"Date": "date"})
uv = uv[
    (uv["date"].dt.year == 2025) &
    (uv["date"].dt.month.isin([6, 7, 8]))
]
uv = uv[["date", "ALLSKY_SFC_UV_INDEX"]].rename(
    columns={"ALLSKY_SFC_UV_INDEX": "uv_index"}
)

# --- Load ED sunburns (daily) ---
ed = pd.read_csv("ED_SUNBURNS.csv", parse_dates=["date"])
ed = ed[
    (ed["date"].dt.year == 2025) &
    (ed["date"].dt.month.isin([6, 7, 8]))
]
ed = ed[["date", "case_count"]]

# --- Merge on date ---
m = pd.merge(ed, uv, on="date", how="inner").sort_values("date")

# --- Plot ---
fig, ax1 = plt.subplots()

ax1.plot(
    m["date"],
    m["case_count"],
    color="firebrick",
    linewidth=2,
    label="ED Sunburns"
)
ax1.set_xlabel("Date")
ax1.set_ylabel("ED Sunburns", color="firebrick")
ax1.tick_params(axis="y", labelcolor="firebrick")

ax2 = ax1.twinx()
ax2.plot(
    m["date"],
    m["uv_index"],
    color="darkorange",
    linewidth=2,
    alpha=0.8,
    label="UV Index"
)
ax2.set_ylabel("UV Index", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")

plt.title("Summer 2025: ED Sunburns vs UV Index")
plt.tight_layout()
plt.show()

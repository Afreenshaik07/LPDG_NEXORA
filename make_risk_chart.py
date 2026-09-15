from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis" / "outputs"

df = pd.read_csv(
    OUT / "risk_bands.csv"
)

# ------------------------------------------------------------
# Chart 2: Repair rate by risk band
# ------------------------------------------------------------

plt.figure(figsize=(9, 5.5))

plt.bar(
    df["risk_band"],
    df["repair_rate"],
)

plt.axhline(
    380 / 600,
    linestyle="--",
    linewidth=1.5,
    label="€380 visit cost / €600 missed-failure cost",
)

plt.xlabel("Relative risk band")
plt.ylabel("Observed historical repair rate")
plt.title(
    "Historical Repair Rate and Economic Visit Threshold"
)

plt.ylim(0, 1)

plt.yticks(
    [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ["0%", "20%", "40%", "60%", "80%", "100%"],
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.legend()

plt.tight_layout()

output = OUT / "02_risk_band_repair_rate.png"

plt.savefig(
    output,
    dpi=200,
)

plt.close()

print()
print("=" * 60)
print("RISK CHART CREATED")
print("=" * 60)
print(output)
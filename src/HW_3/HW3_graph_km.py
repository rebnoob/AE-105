import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# load the CSV we wrote in the previous script
df = pd.read_csv("europa_clipper_accels.csv")

# parse time
df["utc"] = pd.to_datetime(df["utc"])

# Convert all accelerations from m/s² to km/s² by dividing by 1000
acceleration_columns = [
    "a_jupiter_central", "a_jupiter_J2", "a_sun", "a_io", 
    "a_europa", "a_ganymede", "a_callisto", "a_saturn", "a_srp", "a_thrust"
]

for col in acceleration_columns:
    df[col] = df[col] / 1000.0

plt.figure(figsize=(14, 8))

# Plot acceleration sources with improved colors and styling
plt.plot(df["utc"], df["a_jupiter_central"], label="Jupiter central", linewidth=2, color='#1f77b4')
plt.plot(df["utc"], df["a_thrust"], label="Thrust", linewidth=1.5, color='#ff7f0e', linestyle='--')
plt.plot(df["utc"], df["a_sun"], label="Sun", linewidth=1.5, color='#d62728')
plt.plot(df["utc"], df["a_jupiter_J2"], label="Jupiter J2", linewidth=1.5, color='#9467bd')
plt.plot(df["utc"], df["a_europa"], label="Europa", linewidth=1.5, color='#2ca02c', alpha=0.8)
plt.plot(df["utc"], df["a_ganymede"], label="Ganymede", linewidth=1.5, color='#8c564b', alpha=0.8)
plt.plot(df["utc"], df["a_io"], label="Io", linewidth=1.5, color='#e377c2', alpha=0.8)
plt.plot(df["utc"], df["a_callisto"], label="Callisto", linewidth=1.5, color='#7f7f7f', alpha=0.8)
plt.plot(df["utc"], df["a_saturn"], label="Saturn", linewidth=1.5, color='#bcbd22')
plt.plot(df["utc"], df["a_srp"], label="SRP", linewidth=1.5, color='#17becf')

# log scale because ranges are huge
plt.yscale("log")

# Add grid for better readability on log scale
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.grid(True, which="minor", ls=":", alpha=0.2)

# Format x-axis dates
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45, ha='right')

plt.xlabel("Time (UTC)", fontsize=12, fontweight='bold')
plt.ylabel("Acceleration Magnitude (km/s²)", fontsize=12, fontweight='bold')
plt.title("Europa Clipper Acceleration Sources vs Time\n(July 20 - November 15, 2032)", 
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="best", fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.show()

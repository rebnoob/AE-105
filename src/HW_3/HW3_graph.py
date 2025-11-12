import pandas as pd
import matplotlib.pyplot as plt

# load the CSV we wrote in the previous script
df = pd.read_csv("europa_clipper_accels.csv")

# parse time
df["utc"] = pd.to_datetime(df["utc"])

plt.figure(figsize=(11, 6))

# main terms
plt.plot(df["utc"], df["a_jupiter_central"], label="Jupiter central")
plt.plot(df["utc"], df["a_thrust"], label="Thrust")
plt.plot(df["utc"], df["a_sun"], label="Sun")
plt.plot(df["utc"], df["a_jupiter_J2"], label="Jupiter J2")
plt.plot(df["utc"], df["a_europa"], label="Europa")
plt.plot(df["utc"], df["a_io"], label="Io")
plt.plot(df["utc"], df["a_ganymede"], label="Ganymede")
plt.plot(df["utc"], df["a_callisto"], label="Callisto")
plt.plot(df["utc"], df["a_saturn"], label="Saturn")
plt.plot(df["utc"], df["a_srp"], label="SRP")

# log scale because ranges are huge
plt.yscale("log")

plt.xlabel("Time (UTC)")
plt.ylabel("Acceleration magnitude (m/s²)")
plt.title("Europa Clipper acceleration sources vs time")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
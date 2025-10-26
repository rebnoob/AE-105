# pip install spiceypy numpy matplotlib pandas
import spiceypy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --------- CONFIG: update these kernel paths ---------
TLS = "/Users/donnylu/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/donnylu/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK = "/Users/donnylu/Documents/ae105/generic_kernels/spk/planets/de442.bsp"
# -----------------------------------------------------

# Toggle: use Mars system barycenter (recommended for “planet orbits”) or Mars center
USE_BARYCENTER = True
MARS_TARGET = "MARS BARYCENTER" if USE_BARYCENTER else "MARS"   # 4 vs 499
FRAME = "ECLIPJ2000"      # any inertial frame is fine for magnitudes
ABCORR = "NONE"           # geometric; not doing apparent motion for orbital mechanics
AU_KM  = 149_597_870.700  # IAU 2012 exact astronomical unit in km

def load_kernels():
    sp.kclear()
    sp.furnsh(TLS)
    sp.furnsh(PCK)
    sp.furnsh(SPK)

def daterange_utc(start_utc: str, end_utc: str, step_days=1):
    """Generate daily UTC timestamps [inclusive] from start to end."""
    start = datetime.fromisoformat(start_utc.replace("Z",""))
    end   = datetime.fromisoformat(end_utc.replace("Z",""))
    cur = start
    while cur <= end:
        yield cur.strftime("%Y-%m-%dT00:00:00")
        cur += timedelta(days=step_days)

def states_helio(target: str, ets):
    """
    Heliocentric state of target via SSB subtraction in FRAME.
    r_helio = r_target^SSB - r_sun^SSB ; same for v.
    Returns (N,3) arrays for r (km) and v (km/s).
    """
    r_list, v_list = [], []
    for et in ets:
        # target and Sun wrt SSB
        st_tgt, _ = sp.spkezr(target, et, FRAME, ABCORR, "SSB")
        st_sun, _ = sp.spkezr("SUN", et, FRAME, ABCORR, "SSB")
        r_helio = np.array(st_tgt[:3]) - np.array(st_sun[:3])
        v_helio = np.array(st_tgt[3:]) - np.array(st_sun[3:])
        r_list.append(r_helio)
        v_list.append(v_helio)
    return np.vstack(r_list), np.vstack(v_list)

def main():
    load_kernels()

    # --- build daily time grid across 2025-01-01 .. 2027-12-31
    utc_list = list(daterange_utc("2025-01-01T00:00:00", "2027-12-31T00:00:00", step_days=1))
    ets = np.array([sp.utc2et(u) for u in utc_list])

    # --- heliocentric states of Mars (barycenter by default)
    r_km, v_kms = states_helio(MARS_TARGET, ets)

    # --- magnitudes
    radius_km = np.linalg.norm(r_km, axis=1)
    speed_kms = np.linalg.norm(v_kms, axis=1)

    # --- also in AU, if you want
    radius_au = radius_km / AU_KM

    # --- save CSV
    df = pd.DataFrame({
        "utc": utc_list,
        "radius_km": radius_km,
        "radius_au": radius_au,
        "speed_km_s": speed_kms
    })
    df.to_csv("mars_radius_speed_2025_2027.csv", index=False)
    print("Saved: mars_radius_speed_2025_2027.csv")

    # --- Plot radius (AU) vs time
    plt.figure(figsize=(9,4.5))
    plt.plot(pd.to_datetime(df["utc"]), df["radius_au"])
    plt.title(f"Mars {'Barycenter' if USE_BARYCENTER else 'Center'} – Heliocentric Radius (2025–2027)")
    plt.ylabel("Radius [AU]")
    plt.xlabel("Date (UTC)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("mars_radius_2025_2027.png", dpi=180)
    print("Saved: mars_radius_2025_2027.png")
    plt.show()

    # --- Plot speed (km/s) vs time
    plt.figure(figsize=(9,4.5))
    plt.plot(pd.to_datetime(df["utc"]), df["speed_km_s"])
    plt.title(f"Mars {'Barycenter' if USE_BARYCENTER else 'Center'} – Heliocentric Speed (2025–2027)")
    plt.ylabel("Speed [km/s]")
    plt.xlabel("Date (UTC)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("mars_speed_2025_2027.png", dpi=180)
    print("Saved: mars_speed_2025_2027.png")
    plt.show()

if __name__ == "__main__":
    main()
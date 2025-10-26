# pip install spiceypy numpy matplotlib pandas
import spiceypy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------- Kernels (adjust paths) ----------
TLS = "/Users/donnylu/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/donnylu/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK = "/Users/donnylu/Documents/ae105/generic_kernels/spk/planets/de442.bsp"

FRAME = "J2000"   # inertial frame; magnitudes don't depend on frame
ABCORR = "NONE"   # geometric states for orbital mechanics

def load_kernels():
    sp.kclear()
    sp.furnsh(TLS); sp.furnsh(PCK); sp.furnsh(SPK)

def daterange_utc(start_utc: str, end_utc: str, step_days=1):
    start = datetime.fromisoformat(start_utc.replace("Z",""))
    end   = datetime.fromisoformat(end_utc.replace("Z",""))
    cur = start
    while cur <= end:
        yield cur.strftime("%Y-%m-%dT00:00:00")
        cur += timedelta(days=step_days)

def moon_wrt_earth_state(et):
    """
    Return Moon state w.r.t. Earth (position km, velocity km/s).
    Try direct SPICE call; fallback to SSB subtraction if needed.
    """
    try:
        st_me, _ = sp.spkezr("MOON", et, FRAME, ABCORR, "EARTH")
        return np.array(st_me[:3]), np.array(st_me[3:])
    except Exception:
        # Fallback: (Moon wrt SSB) - (Earth wrt SSB)
        st_moon, _ = sp.spkezr("MOON",  et, FRAME, ABCORR, "SSB")
        st_earth, _= sp.spkezr("EARTH", et, FRAME, ABCORR, "SSB")
        r = np.array(st_moon[:3]) - np.array(st_earth[:3])
        v = np.array(st_moon[3:]) - np.array(st_earth[3:])
        print("Fallback: (Moon wrt SSB) - (Earth wrt SSB)")
        return r, v

def main():
    load_kernels()

    # --- daily time grid: 2025-10-01 .. 2025-12-31
    utc_list = list(daterange_utc("2025-10-01T00:00:00", "2025-12-31T00:00:00", step_days=1))
    ets = np.array([sp.utc2et(u) for u in utc_list])

    # --- geocentric Moon states and magnitudes
    r_list, v_list = [], []
    for et in ets:
        r, v = moon_wrt_earth_state(et)
        r_list.append(r); v_list.append(v)
    r_km  = np.vstack(r_list)
    v_kms = np.vstack(v_list)

    radius_km = np.linalg.norm(r_km, axis=1)
    speed_kms = np.linalg.norm(v_kms, axis=1)

    df = pd.DataFrame({
        "utc": utc_list,
        "radius_km": radius_km,
        "speed_km_s": speed_kms
    })
    df.to_csv("moon_earth_radius_speed_2025Q4.csv", index=False)
    print("Saved: moon_earth_radius_speed_2025Q4.csv")

    # --- min/max diagnostics
    t = pd.to_datetime(df["utc"])
    i_rmin, i_rmax = radius_km.argmin(), radius_km.argmax()
    i_vmin, i_vmax = speed_kms.argmin(), speed_kms.argmax()
    print(f"Perigee-ish (min radius): {t[i_rmin]}  {radius_km[i_rmin]:,.0f} km")
    print(f"Apogee-ish (max radius): {t[i_rmax]}  {radius_km[i_rmax]:,.0f} km")
    print(f"Min speed:               {t[i_vmin]}  {speed_kms[i_vmin]:.4f} km/s")
    print(f"Max speed:               {t[i_vmax]}  {speed_kms[i_vmax]:.4f} km/s")

    # --- Plot radius vs time
    plt.figure(figsize=(9,4.5))
    plt.plot(t, radius_km)
    plt.title("Moon – Geocentric Radius (2025-10-01 to 2025-12-31)")
    plt.ylabel("Radius [km]")
    plt.xlabel("Date (UTC)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("moon_radius_2025Q4.png", dpi=180)
    print("Saved: moon_radius_2025Q4.png")
    plt.show()

    # --- Plot speed vs time
    plt.figure(figsize=(9,4.5))
    plt.plot(t, speed_kms)
    plt.title("Moon – Geocentric Speed (2025-10-01 to 2025-12-31)")
    plt.ylabel("Speed [km/s]")
    plt.xlabel("Date (UTC)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("moon_speed_2025Q4.png", dpi=180)
    print("Saved: moon_speed_2025Q4.png")
    plt.show()

if __name__ == "__main__":
    main()
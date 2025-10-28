# HW2_P3_1b.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import spiceypy as sp
from HW2_P3_1a import rv2coe  
# ---------------------------- CONFIG (edit paths) ----------------------------
TLS = "/Users/rebnoob/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/rebnoob/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK = "/Users/rebnoob/Documents/ae105/generic_kernels/spk/planets/de442.bsp"

FRAME = "ECLIPJ2000"   # inertial frame
ABCORR = "NONE"        # geometric states for orbital mechanics

# GM (km^3/s^2)
MU_SUN  = 132712440018.0     # IAU 2015 (km^3/s^2)
MU_MARS = 42828.375214       # Mars GM from NAIF (km^3/s^2)

# ---------------------------------------------------------------------------

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

def heliocentric_states(target: str, ets):
    """
    r,v of 'target' relative to Sun:  r = r_target^SSB - r_sun^SSB (same for v)
    """
    r_list, v_list = [], []
    for et in ets:
        st_tgt, _ = sp.spkezr(target, et, FRAME, ABCORR, "SSB")
        st_sun, _ = sp.spkezr("SUN",   et, FRAME, ABCORR, "SSB")
        r = np.array(st_tgt[:3]) - np.array(st_sun[:3])
        v = np.array(st_tgt[3:]) - np.array(st_sun[3:])
        r_list.append(r); v_list.append(v)
    return np.vstack(r_list), np.vstack(v_list)

def relative_states(target: str, center: str, ets):
    """
    r,v of 'target' relative to 'center' (e.g., PHOBOS wrt MARS).
    Uses direct SPICE call; will fall back to SSB subtraction if needed.
    """
    r_list, v_list = [], []
    for et in ets:
        try:
            st, _ = sp.spkezr(target, et, FRAME, ABCORR, center)
            r = np.array(st[:3]); v = np.array(st[3:])
        except Exception:
            st_tgt, _ = sp.spkezr(target, et, FRAME, ABCORR, "SSB")
            st_ctr, _ = sp.spkezr(center, et, FRAME, ABCORR, "SSB")
            r = np.array(st_tgt[:3]) - np.array(st_ctr[:3])
            v = np.array(st_tgt[3:]) - np.array(st_ctr[3:])
        r_list.append(r); v_list.append(v)
    return np.vstack(r_list), np.vstack(v_list)

# ------------------------- Elements time-series core -------------------------

def _unwrap_deg(a_deg):
    """Unwrap a 1D array of angles in degrees to make continuous curves."""
    return np.rad2deg(np.unwrap(np.deg2rad(a_deg)))

def osculating_elements_timeseries(r_arr, v_arr, mu, deg=True):
    """
    Vectorized wrapper around rv2coe for time-series r,v.

    Parameters
    ----------
    r_arr : (N,3) km
    v_arr : (N,3) km/s
    mu    : km^3/s^2
    deg   : return angles in degrees

    Returns
    -------
    DataFrame with columns: a, p, e, i, RAAN, argp, nu
    (angles unwrapped if deg=True)
    """
    rows = []
    for r, v in zip(r_arr, v_arr):
        coe = rv2coe(r, v, mu=mu, deg=True)  # keep degrees for unwrapping logic
        rows.append(coe)

    df = pd.DataFrame(rows, columns=["a","p","e","i","RAAN","argp","nu"])
    # rv2coe in part (a) used keys: 'a','p','e','i','RAAN','argp','nu'
    # Ensure correct ordering / missing keys
    for k in ["a","p","e","i","RAAN","argp","nu"]:
        if k not in df.columns:
            df[k] = np.nan

    if deg:
        # unwrap angles for nice plots
        for ang in ["i","RAAN","argp","nu"]:
            df[ang] = _unwrap_deg(df[ang].to_numpy(dtype=float))
    else:
        # convert to radians and unwrap if needed
        for ang in ["i","RAAN","argp","nu"]:
            rad = np.deg2rad(df[ang].to_numpy(dtype=float))
            df[ang] = np.unwrap(rad)
    return df

def plot_elements(df, utc_list, title_prefix):
    """
    Quick multi-panel plot of osculating elements vs time.
    Angles assumed in degrees.
    """
    t = pd.to_datetime(utc_list)

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axs = axs.ravel()

    axs[0].plot(t, df["a"]);     axs[0].set_ylabel("a [km]")
    axs[1].plot(t, df["e"]);     axs[1].set_ylabel("e [-]")
    axs[2].plot(t, df["i"]);     axs[2].set_ylabel("i [deg]")
    axs[3].plot(t, df["RAAN"]);  axs[3].set_ylabel("Ω [deg]")
    axs[4].plot(t, df["argp"]);  axs[4].set_ylabel("ω [deg]")
    axs[5].plot(t, df["nu"]);    axs[5].set_ylabel("ν [deg]")

    for ax in axs:
        ax.grid(True, alpha=0.35)

    fig.suptitle(f"{title_prefix} — Osculating Elements (Conic Fits)")
    axs[-1].set_xlabel("UTC")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --------------------- Example driver that calls everything -------------------

MU_EARTH = 398600.4418  # km^3/s^2
def moon_wrt_earth_states(ets):
    # r,v of Moon relative to Earth's center in FRAME, ABCORR
    return relative_states("MOON", "EARTH", ets)

def run_mars_and_moon_earth(start_utc="2025-10-01T00:00:00",
                            end_utc="2025-12-31T00:00:00",
                            step_days=2,
                            use_mars_barycenter=True):
    """
    Build osculating elements for:
      • Mars (heliocentric)   -> μ = MU_SUN
      • Moon wrt Earth center -> μ = MU_EARTH
    """
    load_kernels()
    utc_list = list(daterange_utc(start_utc, end_utc, step_days=step_days))
    ets = np.array([sp.utc2et(u) for u in utc_list])

    # ---- Mars heliocentric (unchanged)
    mars_target = "MARS BARYCENTER" if use_mars_barycenter else "MARS"
    r_mars, v_mars = heliocentric_states(mars_target, ets)
    df_mars = osculating_elements_timeseries(r_mars, v_mars, MU_SUN, deg=True)
    plot_elements(df_mars, utc_list, title_prefix=f"{mars_target} (heliocentric)")

    # ---- Moon relative to Earth (THIS replaces the old Phobos/Deimos block)
    r_moon, v_moon = moon_wrt_earth_states(ets)             # MOON wrt EARTH
    df_moon = osculating_elements_timeseries(r_moon, v_moon, MU_EARTH, deg=True)
    plot_elements(df_moon, utc_list, title_prefix="MOON wrt EARTH")

    return df_mars, df_moon


# ------------------------------- CLI entry -----------------------------------
if __name__ == "__main__":
    # Example: whole year of 2025, every 3 days
    run_mars_and_moon_earth(step_days=1)

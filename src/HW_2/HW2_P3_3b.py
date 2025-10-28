# porkchop_earth_mars_2035.py (revised)
# Earth→Mars porkchop with selectable ephemerides (SPICE or Kepler J2000 mean elements),
# Izzo 2015 Lambert (lamberthub), and plots for C3 or Total ΔV (v∞,dep + v∞,arr).
#
# - Two branches: short/long (single-rev, prograde)
# - Exports CSV with C3, v∞ (dep/arr), Total ΔV, TOF
# - Optional heliocentric XY preview (trajectory context)
#
# NOTE on "Total ΔV":
#   This script uses |v∞,dep| + |v∞,arr| as a mission-planning proxy, matching typical porkchop usage
#   when you don't model parking orbits/SoI capture. If you want LEO injection / capture ΔV, add those
#   models where v∞ maps to hyperbolic excess over specified circular orbits.

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spiceypy as sp
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib as mpl

# --- Lambert solver (izzo2015) ---
from lamberthub import izzo2015  # v1, v2 = izzo2015(mu, r1, r2, tof, M=0, prograde=True, low_path=True)

# ===================== CONFIG =====================
# Ephemeris selection: "SPICE" (high fidelity) or "KEPLER" (J2000 mean elements; fixed, simple conic)
EPHEMERIS_MODE = "SPICE"  # "SPICE" or "KEPLER"

# SPICE kernel paths (used when EPHEMERIS_MODE == "SPICE")
TLS = "/Users/rebnoob/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/rebnoob/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK = "/Users/rebnoob/Documents/ae105/generic_kernels/spk/planets/de442.bsp"

FRAME = "ECLIPJ2000"     # inertial ecliptic for heliocentric geometry
ABCORR = "NONE"          # geometric (good for conic transfers)
CENTER = "SSB"           # subtract Sun wrt SSB to get Sun-centered states

# Physics
MU_SUN = 1.327124400419e11  # km^3/s^2 (IAU 2015)
AU_KM  = 149_597_870.700    # km per AU
DAY_S  = 86400.0

# Date windows (inclusive)
DEP_START = "2035-01-01T00:00:00"
DEP_END   = "2035-12-31T00:00:00"
ARR_START = "2035-09-01T00:00:00"
ARR_END   = "2036-08-31T00:00:00"

# Grid step sizes
DEP_STEP_DAYS = 3
ARR_STEP_DAYS = 3

# Filters on TOF (to avoid too short/long arcs)
TOF_MIN_DAYS = 90
TOF_MAX_DAYS = 500

# Plotting choices
FILLED_CONTOUR_METRIC = "DV"  # "DV" for Total ΔV proxy (|v∞dep|+|v∞arr|) or "C3"
SHOW_ARRIVAL_VINF_OVERLAY = True
SHOW_HELIOCENTRIC_XY_PREVIEW = True

# Output
OUTDIR = Path(".")
PNG_NAME_SHORT = "porkchop_earth_to_mars_2035_short.png"
PNG_NAME_LONG  = "porkchop_earth_to_mars_2035_long.png"
CSV_NAME = "porkchop_earth_to_mars_2035.csv"
# ==================================================


# ===================== KEPLERIAN ELEMENTS (J2000) =====================
# Mean (osculating-mean) elements at J2000, suitable for simple, pedagogical propagation.
# Replace with your preferred dataset if desired.
# Elements: a[km], e[-], i[rad], Omega[rad], w[rad], M0[rad] at epoch t0 = J2000 TDB
# These are representative commonly used values; for precise work use SPICE.
J2000_ISO = "2000-01-01T12:00:00"  # J2000 epoch
def d2r(d): return np.deg2rad(d)

EARTH_MEAN = dict(
    a = 1.00000011 * AU_KM,
    e = 0.01671022,
    i = d2r(0.00005),
    Omega = d2r(-11.26064),     # longitude of ascending node
    w = d2r(102.94719),         # argument of periapsis
    L = d2r(100.46435),         # mean longitude
)
# Compute M0 = L - w - Omega
EARTH_MEAN["M0"] = EARTH_MEAN["L"] - EARTH_MEAN["w"] - EARTH_MEAN["Omega"]

MARS_MEAN = dict(
    a = 1.52366231 * AU_KM,
    e = 0.09341233,
    i = d2r(1.85061),
    Omega = d2r(49.57854),
    w = d2r(336.04084),
    L = d2r(355.45332),
)
MARS_MEAN["M0"] = MARS_MEAN["L"] - MARS_MEAN["w"] - MARS_MEAN["Omega"]
# =====================================================================


# ---------------- Utilities ----------------
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

def utc_to_et_list(utc_list):
    return np.array([sp.utc2et(u) for u in utc_list], dtype=float)

def utc_to_et(utc: str) -> float:
    return float(sp.utc2et(utc))


# ---------------- SPICE heliocentric states (barycenters) ----------------
def bary_heliocentric_rv_km(target_bary_name: str, et: float):
    """
    Sun-centered heliocentric state (r[km], v[km/s]) for a SYSTEM BARYCENTER.
    r = r_target^SSB - r_sun^SSB, v = v_target^SSB - v_sun^SSB, in FRAME.
    """
    st_tgt, _ = sp.spkezr(target_bary_name, et, FRAME, ABCORR, CENTER)
    st_sun, _ = sp.spkezr("SUN",              et, FRAME, ABCORR, CENTER)
    r = np.array(st_tgt[:3]) - np.array(st_sun[:3])
    v = np.array(st_tgt[3:]) - np.array(st_sun[3:])
    return r, v

def bary_heliocentric_xy_au(target_bary_name: str, ets):
    r_tgt = np.array([sp.spkezr(target_bary_name, et, FRAME, ABCORR, CENTER)[0][:3] for et in ets])
    r_sun = np.array([sp.spkezr("SUN",              et, FRAME, ABCORR, CENTER)[0][:3] for et in ets])
    r_helio = (r_tgt - r_sun) / AU_KM
    return r_helio[:,0], r_helio[:,1]


# ---------------- Keplerian propagator (mean elements, J2000) ----------------
def kepler_E_from_M(M, e, tol=1e-12, max_iter=50):
    """Solve Kepler's equation E - e sin E = M for E (radians)."""
    M = np.mod(M, 2*np.pi)
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E

def rot3(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[ c, s, 0],
                     [-s, c, 0],
                     [ 0, 0, 1]])

def rot1(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0],
                     [0, c,  s],
                     [0,-s,  c]])

def kepler_elements_to_rv(mu, a, e, i, Omega, w, M):
    """Return heliocentric r,v from Keplerian elements at anomaly M (all radians; a in km)."""
    E = kepler_E_from_M(M, e)
    # PQW coordinates
    cosE, sinE = np.cos(E), np.sin(E)
    r_p = a*(1 - e*cosE)
    x_p = a*(cosE - e)
    y_p = a*np.sqrt(1 - e**2)*sinE
    r_pf = np.array([x_p, y_p, 0.0])
    # velocities in PQW
    n = np.sqrt(mu/a**3)
    rdot = (a*n)/(1 - e*cosE) * (-sinE)
    rfdot = (a*n)/(1 - e*cosE) * (np.sqrt(1 - e**2)*cosE)
    v_pf = np.array([rdot, rfdot, 0.0])
    # Rotation to inertial: R3(-Omega) R1(-i) R3(-w)
    Q = rot3(-Omega) @ rot1(-i) @ rot3(-w)
    r_eci = Q @ r_pf
    v_eci = Q @ v_pf
    return r_eci, v_eci

def kepler_heliocentric_rv(body_mean, et):
    """Heliocentric r,v from fixed J2000 mean elements (simple conic)."""
    a, e, inc, Om, w, M0 = body_mean["a"], body_mean["e"], body_mean["i"], body_mean["Omega"], body_mean["w"], body_mean["M0"]
    t0 = utc_to_et(J2000_ISO)
    n = np.sqrt(MU_SUN/a**3)             # rad/s
    M = M0 + n*(et - t0)                 # rad
    return kepler_elements_to_rv(MU_SUN, a, e, inc, Om, w, M)

def kepler_xy_au(body_mean, ets):
    xy = []
    for et in ets:
        r, _ = kepler_heliocentric_rv(body_mean, et)
        xy.append(r / AU_KM)
    xy = np.array(xy)
    return xy[:,0], xy[:,1]


# ---------------- Lambert helper ----------------
def solve_lambert_both(mu_sun, r1_km, r2_km, dt_sec, prograde=True):
    """
    Try both short (low_path=True) and long (low_path=False) single-rev solutions.
    Returns dict with keys 'short' and 'long' where solutions exist, each mapping to (v1, v2).
    """
    out = {}
    for label, low_path in (("short", True), ("long", False)):
        try:
            v1_kms, v2_kms = izzo2015(mu_sun, r1_km, r2_km, dt_sec, M=0,
                                      prograde=prograde, low_path=low_path)
            out[label] = (v1_kms, v2_kms)
        except Exception:
            pass
    return out


# ---------------- Porkchop core ----------------
def planetary_rv(et, which: str):
    """
    Return heliocentric r,v for 'EARTH' or 'MARS' using the selected ephemeris mode.
    """
    if EPHEMERIS_MODE.upper() == "SPICE":
        name = "EARTH BARYCENTER" if which.upper()=="EARTH" else "MARS BARYCENTER"
        return bary_heliocentric_rv_km(name, et)
    elif EPHEMERIS_MODE.upper() == "KEPLER":
        mean = EARTH_MEAN if which.upper()=="EARTH" else MARS_MEAN
        return kepler_heliocentric_rv(mean, et)
    else:
        raise ValueError("EPHEMERIS_MODE must be 'SPICE' or 'KEPLER'")

def preview_xy(dep_utcs):
    ets_preview = utc_to_et_list(dep_utcs)
    if EPHEMERIS_MODE.upper() == "SPICE":
        ex, ey = bary_heliocentric_xy_au("EARTH BARYCENTER", ets_preview)
        mx, my = bary_heliocentric_xy_au("MARS BARYCENTER",  ets_preview)
    else:
        ex, ey = kepler_xy_au(EARTH_MEAN, ets_preview)
        mx, my = kepler_xy_au(MARS_MEAN,  ets_preview)
    plt.figure(figsize=(6.8, 6.3))
    plt.plot(ex, ey, label="Earth (heliocentric) [AU]")
    plt.plot(mx, my, label="Mars  (heliocentric) [AU]")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X [AU] (ECLIPJ2000)"); plt.ylabel("Y [AU]")
    plt.title(f"Heliocentric ecliptic XY preview — {EPHEMERIS_MODE}")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()

def build_porkchop(mu_sun: float):
    # Prepare date grids
    dep_utcs = list(daterange_utc(DEP_START, DEP_END, DEP_STEP_DAYS))
    arr_utcs = list(daterange_utc(ARR_START, ARR_END, ARR_STEP_DAYS))
    dep_ets  = utc_to_et_list(dep_utcs)
    arr_ets  = utc_to_et_list(arr_utcs)

    # Allocate arrays (rows: arrivals, cols: departures)
    shape = (len(arr_ets), len(dep_ets))
    TOF_days    = np.full(shape, np.nan)

    # Short/Long metrics
    C3_short   = np.full(shape, np.nan)
    C3_long    = np.full(shape, np.nan)
    vinfD_short = np.full(shape, np.nan)
    vinfA_short = np.full(shape, np.nan)
    vinfD_long  = np.full(shape, np.nan)
    vinfA_long  = np.full(shape, np.nan)
    DVtot_short = np.full(shape, np.nan)  # |v∞dep| + |v∞arr|
    DVtot_long  = np.full(shape, np.nan)

    # Loop over grid
    for j, et_arr in enumerate(arr_ets):
        for i, et_dep in enumerate(dep_ets):
            dt_sec = et_arr - et_dep
            if dt_sec <= 0:
                continue
            tof_days = dt_sec / DAY_S
            if not (TOF_MIN_DAYS <= tof_days <= TOF_MAX_DAYS):
                continue

            r1_km, vE_kms = planetary_rv(et_dep, "EARTH")
            r2_km, vM_kms = planetary_rv(et_arr, "MARS")
            sols = solve_lambert_both(mu_sun, r1_km, r2_km, dt_sec, prograde=True)
            if not sols:
                continue

            TOF_days[j, i] = tof_days

            # Short path
            if "short" in sols:
                v1s, v2s = sols["short"]
                dv_dep = v1s - vE_kms
                dv_arr = v2s - vM_kms
                vinfD_short[j, i] = float(np.linalg.norm(dv_dep))
                vinfA_short[j, i] = float(np.linalg.norm(dv_arr))
                C3_short[j, i]    = float(np.dot(dv_dep, dv_dep))
                DVtot_short[j, i] = vinfD_short[j, i] + vinfA_short[j, i]

            # Long path
            if "long" in sols:
                v1l, v2l = sols["long"]
                dv_dep = v1l - vE_kms
                dv_arr = v2l - vM_kms
                vinfD_long[j, i]  = float(np.linalg.norm(dv_dep))
                vinfA_long[j, i]  = float(np.linalg.norm(dv_arr))
                C3_long[j, i]     = float(np.dot(dv_dep, dv_dep))
                DVtot_long[j, i]  = vinfD_long[j, i] + vinfA_long[j, i]

    grid = {
        "dep_utcs": dep_utcs,
        "arr_utcs": arr_utcs,
        "TOF_days": TOF_days,
        "C3_short": C3_short,
        "C3_long": C3_long,
        "vinfD_short": vinfD_short,
        "vinfA_short": vinfA_short,
        "vinfD_long":  vinfD_long,
        "vinfA_long":  vinfA_long,
        "DVtot_short": DVtot_short,
        "DVtot_long":  DVtot_long,
    }
    return grid


def plot_porkchop_branch(dep_utcs, arr_utcs, Z_filled, TOF_days, vinfA=None,
                         title_suffix="", outfile="porkchop.png", metric_label=""):
    # Convert date axes
    dep_dates = pd.to_datetime(dep_utcs)
    arr_dates = pd.to_datetime(arr_utcs)
    X, Y = np.meshgrid(dep_dates, arr_dates)

    finite_Z = Z_filled[np.isfinite(Z_filled)]
    if finite_Z.size == 0:
        print(f"[{title_suffix}] No valid Lambert solutions, skipping plot.")
        return
    z5, z95 = np.percentile(finite_Z, [5, 95])
    levels = np.linspace(z5, z95, 12) if z95 > z5 else np.linspace(z5, z5 + 1.0, 12)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    pc = ax.contourf(X, Y, Z_filled, levels=levels, cmap="viridis", extend="both")
    cbar = fig.colorbar(pc, ax=ax, pad=0.02)
    cbar.set_label(metric_label)

    # TOF contours
    tof_levels = np.arange(TOF_MIN_DAYS, TOF_MAX_DAYS+1, 30)
    CS = ax.contour(X, Y, TOF_days, levels=tof_levels, colors="w", linewidths=1.0, alpha=0.9)
    ax.clabel(CS, fmt=lambda d: f"{int(d)} d", inline=True, fontsize=8)

    # Optional arrival v_inf overlay (thin lines)
    if vinfA is not None and np.isfinite(vinfA).any():
        try:
            finite_v = vinfA[np.isfinite(vinfA)]
            a5, a95 = np.percentile(finite_v, [5, 95])
            v_levels = np.linspace(a5, a95, 6) if a95 > a5 else [a5]
            CSv = ax.contour(X, Y, vinfA, levels=v_levels, colors="magenta", linewidths=0.8)
            ax.clabel(CSv, fmt=lambda x: f"{x:.1f} km/s", fontsize=7)
        except Exception:
            pass

    ax.set_title(f"Earth → Mars porkchop ({metric_label}), {title_suffix} — {EPHEMERIS_MODE}")
    ax.set_xlabel("Departure date (UTC)")
    ax.set_ylabel("Arrival date (UTC)")
    ax.grid(True, alpha=0.3)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / outfile
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved porkchop figure: {out_path}")
    plt.show()


def save_grid_csv(grid):
    dep_utcs = grid["dep_utcs"]
    arr_utcs = grid["arr_utcs"]
    rows = []
    for j, arr_utc in enumerate(arr_utcs):
        for i, dep_utc in enumerate(dep_utcs):
            rows.append({
                "dep_utc": dep_utc,
                "arr_utc": arr_utc,
                "TOF_days": grid["TOF_days"][j, i],
                "C3_short": grid["C3_short"][j, i],
                "C3_long":  grid["C3_long"][j, i],
                "vinfD_short_kms": grid["vinfD_short"][j, i],
                "vinfA_short_kms": grid["vinfA_short"][j, i],
                "vinfD_long_kms":  grid["vinfD_long"][j, i],
                "vinfA_long_kms":  grid["vinfA_long"][j, i],
                "DVtot_short_kms": grid["DVtot_short"][j, i],
                "DVtot_long_kms":  grid["DVtot_long"][j, i],
            })
    df = pd.DataFrame(rows)
    out_csv = OUTDIR / CSV_NAME
    df.to_csv(out_csv, index=False)
    print(f"Saved grid CSV: {out_csv}")
# --- NEW: line-contour (no fill) porkchop plot ------------------------

def plot_porkchop_contours(dep_utcs, arr_utcs, Z, TOF_days,
                           title_suffix="", outfile="porkchop.png",
                           metric_label=r"Total $\Delta V$ (km/s)",
                           mark_min=True,
                           cmap="turbo",
                           line_w=1.0,
                           tof_step_days=30):
    """
    Draws colored line contours (no fill) + gray TOF contours.
    Matches the classic NASA porkchop look.
    """
    # axes grids
    dep_dates = pd.to_datetime(dep_utcs)
    arr_dates = pd.to_datetime(arr_utcs)
    X, Y = np.meshgrid(dep_dates, arr_dates)

    # guard: any data?
    finite = np.isfinite(Z)
    if not finite.any():
        print(f"[{title_suffix}] No valid Lambert solutions, skipping plot.")
        return

    Zf = Z.copy()
    # choose clean contour levels
    zmin = np.nanpercentile(Zf, 5)
    zmax = np.nanpercentile(Zf, 95)
    if np.isclose(zmin, zmax):
        zmin, zmax = np.nanmin(Zf), np.nanmax(Zf)
    # round to nice values (ΔV uses km/s; C3 would be km^2/s^2)
    step = 0.5 if "ΔV" in metric_label or "Delta" in metric_label or "DV" in metric_label.upper() else (zmax - zmin)/12
    levels = np.arange(np.floor(zmin/step)*step, np.ceil(zmax/step)*step + 0.5*step, step)

    fig, ax = plt.subplots(figsize=(10.8, 7.2))

    # draw colored line contours
    cs = ax.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=line_w)
    ax.clabel(cs, fmt=lambda v: f"{v:.1f}", inline=True, fontsize=7)  # label a subset automatically

    # colorbar that reflects the line colors
    norm = mpl.colors.Normalize(vmin=levels.min(), vmax=levels.max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(metric_label)

    # TOF contours (thin gray)
    if np.isfinite(TOF_days).any():
        tof_levels = np.arange(TOF_MIN_DAYS, TOF_MAX_DAYS + 1, tof_step_days)
        cs_tof = ax.contour(X, Y, TOF_days, levels=tof_levels,
                            colors="#555555", linewidths=0.8, linestyles="solid")
        ax.clabel(cs_tof, fmt=lambda d: f"{int(d)} d", inline=True, fontsize=8)

    # mark global minimum, if desired
    if mark_min:
        j,i = np.unravel_index(np.nanargmin(Zf), Zf.shape)
        ax.plot(dep_dates[i], arr_dates[j], marker="o",
                markerfacecolor="none", markeredgecolor="magenta", markersize=8, lw=0.0)
        ax.plot(dep_dates[i], arr_dates[j], marker="o",
                markerfacecolor="magenta", markeredgecolor="magenta", markersize=3, lw=0.0)

    ax.set_title(f"Earth → Mars porkchop (line contours), {title_suffix} — {EPHEMERIS_MODE}")
    ax.set_xlabel("Launch Date (UTC)")
    ax.set_ylabel("Arrival Date (UTC)")
    ax.grid(True, alpha=0.35)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / outfile
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved porkchop figure: {out_path}")
    plt.show()


# ---------------- Main ----------------
def main():
    if EPHEMERIS_MODE.upper() == "SPICE":
        load_kernels()

    # Optional: quick preview XY in AU over the dep window
    if SHOW_HELIOCENTRIC_XY_PREVIEW:
        dep_utcs_preview = list(daterange_utc(DEP_START, DEP_END, DEP_STEP_DAYS))
        preview_xy(dep_utcs_preview)

    # Build porkchop grid
    grid = build_porkchop(MU_SUN)
    # After grid = build_porkchop(MU_SUN)
    plot_porkchop_contours(
        grid["dep_utcs"], grid["arr_utcs"],
        grid["DVtot_short"], grid["TOF_days"],
        title_suffix="short path (single-rev, prograde)",
        outfile=PNG_NAME_SHORT.replace(".png","_lines.png"),
        metric_label=r"Total $\Delta V$ (km/s)"
    )
    plot_porkchop_contours(
        grid["dep_utcs"], grid["arr_utcs"],
        grid["DVtot_long"], grid["TOF_days"],
        title_suffix="long path (single-rev, prograde)",
        outfile=PNG_NAME_LONG.replace(".png","_lines.png"),
        metric_label=r"Total $\Delta V$ (km/s)"
    )
    plot_porkchop_contours(
    grid["dep_utcs"], grid["arr_utcs"],
    grid["C3_short"], grid["TOF_days"],
    title_suffix="short path (single-rev, prograde)",
    outfile=PNG_NAME_SHORT.replace(".png","_C3_lines.png"),
    metric_label=r"$C_3$ (km$^2$/s$^2$)"
    )


    # Choose filled-contour metric
    if FILLED_CONTOUR_METRIC.upper() == "DV":
        metric_short = grid["DVtot_short"]
        metric_long  = grid["DVtot_long"]
        label = r"Total $\Delta V$ proxy  $|v_{\infty,dep}| + |v_{\infty,arr}|$  [km/s]"
    elif FILLED_CONTOUR_METRIC.upper() == "C3":
        metric_short = grid["C3_short"]
        metric_long  = grid["C3_long"]
        label = r"$C_3$  [km$^2$/s$^2$]"
    else:
        raise ValueError("FILLED_CONTOUR_METRIC must be 'DV' or 'C3'")

    # Plot both branches
    plot_porkchop_branch(
        grid["dep_utcs"], grid["arr_utcs"],
        metric_short, grid["TOF_days"],
        vinfA=(grid["vinfA_short"] if SHOW_ARRIVAL_VINF_OVERLAY else None),
        title_suffix="short path (single-rev, prograde)", outfile=PNG_NAME_SHORT,
        metric_label=label
    )
    plot_porkchop_branch(
        grid["dep_utcs"], grid["arr_utcs"],
        metric_long, grid["TOF_days"],
        vinfA=(grid["vinfA_long"] if SHOW_ARRIVAL_VINF_OVERLAY else None),
        title_suffix="long path (single-rev, prograde)", outfile=PNG_NAME_LONG,
        metric_label=label
    )

    save_grid_csv(grid)


if __name__ == "__main__":
    main()

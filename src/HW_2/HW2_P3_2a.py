# sun_mars_compare.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import spiceypy as sp

from scipy.integrate import solve_ivp

from ds_2bpSTM import ds_2bpSTM

# =============================== CONFIG ===============================
TLS = "/Users/rebnoob/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/rebnoob/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK = "/Users/rebnoob/Documents/ae105/generic_kernels/spk/planets/de442.bsp"

FRAME = "J2000"         # inertial frame
ABCORR = "NONE"         # geometric states for pure two-body comparisons
USE_MARS_BARY = True    # True  -> "MARS BARYCENTER" (ID 4)
                         # False -> "MARS" (planet center, 499)

# Sun GM (km^3/s^2)
MU_SUN = 1.327124400419e11

# Time grid
UTC_START = "2025-01-01T00:00:00"
UTC_END   = "2027-12-31T00:00:00"
STEP_DAYS = 3  # sampling cadence for the comparison

# Optional: also compare a numeric ODE flow
DO_ODE = True
ODE_METHOD = "DOP853"
TOLS = [1e-6, 1e-8]  # relative tolerances for ODE comparison

OUTDIR = Path(".")    # where to save plots/CSVs
# =====================================================================


# ---------------------------- SPICE helpers ---------------------------
def load_kernels():
    sp.kclear()
    sp.furnsh(TLS); sp.furnsh(PCK); sp.furnsh(SPK)

def utc_linspace(utc0: str, utc1: str, step_days: int):
    et0 = sp.utc2et(utc0)
    et1 = sp.utc2et(utc1)
    step = step_days * 86400.0
    N = int(np.floor((et1 - et0)/step)) + 1
    ets = et0 + np.arange(N) * step
    utcs = [sp.et2utc(et, 'C', 0) for et in ets]
    return ets, utcs

def heliocentric_state(target: str, et: float, frame=FRAME, abcorr=ABCORR):
    """Return heliocentric r,v of target: (target^SSB - sun^SSB)."""
    st_tgt, _ = sp.spkezr(target, et, frame, abcorr, "SSB")
    st_sun, _ = sp.spkezr("SUN",   et, frame, abcorr, "SSB")
    r = np.array(st_tgt[:3]) - np.array(st_sun[:3])
    v = np.array(st_tgt[3:]) - np.array(st_sun[3:])
    return r, v


# ---------------------- Universal-stepper propagation -----------------
def propagate_universal_series(et0: float, r0: np.ndarray, v0: np.ndarray,
                               ets: np.ndarray, GM: float):
    """
    Step the two-body solution across ets using ds_2bpSTM.
    Returns arrays r,v with shape (N,3).
    """
    N = len(ets)
    r = np.zeros((N,3)); v = np.zeros((N,3))
    r[0] = r0; v[0] = v0
    x = np.hstack([r0, v0])
    for k in range(1, N):
        dt = float(ets[k] - ets[k-1])  # [s]
        x, _STM = ds_2bpSTM(x, dt, GM) # one exact 2-body step
        r[k] = x[:3]; v[k] = x[3:]
    return r, v


# ----------------------------- ODE (optional) -------------------------
def two_body_ode(t, y, mu=MU_SUN):
    rx, ry, rz, vx, vy, vz = y
    r3 = (rx*rx + ry*ry + rz*rz)**1.5
    ax = -mu * rx / r3
    ay = -mu * ry / r3
    az = -mu * rz / r3
    return [vx, vy, vz, ax, ay, az]

def propagate_two_body(et0, y0, ets_eval, rtol, atol=1e-12, method=ODE_METHOD):
    """Propagate two-body ODE from et0 to ets_eval[-1]; sample at ets_eval."""
    t0, tf = 0.0, float(ets_eval[-1] - et0)
    t_eval = (ets_eval - et0).astype(float)
    sol = solve_ivp(two_body_ode, (t0, tf), y0, method=method,
                    rtol=rtol, atol=atol, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.y.T
    return Y[:, :3], Y[:, 3:]


# ------------------------------ Utilities -----------------------------
def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)


# --------------------------------- Main --------------------------------
def main():
    ensure_outdir()
    load_kernels()

    target = "MARS BARYCENTER" if USE_MARS_BARY else "MARS"
    ets, utcs = utc_linspace(UTC_START, UTC_END, STEP_DAYS)
    dates = pd.to_datetime(utcs)

    # Truth (SPICE) across grid
    r_true = np.zeros((len(ets), 3)); v_true = np.zeros((len(ets), 3))
    for i, et in enumerate(ets):
        r_true[i], v_true[i] = heliocentric_state(target, et)

    # Initial state
    et0 = ets[0]
    r0, v0 = r_true[0], v_true[0]
    print(f"Initial heliocentric state for {target} @ {UTC_START} ({FRAME}, {ABCORR})")
    print("r0 [km] =", r0)
    print("v0 [km/s] =", v0, "\n")

    # ---------- Universal-variable propagation ----------
    r_uni, v_uni = propagate_universal_series(et0, r0, v0, ets, MU_SUN)
    pos_err_uni = np.linalg.norm(r_uni - r_true, axis=1)
    vel_err_uni = np.linalg.norm(v_uni - v_true, axis=1)
    print("Universal-variable (ds_2bpSTM) vs SPICE:")
    print(f"  max|Δr| = {pos_err_uni.max():.3f} km   max|Δv| = {vel_err_uni.max():.6f} km/s")
    print(f"  RMS|Δr| = {rms(pos_err_uni):.3f} km   RMS|Δv| = {rms(vel_err_uni):.6f} km/s\n")

    # ---------- Optional ODE propagation(s) ----------
    ode_results = {}
    if DO_ODE:
        y0 = np.hstack([r0, v0])
        for tol in TOLS:
            r_ode, v_ode = propagate_two_body(et0, y0, ets, rtol=tol, atol=tol*1e-4)
            ode_results[tol] = {
                "r": r_ode, "v": v_ode,
                "pos_err": np.linalg.norm(r_ode - r_true, axis=1),
                "vel_err": np.linalg.norm(v_ode - v_true, axis=1)
            }
            print(f"ODE rtol={tol:g} vs SPICE: "
                  f"max|Δr|={ode_results[tol]['pos_err'].max():.3f} km, "
                  f"max|Δv|={ode_results[tol]['vel_err'].max():.6f} km/s")

        # Pairwise diffs between ODE flows (numerical convergence check)
        print("\nPairwise ODE flow diffs (max over time):")
        for a, b in itertools.pairwise(sorted(TOLS)):
            ra, va = ode_results[a]["r"], ode_results[a]["v"]
            rb, vb = ode_results[b]["r"], ode_results[b]["v"]
            dpos = np.linalg.norm(ra - rb, axis=1).max()
            dvel = np.linalg.norm(va - vb, axis=1).max()
            print(f"  rtol {a:g} vs {b:g}: max|Δr|={dpos:.3f} km, max|Δv|={dvel:.6e} km/s")

    # ---------- Plots ----------
    # A) Position error vs time
    plt.figure(figsize=(10,4.6))
    plt.plot(dates, pos_err_uni, 'k--', lw=2.0, label="Universal (ds_2bpSTM)")
    if DO_ODE:
        for tol in TOLS:
            plt.plot(dates, ode_results[tol]["pos_err"], label=f"ODE rtol={tol:g}")
    plt.title(f"{target} – Two-body propagation: |Δr| vs SPICE (heliocentric, {FRAME})")
    plt.ylabel("|Δr|  [km]")
    plt.xlabel("UTC")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "mars_two_body_pos_error.png", dpi=180)

    # B) Velocity error vs time
    plt.figure(figsize=(10,4.6))
    plt.plot(dates, vel_err_uni, 'k--', lw=2.0, label="Universal (ds_2bpSTM)")
    if DO_ODE:
        for tol in TOLS:
            plt.plot(dates, ode_results[tol]["vel_err"], label=f"ODE rtol={tol:g}")
    plt.title(f"{target} – Two-body propagation: |Δv| vs SPICE (heliocentric, {FRAME})")
    plt.ylabel("|Δv|  [km/s]")
    plt.xlabel("UTC")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "mars_two_body_vel_error.png", dpi=180)

    # C) XY overlay (AU) for universal vs SPICE
    AU_KM = 149_597_870.700
    r_uni_au  = r_uni / AU_KM
    r_true_au = r_true / AU_KM
    plt.figure(figsize=(7.2,7.2))
    plt.plot(r_true_au[:,0], r_true_au[:,1], 'k', lw=2.0, label="SPICE (truth)")
    plt.plot(r_uni_au[:,0],  r_uni_au[:,1],  'C0', lw=1.6, label="Universal (ds_2bpSTM)")
    plt.scatter(r_true_au[0,0],  r_true_au[0,1],  s=35, c='k', marker='o', label='Start')
    plt.scatter(r_true_au[-1,0], r_true_au[-1,1], s=35, c='k', marker='x', label='End')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{target} heliocentric XY — Universal vs SPICE (AU)")
    plt.xlabel("X [AU]"); plt.ylabel("Y [AU]")
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "mars_xy_overlay_universal_au.png", dpi=180)

    # D) Residual XY (km): Universal − SPICE
    dR_km = r_uni - r_true
    plt.figure(figsize=(6.4,6.4))
    plt.plot(dR_km[:,0], dR_km[:,1], lw=1.6, color='tab:red')
    plt.scatter(dR_km[0,0], dR_km[0,1], s=28, c='k', marker='o', label='Start')
    plt.scatter(dR_km[-1,0], dR_km[-1,1], s=28, c='k', marker='x', label='End')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Residual XY (Universal − SPICE) [km]")
    plt.xlabel("ΔX [km]"); plt.ylabel("ΔY [km]")
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "mars_xy_residual_universal_km.png", dpi=180)

    # E) CSV summary
    summary = {
        "scheme": ["Universal (ds_2bpSTM)"],
        "max_pos_km": [float(pos_err_uni.max())],
        "rms_pos_km": [rms(pos_err_uni)],
        "max_vel_kms": [float(vel_err_uni.max())],
        "rms_vel_kms": [rms(vel_err_uni)],
    }
    if DO_ODE:
        for tol in TOLS:
            pe = ode_results[tol]["pos_err"]; ve = ode_results[tol]["vel_err"]
            summary["scheme"].append(f"ODE rtol={tol:g}")
            summary["max_pos_km"].append(float(pe.max()))
            summary["rms_pos_km"].append(rms(pe))
            summary["max_vel_kms"].append(float(ve.max()))
            summary["rms_vel_kms"].append(rms(ve))
    #pd.DataFrame(summary).to_csv(OUTDIR / "mars_two_body_vs_spice_summary.csv", index=False)

    #print("\nSaved figures and CSV to", OUTDIR.resolve())


if __name__ == "__main__":
    main()

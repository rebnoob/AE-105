# Two-body Sun–Mars propagation with tolerance sweep and SPICE comparison
# Prereqs: spiceypy, scipy, numpy, matplotlib, pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.integrate import solve_ivp
import spiceypy as sp
import itertools

# -------- CONFIG: update kernel paths --------
TLS = "/Users/donnylu/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/donnylu/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK = "/Users/donnylu/Documents/ae105/generic_kernels/spk/planets/de442.bsp"

FRAME = "J2000"     # inertial frame for dynamics
ABCORR = "NONE"     # geometric states; we're comparing to geometric n-body ephemerides
USE_MARS_BARY = True  # True -> "MARS BARYCENTER" (ID 4), False -> "MARS" (499)

# Sun GM (km^3/s^2), IAU 2015 (consistent with DE ephemerides usage)
MU_SUN = 1.327124400419e11
AU_KM = 149_597_870.700  # km per AU

# Time grid for comparison (e.g., every 3 days for 3 years)
UTC_START = "2025-01-01T00:00:00"
UTC_END   = "2027-12-31T00:00:00"
STEP_DAYS = 3

# ---------- SPICE helpers ----------
def load_kernels():
    sp.kclear()
    sp.furnsh(TLS); sp.furnsh(PCK); sp.furnsh(SPK)

def utc_linspace(utc0, utc1, step_days):
    et0 = sp.utc2et(utc0)
    et1 = sp.utc2et(utc1)
    step = step_days * 86400.0
    N = int(np.floor((et1 - et0)/step)) + 1
    ets = et0 + np.arange(N) * step
    utcs = [sp.et2utc(et, 'C', 0) for et in ets]
    return ets, utcs

def heliocentric_state(target, et, frame=FRAME, abcorr=ABCORR):
    """Return heliocentric state (r, v) of target in given frame via SSB subtraction."""
    st_tgt, _ = sp.spkezr(target, et, frame, abcorr, "SSB")
    st_sun, _ = sp.spkezr("SUN",  et, frame, abcorr, "SSB")
    r = np.array(st_tgt[:3]) - np.array(st_sun[:3])
    v = np.array(st_tgt[3:]) - np.array(st_sun[3:])
    return r, v

# ---------- Two-body dynamics ----------
def two_body_ode(t, y, mu=MU_SUN):
    # y = [rx, ry, rz, vx, vy, vz]
    rx, ry, rz, vx, vy, vz = y
    r3 = (rx*rx + ry*ry + rz*rz)**1.5
    ax = -mu * rx / r3
    ay = -mu * ry / r3
    az = -mu * rz / r3
    return [vx, vy, vz, ax, ay, az]

def propagate_two_body(et0, y0, ets_eval, rtol, atol=1e-12, method="DOP853"):
    """
    Propagate the two-body equations of motion from et0 to ets_eval[-1],
    and return position/velocity arrays sampled at ets_eval.
    """
    # Time in seconds relative to start epoch
    t0 = 0.0
    tf = float(ets_eval[-1] - et0)
    t_eval = (ets_eval - et0).astype(float)

    # Integrate
    sol = solve_ivp(two_body_ode,
                    t_span=(t0, tf),
                    y0=y0,
                    method=method,
                    rtol=rtol,
                    atol=atol,
                    t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)

    Y = sol.y.T  # shape (N,6)
    r = Y[:, :3]
    v = Y[:, 3:]
    return r, v
def specific_energy(r, v, mu):
    """Two-body specific mechanical energy ε = v^2/2 - mu/|r| for each sample."""
    rnorm = np.linalg.norm(r, axis=1)
    v2 = np.sum(v*v, axis=1)
    return 0.5*v2 - mu / rnorm
    
# ---------- Main workflow ----------
def main():
    load_kernels()
    target = "MARS BARYCENTER" if USE_MARS_BARY else "MARS"

    # Build comparison timeline
    ets, utcs = utc_linspace(UTC_START, UTC_END, STEP_DAYS)

    # Initial heliocentric state at t0
    et0 = sp.utc2et(UTC_START)
    r0, v0 = heliocentric_state(target, et0, FRAME, ABCORR)
    y0 = np.hstack([r0, v0])

    # “Truth” states from SPICE on the whole grid
    r_true = np.zeros((len(ets), 3)); v_true = np.zeros((len(ets), 3))
    for i, et in enumerate(ets):
        r_true[i], v_true[i] = heliocentric_state(target, et, FRAME, ABCORR)

    # Tolerances to test (ode113-like ladder)
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    results = {}

    for tol in tolerances:
        r_prop, v_prop = propagate_two_body(et0, y0, ets, rtol=tol, atol=tol*1e-4, method="DOP853")
        pos_err = np.linalg.norm(r_prop - r_true, axis=1)              # km
        vel_err = np.linalg.norm(v_prop - v_true, axis=1)              # km/s
        results[tol] = dict(r=r_prop, v=v_prop, pos_err=pos_err, vel_err=vel_err)

    # ----- Quick text summary -----
    print(f"Initial state @ {UTC_START} ({'Mars barycenter' if USE_MARS_BARY else 'Mars center'}; {FRAME})")
    print(f"r0 [km] = {r0}")
    print(f"v0 [km/s] = {v0}\n")

    for tol in tolerances:
        pe = results[tol]["pos_err"]; ve = results[tol]["vel_err"]
        print(f"rtol={tol:>8}:  max|Δr| = {pe.max():12.3f} km   max|Δv| = {ve.max():10.6f} km/s")

    # ----- Plots: error vs time -----
    tdates = pd.to_datetime(utcs)

    plt.figure(figsize=(10,4.5))
    for tol in tolerances:
        plt.plot(tdates, results[tol]["pos_err"], label=f"rtol={tol:g}")
    plt.title(f"{'Mars Barycenter' if USE_MARS_BARY else 'Mars Center'} – Two-Body Propagation Position Error vs SPICE")
    plt.ylabel("|Δr| [km]  (heliocentric, J2000)")
    plt.xlabel("Date (UTC)")
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig("mars_two_body_pos_error.png", dpi=180)
    print("Saved: mars_two_body_pos_error.png")

    plt.figure(figsize=(10,4.5))
    for tol in tolerances:
        plt.plot(tdates, results[tol]["vel_err"], label=f"rtol={tol:g}")
    plt.title(f"{'Mars Barycenter' if USE_MARS_BARY else 'Mars Center'} – Two-Body Propagation Velocity Error vs SPICE")
    plt.ylabel("|Δv| [km/s]  (heliocentric, J2000)")
    plt.xlabel("Date (UTC)")
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig("mars_two_body_vel_error.png", dpi=180)
    print("Saved: mars_two_body_vel_error.png")

    # Optional: save a CSV of max errors for a table
        # ===== Extra analysis & visuals to compare flows and pick a RelTol =====

    # A) Pairwise differences BETWEEN flows (numerical error proxy)
    print("\nPairwise flow diffs (max over time):")
    tolerances_sorted = sorted(results.keys())
    pair_rows = []
    for a, b in itertools.pairwise(tolerances_sorted):
        ra, va = results[a]["r"], results[a]["v"]
        rb, vb = results[b]["r"], results[b]["v"]
        dpos = np.linalg.norm(ra - rb, axis=1)       # km
        dvel = np.linalg.norm(va - vb, axis=1)       # km/s
        row = dict(tol_a=a, tol_b=b, max_dpos_km=float(dpos.max()), max_dvel_kms=float(dvel.max()))
        pair_rows.append(row)
        print(f"rtol {a:g} vs {b:g}:  max|Δr| = {row['max_dpos_km']:10.3f} km   max|Δv| = {row['max_dvel_kms']:.6e} km/s")

    pd.DataFrame(pair_rows).to_csv("mars_two_body_pairwise_flow_diffs.csv", index=False)
    print("Saved: mars_two_body_pairwise_flow_diffs.csv")

    # B) Flow vs SPICE diffs (you already printed max; also save a per-tol CSV)
    per_tol_rows = []
    for tol in tolerances_sorted:
        pe = results[tol]["pos_err"]; ve = results[tol]["vel_err"]
        per_tol_rows.append(dict(rtol=tol,
                                 max_pos_km=float(pe.max()),
                                 rms_pos_km=float(np.sqrt(np.mean(pe**2))),
                                 max_vel_kms=float(ve.max()),
                                 rms_vel_kms=float(np.sqrt(np.mean(ve**2)))))
    pd.DataFrame(per_tol_rows).to_csv("mars_two_body_vs_spice_diffs.csv", index=False)
    print("Saved: mars_two_body_vs_spice_diffs.csv")

    # C) Energy drift (should be tiny if integration is converged)
    plt.figure(figsize=(10,4.5))
    for tol in tolerances_sorted:
        eps = specific_energy(results[tol]["r"], results[tol]["v"], MU_SUN)
        drift = eps - eps[0]
        plt.plot(tdates, drift, label=f"rtol={tol:g}")
    plt.title("Two-Body Specific Energy Drift (relative to start)")
    plt.ylabel("Δε [km²/s²]")
    plt.xlabel("Date (UTC)")
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig("mars_two_body_energy_drift.png", dpi=180)
    print("Saved: mars_two_body_energy_drift.png")
    plt.show()

    # E) Simple auto-recommendation for RelTol:
    # Pick the smallest tolerance for which (a) numerical diffs vs next tighter tol are small
    # and (b) SPICE error doesn't materially shrink further at tighter tol
    # Thresholds (tweak if you like):
    NUM_POS_THRESH_KM = 1.0          # flows within 1 km is "same" numerically
    NUM_VEL_THRESH_KMS = 1e-6         # km/s
    SPICE_IMPROVE_FRAC = 0.02         # <2% improvement vs next tighter tol = plateau

    # Build maps for quick lookups
    max_pos_vs_spice = {row["rtol"]: row["max_pos_km"] for row in per_tol_rows}
    rec_tol = tolerances_sorted[-1]   # fallback to tightest
    for a, b in itertools.pairwise(tolerances_sorted):  # a is looser, b is tighter
        # criterion (a): flows a vs b nearly identical
        pair = next(r for r in pair_rows if r["tol_a"]==a and r["tol_b"]==b)
        numerically_same = (pair["max_dpos_km"] <= NUM_POS_THRESH_KM) and (pair["max_dvel_kms"] <= NUM_VEL_THRESH_KMS)
        # criterion (b): tightening from a to b barely reduces SPICE error
        spa, spb = max_pos_vs_spice[a], max_pos_vs_spice[b]
        plateau = (spa - spb) <= SPICE_IMPROVE_FRAC * spa
        if numerically_same and plateau:
            rec_tol = b  # choose the tighter of the two that meets criteria
            break

    print(f"\n>>> Suggested RelTol going forward: rtol = {rec_tol:g}")
    print("    Reason: further tightening changes the two-body trajectory negligibly and does not\n"
          "    materially reduce the (dominant) model error vs SPICE.")
    
        # ----- Overlay ONE ODE flow vs SPICE in AU, plus a zoomed residual -----
    PLOT_TOL = 1e-6  # pick which flow you want to compare visually (e.g., 1e-6 or your recommended tol)
    r_ode_km = results[PLOT_TOL]["r"]
    r_spk_km = r_true

    # Convert to AU for the full-orbit overlay
    r_ode_au = r_ode_km / AU_KM
    r_spk_au = r_spk_km / AU_KM

    # (1) Full heliocentric XY overlay (in AU)
    plt.figure(figsize=(7.0, 7.0))
    plt.plot(r_spk_au[:,0], r_spk_au[:,1], 'k', lw=2.0, label='SPICE (truth)')
    plt.plot(r_ode_au[:,0], r_ode_au[:,1], color='tab:blue', lw=1.6, label=f'Two-body ODE (rtol={PLOT_TOL:g})')
    # Mark start/end
    plt.scatter(r_spk_au[0,0],  r_spk_au[0,1],  s=30, c='k', marker='o', label='Start (SPICE)')
    plt.scatter(r_spk_au[-1,0], r_spk_au[-1,1], s=30, c='k', marker='x', label='End (SPICE)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Mars heliocentric XY: Two-body vs SPICE (AU)')
    plt.xlabel('X [AU] (J2000)'); plt.ylabel('Y [AU] (J2000)')
    plt.grid(True, alpha=0.4); plt.legend(loc='best')
    plt.tight_layout(); plt.savefig('mars_xy_overlay_au.png', dpi=180)
    print('Saved: mars_xy_overlay_au.png')
    plt.show()

        # ===== Daily SPICE positions (every day) and plot =====
    ets_daily, utcs_daily = utc_linspace(UTC_START, UTC_END, 1)  # 1 day step
    r_spk_daily_km = np.array([heliocentric_state(target, et, FRAME, ABCORR)[0] for et in ets_daily])
    v_spk_daily_kms = np.array([heliocentric_state(target, et, FRAME, ABCORR)[1] for et in ets_daily])

        # ===== ODE at the same daily epochs + overlay with SPICE =====
    PLOT_TOL = 1e-6  # choose which ODE tolerance to visualize
    # Reuse the same initial state and integrate with t_eval at daily times
    r_ode_daily_km, v_ode_daily_kms = propagate_two_body(
        et0, y0, ets_daily, rtol=PLOT_TOL, atol=PLOT_TOL*1e-4, method="DOP853"
    )

    # (A) Overlay SPICE daily vs ODE daily in AU
    r_spk_daily_au  = r_spk_daily_km  / AU_KM
    r_ode_daily_au  = r_ode_daily_km  / AU_KM

    plt.figure(figsize=(7.2,7.2))
    # SPICE daily as points+thin line
    plt.plot(r_spk_daily_au[:,0], r_spk_daily_au[:,1], 'k-', lw=1.0, alpha=0.7, label="SPICE (daily)")
    plt.scatter(r_spk_daily_au[:,0], r_spk_daily_au[:,1], s=10, c='k', alpha=0.4)
    # ODE daily as a colored line + markers
    plt.plot(r_ode_daily_au[:,0], r_ode_daily_au[:,1], lw=1.6, label=f"Two-body ODE (rtol={PLOT_TOL:g})")
    plt.scatter(r_ode_daily_au[:,0], r_ode_daily_au[:,1], s=10, alpha=0.4)
    # start/end markers
    plt.scatter(r_spk_daily_au[0,0],  r_spk_daily_au[0,1],  s=35, c='tab:green', marker='o', label='Start')
    plt.scatter(r_spk_daily_au[-1,0], r_spk_daily_au[-1,1], s=35, c='tab:red',   marker='x', label='End')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Mars heliocentric XY – SPICE (daily) vs Two-body ODE (daily) [AU]")
    plt.xlabel("X [AU] (J2000)"); plt.ylabel("Y [AU] (J2000)")
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig("mars_xy_spice_vs_ode_daily_au.png", dpi=180)
    print("Saved: mars_xy_spice_vs_ode_daily_au.png")
    plt.show()

    # (B) Residual XY at daily epochs (ODE − SPICE) in km — makes the difference visible
    dR_daily_km = r_ode_daily_km - r_spk_daily_km
    plt.figure(figsize=(6.6,6.6))
    plt.plot(dR_daily_km[:,0], dR_daily_km[:,1], lw=1.6)
    plt.scatter(dR_daily_km[0,0],  dR_daily_km[0,1],  s=30, c='k', marker='o', label='Start')
    plt.scatter(dR_daily_km[-1,0], dR_daily_km[-1,1], s=30, c='k', marker='x', label='End')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Residual XY (Two-body − SPICE) at daily epochs, rtol={PLOT_TOL:g} [km]")
    plt.xlabel("ΔX [km]"); plt.ylabel("ΔY [km]")
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig("mars_xy_residual_daily_km.png", dpi=180)
    print("Saved: mars_xy_residual_daily_km.png")
    plt.show()

    # (2) Residual trajectory in XY (in km): r_ode - r_spice
    dR_km = r_ode_km - r_spk_km
    plt.figure(figsize=(6.5, 6.5))
    plt.plot(dR_km[:,0], dR_km[:,1], lw=1.6, color='tab:red')
    plt.scatter(dR_km[0,0],  dR_km[0,1],  s=25, c='k', marker='o', label='Start')
    plt.scatter(dR_km[-1,0], dR_km[-1,1], s=25, c='k', marker='x', label='End')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Residual XY (Two-body − SPICE), rtol={PLOT_TOL:g}')
    plt.xlabel('ΔX [km]'); plt.ylabel('ΔY [km]')
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig('mars_xy_residual_km.png', dpi=180)
    print('Saved: mars_xy_residual_km.png')
    plt.show()

    # Optional: zoom last ~60 days (helps show separation near the end)
    days_to_zoom = 60
    n_last = max(2, int(np.ceil(days_to_zoom / STEP_DAYS)))
    idx = slice(-n_last, None)
    plt.figure(figsize=(6.5, 6.5))
    plt.plot(r_spk_km[idx,0], r_spk_km[idx,1], 'k', lw=2.0, label='SPICE (last ~60d)')
    plt.plot(r_ode_km[idx,0], r_ode_km[idx,1], color='tab:blue', lw=1.6, label=f'Two-body (last ~60d, rtol={PLOT_TOL:g})')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Zoomed heliocentric XY (last ~60 days, km)')
    plt.xlabel('X [km] (J2000)'); plt.ylabel('Y [km] (J2000)')
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig('mars_xy_zoom_last60d_km.png', dpi=180)
    print('Saved: mars_xy_zoom_last60d_km.png')
    plt.show()



if __name__ == "__main__":
    main()
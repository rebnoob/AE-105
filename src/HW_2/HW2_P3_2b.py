# check_symplectic.py
# Verify that the two-body step map from ds_2bpSTM is symplectic.

from __future__ import annotations
import numpy as np
import spiceypy as sp
from pathlib import Path
from ds_2bpSTM import ds_2bpSTM

# ------------- CONFIG (match your setup) -------------
TLS = "/Users/rebnoob/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/rebnoob/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK = "/Users/rebnoob/Documents/ae105/generic_kernels/spk/planets/de442.bsp"

FRAME = "J2000"
ABCORR = "NONE"
TARGET = "MARS BARYCENTER"  # or "MARS"
MU_SUN = 1.327124400419e11  # km^3/s^2

UTC0 = "2025-01-01T00:00:00"
DT_SEC = 1 * 86400.0        # one step (e.g., 3 days)
N_STEPS = 120               # number of checks along the trajectory
# -----------------------------------------------------

def load_kernels():
    sp.kclear()
    sp.furnsh(TLS); sp.furnsh(PCK); sp.furnsh(SPK)

def heliocentric_state(target: str, et: float):
    st_tgt, _ = sp.spkezr(target, et, FRAME, ABCORR, "SSB")
    st_sun, _ = sp.spkezr("SUN",   et, FRAME, ABCORR, "SSB")
    r = np.array(st_tgt[:3]) - np.array(st_sun[:3])
    v = np.array(st_tgt[3:]) - np.array(st_sun[3:])
    return r, v

def is_symplectic(Phi: np.ndarray, rtol=1e-9, atol=1e-12):
    """Return residual norms for ΦᵀJΦ - J and |det Φ - 1|."""
    I = np.eye(3)
    J = np.block([[np.zeros((3,3)),  I],
                  [-I,               np.zeros((3,3))]])
    R = Phi.T @ J @ Phi - J
    res_norm = np.linalg.norm(R, ord=np.inf)
    det_dev  = abs(np.linalg.det(Phi) - 1.0)
    return res_norm, det_dev

def main():
    load_kernels()
    et0 = sp.utc2et(UTC0)
    r0, v0 = heliocentric_state(TARGET, et0)
    x = np.hstack([r0, v0])

    # Per-step symplecticity and cumulative product (composition) check
    max_res, max_detdev = 0.0, 0.0
    max_res_cum, max_detdev_cum = 0.0, 0.0

    Phi_cum = np.eye(6)  # product of STMs across steps
    et = et0

    for k in range(N_STEPS):
        # one exact two-body step (ds_2bpSTM returns dimensional STM)
        x1, Phi = ds_2bpSTM(x, DT_SEC, MU_SUN)

        # per-step symplectic check
        res, detdev = is_symplectic(Phi)
        max_res = max(max_res, res)
        max_detdev = max(max_detdev, detdev)

        # accumulate STM and check composition remains symplectic
        Phi_cum = Phi @ Phi_cum
        res_c, detdev_c = is_symplectic(Phi_cum)
        max_res_cum = max(max_res_cum, res_c)
        max_detdev_cum = max(max_detdev_cum, detdev_c)

        # advance
        x = x1
        et += DT_SEC

    print(f"Symplecticity check for {TARGET} two-body flow using ds_2bpSTM")
    print(f"  Steps: {N_STEPS},  dt = {DT_SEC/86400.0:.1f} days")
    print(f"Per-step Φ:")
    print(f"  ||Φᵀ J Φ − J||_∞   = {max_res:.3e}")
    print(f"  |det Φ − 1|        = {max_detdev:.3e}")
    print(f"Cumulative Φ (product over steps):")
    print(f"  ||Φᵀ J Φ − J||_∞   = {max_res_cum:.3e}")
    print(f"  |det Φ − 1|        = {max_detdev_cum:.3e}")

    # Optional: also verify J-inverse identity numerically
    # J^{-1} = -J for canonical J; check Φ^{-1} ≈ -J Φᵀ J
    I = np.eye(3)
    J = np.block([[np.zeros((3,3)),  I],
                  [-I,               np.zeros((3,3))]])
    Phi_inv_from_J = -J @ Phi_cum.T @ J
    rel_inv_err = np.linalg.norm(np.linalg.inv(Phi_cum) - Phi_inv_from_J, ord=np.inf) / np.linalg.norm(Phi_inv_from_J, ord=np.inf)
    print(f"Inverse identity check (Φ⁻¹ vs −J Φᵀ J): rel. ∞-norm error = {rel_inv_err:.3e}")

if __name__ == "__main__":
    main()

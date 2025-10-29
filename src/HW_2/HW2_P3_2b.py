# HW2_P3_2b.py
# Symplectic check with longdouble accumulation, SciPy linalg, and heatmaps.

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import spiceypy as sp
import scipy.linalg as la 

from ds_2bpSTM import ds_2bpSTM 

# ============================= CONFIG =============================
TLS = "/Users/rebnoob/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/rebnoob/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK = "/Users/rebnoob/Documents/ae105/generic_kernels/spk/planets/de442.bsp"

FRAME, ABCORR = "J2000", "NONE"
TARGET = "MARS BARYCENTER"      
MU_SUN = 1.327124400419e11      # km^3/s^2

UTC0   = "2025-01-01T00:00:00"
DT_SEC = 86400.0                # 1 day
N_STEPS = 120

USE_LONGDOUBLE = True           # accumulate Phi in extended precision
RESYMPLECTIFY_EVERY = 0        # set None/0 to disable
OUTDIR = Path(".")
SHOW_PLOTS = False
# ==================================================================

def load_kernels():
    sp.kclear(); sp.furnsh(TLS); sp.furnsh(PCK); sp.furnsh(SPK)

def heliocentric_state(target: str, et: float):
    st_tgt, _ = sp.spkezr(target, et, FRAME, ABCORR, "SSB")
    st_sun, _ = sp.spkezr("SUN",   et, FRAME, ABCORR, "SSB")
    r = np.array(st_tgt[:3]) - np.array(st_sun[:3])
    v = np.array(st_tgt[3:]) - np.array(st_sun[3:])
    return r, v

def J_canonical(dtype=float):
    I = np.eye(3, dtype=dtype); Z = np.zeros((3,3), dtype=dtype)
    return np.block([[Z, I], [-I, Z]])

# ---------- float64-safe conversions ----------
def as64(A):  # contiguous float64 view/copy
    return np.asarray(A, dtype=np.float64, order="C")

# ---------- SciPy linalg wrappers ----------
def det64(A):
    return float(la.det(as64(A), check_finite=False))

def solve64(A, B):
    return la.solve(as64(A), as64(B), assume_a="gen", check_finite=False)

# ------------------------- symplectic utilities -------------------------
def symplectic_residual(Phi: np.ndarray):
    """R = Phi^T J Phi - J"""
    J = J_canonical(dtype=Phi.dtype)
    return Phi.T @ J @ Phi - J

def res_norm_inf(R: np.ndarray) -> float:
    return float(np.linalg.norm(R, ord=np.inf))


def symplectic_project(M_in):
    """
    1) Try symplectic polar projection with SPD K = (-J) M^T J M.
    2) If sqrtm/conditioning is sketchy, fall back to a first-order Cayley update.

    All heavy linalg in float64 for robustness, then cast back to M_in.dtype.
    """
    dtype_out = M_in.dtype
    M = np.asarray(M_in, dtype=np.float64, order="C")
    J = J_canonical(dtype=np.float64)

    # K should be SPD if M is close to symplectic
    K = (-J) @ (M.T @ J @ M)
    # Symmetrize to damp round-off; tiny negative eigs will be clipped by sqrtm
    K = 0.5 * (K + K.T)
    try:
        P = la.sqrtm(K)
        if np.iscomplexobj(P):
            # discard tiny imaginary noise
            if np.max(np.abs(P.imag)) > 1e-8:
                raise ValueError("sqrtm produced large imaginary part")
            P = P.real
        # Solve S = M @ P^{-1} stably via linear solve
        S = la.solve(P, M.T, assume_a="pos", check_finite=False).T  # S = M @ inv(P)
        # Small clean-up: symmetrize the residual once
        R = S.T @ J @ S - J
        if np.linalg.norm(R, ord=np.inf) < 1e-6:
            return S.astype(dtype_out, copy=False)
        # else continue to fallback for extra safety
    except Exception:
        pass

    # E = M^T J M - J ; want S so that S^T J S = J to first order.
    E = M.T @ J @ M - J
    # Build correction C = (I - 0.5 J^{-1}E)^{-1}(I + 0.5 J^{-1}E)
    # Since J^{-1} = -J:
    A = np.eye(6) + 0.5 * (-J) @ E
    B = np.eye(6) - 0.5 * (-J) @ E
    # Solve for C: A C = B  (avoid explicit inverse)
    C = la.solve(A, B, assume_a="gen", check_finite=False)
    S = M @ C
    return S.astype(dtype_out, copy=False)

# ------------------------------- plotting -------------------------------
def heatmap_R(R: np.ndarray, title: str, fname: Path):
    plt.figure(figsize=(5.2, 4.6))
    im = plt.imshow(R, cmap="coolwarm", interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=r"$\Phi^\mathsf{T}J\Phi - J$")
    plt.xticks(range(6)); plt.yticks(range(6))
    plt.tight_layout(); plt.savefig(fname, dpi=180)
    if SHOW_PLOTS: plt.show()
    plt.close()

# --------------------------------- main ---------------------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    load_kernels()

    et = sp.utc2et(UTC0)
    r0, v0 = heliocentric_state(TARGET, et)
    x = np.hstack([r0, v0])

    Phi_dtype = np.longdouble if USE_LONGDOUBLE else np.float64
    Phi_cum = np.eye(6, dtype=Phi_dtype)

    first_step_R = None
    max_step_def = 0.0
    snapshots = []

    snap_idx = set(np.linspace(1, N_STEPS, 6, dtype=int)) if N_STEPS >= 6 else {N_STEPS}

    for k in range(1, N_STEPS+1):
        x1, Phi_k = ds_2bpSTM(x, DT_SEC, MU_SUN)  # Phi_k is float64

        # per-step residual
        Rk = symplectic_residual(Phi_k)
        max_step_def = max(max_step_def, res_norm_inf(Rk))
        if first_step_R is None:
            first_step_R = Rk.copy()

        # accumulate
        if USE_LONGDOUBLE:
            Phi_cum = Phi_k.astype(np.longdouble) @ Phi_cum
        else:
            Phi_cum = Phi_k @ Phi_cum

        # optional re-symplectify (via solve, no inv)
        if RESYMPLECTIFY_EVERY and (k % RESYMPLECTIFY_EVERY == 0):
            Phi_cum = symplectic_project(Phi_cum)

        # snapshots
        if (k in snap_idx) or (k == N_STEPS):
            Rc = symplectic_residual(as64(Phi_cum))
            snapshots.append((k, Rc))

        x = x1; et += DT_SEC

    # final cumulative residual + determinant deviation
    Phi_cum64 = as64(Phi_cum)
    Rc_final = symplectic_residual(Phi_cum64)
    cum_def = res_norm_inf(Rc_final)
    det_dev = abs(det64(Phi_cum64) - 1.0)

    print(f"Symplecticity check for {TARGET} (ds_2bpSTM)")
    print(f"  steps = {N_STEPS}, dt = {DT_SEC/86400.0:.2f} d")
    print(f"  accumulate longdouble: {USE_LONGDOUBLE}")
    print(f"  re-symplectify every : {RESYMPLECTIFY_EVERY if RESYMPLECTIFY_EVERY else 'disabled'}")
    print(f"Per-step max ||ΦᵀJΦ−J||_∞ = {max_step_def:.3e}")
    print(f"Cumulative  ||ΦᵀJΦ−J||_∞ = {cum_def:.3e}")
    print(f"Cumulative  |det Φ − 1|   = {det_dev:.3e}")

    # heat maps: exactly the condition Φ^T J Φ = J
    heatmap_R(first_step_R, "Per-step residual  R₁ = Φ₁ᵀJΦ₁ − J",
              OUTDIR/"stm_residual_step1.png")
    heatmap_R(Rc_final, "Cumulative residual  R = ΦᵀJΦ − J  (t₀→t_N)",
              OUTDIR/"stm_residual_cumulative.png")
    for k, Rc in snapshots:
        heatmap_R(Rc, f"Residual at step {k}/{N_STEPS}",
                  OUTDIR/f"stm_residual_step{k:04d}.png")

if __name__ == "__main__":
    main()
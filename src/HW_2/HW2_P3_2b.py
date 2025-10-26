# test_symplectic_ds2bpstm.py
# Verify that ds_2bpSTM's STM is (numerically) symplectic.
# Prereqs: numpy; optional spiceypy for the Sun–Mars state

from __future__ import annotations
import numpy as np

# ---- import your propagator (Python port of ds_2bpSTM.m)
from ds_2bpSTM import ds_2bpSTM

# ---- (Optional) SPICE helpers (comment out if you don’t want SPICE) ----
USE_SPICE = True
if USE_SPICE:
    import spiceypy as sp

    # UPDATE these paths for your machine:
    TLS = "/Users/rebnoob/Documents/ae105/generic_kernels/lsk/naif0012.tls"
    PCK = "/Users/rebnoob/Documents/ae105/generic_kernels/pck/pck00010.tpc"
    SPK = "/Users/rebnoob/Documents/ae105/generic_kernels/spk/planets/de442.bsp"

    FRAME  = "J2000"
    ABCORR = "NONE"

    def load_kernels():
        sp.kclear()
        sp.furnsh(TLS); sp.furnsh(PCK); sp.furnsh(SPK)

    def heliocentric_state(target: str, et: float):
        st_tgt, _ = sp.spkezr(target, et, FRAME, ABCORR, "SSB")
        st_sun, _ = sp.spkezr("SUN",   et, FRAME, ABCORR, "SSB")
        r = np.array(st_tgt[:3]) - np.array(st_sun[:3])
        v = np.array(st_tgt[3:]) - np.array(st_sun[3:])
        return r, v

# ---- Physics constants (km, s)
MU_SUN   = 1.327124400419e11   # km^3/s^2
MU_EARTH = 3.986004418e5       # km^3/s^2

# ---- Symplectic utilities
def J6():
    I3 = np.eye(3)
    return np.block([[np.zeros((3,3)),  I3],
                     [-I3,              np.zeros((3,3))]])

def symplectic_residual(Phi):
    J = J6()
    return Phi.T @ J @ Phi - J

def check_symplectic_at_state(x0, dt, mu,
                              tol_sym_fro=5e-5,
                              tol_sym_inf=5e-5,
                              tol_det=5e-5,
                              tol_rev_fro=5e-5,
                              verbose=True):
    """
    Compute ds_2bpSTM forward/backward STMs and check:
      1) ||Phi^T J Phi - J||_F, ||.||_inf
      2) det(Phi) ~ 1
      3) reversibility: Phi(-dt) @ Phi(dt) ~ I
    Throws AssertionError if any check exceeds tolerance.
    """
    # forward step
    x1, Phi = ds_2bpSTM(x0, dt, mu)
    # symplectic defect
    S = symplectic_residual(Phi)
    sF = np.linalg.norm(S, 'fro')
    sI = np.linalg.norm(S, np.inf)
    detP = np.linalg.det(Phi)
    # backward step at x1
    _, Phi_back = ds_2bpSTM(x1, -dt, mu)
    rev = Phi_back @ Phi - np.eye(6)
    rF = np.linalg.norm(rev, 'fro')

    if verbose:
        print(f"  dt = {dt: .3e} s")
        print(f"    ||Phi^T J Phi - J||_F   = {sF:.3e}")
        print(f"    ||Phi^T J Phi - J||_inf = {sI:.3e}")
        print(f"    det(Phi)                = {detP:.12f}")
        print(f"    ||Phi(-dt)Phi(dt)-I||_F = {rF:.3e}")

    # assertions
    assert sF <= tol_sym_fro, f"Symplectic Frobenius residual too large: {sF}"
    assert sI <= tol_sym_inf, f"Symplectic infinity-norm residual too large: {sI}"
    assert abs(detP - 1.0) <= tol_det, f"det(Phi) not ~1: {detP}"
    assert rF <= tol_rev_fro, f"Reversibility residual too large: {rF}"

def make_circular_state(a_km, mu):
    """Return Cartesian (r,v) for a circular equatorial orbit in the x–y plane."""
    r = np.array([a_km, 0.0, 0.0])
    vmag = np.sqrt(mu / a_km)
    v = np.array([0.0, vmag, 0.0])
    return r, v

def random_kepler_state(mu, a_range=(0.5, 3.0), e_range=(0.0, 0.8),
                        i_range=(0.0, np.deg2rad(85.0)),
                        seed=None):
    """
    Draw a random Keplerian set and convert to Cartesian in PQW->IJK.
    (Simple generator—sufficient for stress testing.)
    """
    rng = np.random.default_rng(seed)
    a = 1.0e8 * rng.uniform(*a_range)          # scale semi-major axis (km)
    e = rng.uniform(*e_range)
    i = rng.uniform(*i_range)
    RAAN  = rng.uniform(0, 2*np.pi)
    argp  = rng.uniform(0, 2*np.pi)
    nu    = rng.uniform(0, 2*np.pi)
    p = a*(1 - e*e)
    cnu, snu = np.cos(nu), np.sin(nu)
    r_pqw = np.array([p*cnu/(1+e*cnu), p*snu/(1+e*cnu), 0.0])
    vp = np.sqrt(mu/p)
    v_pqw = np.array([-vp*snu, vp*(e + cnu), 0.0])
    # rotation: R3(RAAN) R1(i) R3(argp)
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    ci, si = np.cos(i),    np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)
    R3_O = np.array([[ cO,-sO,0],[ sO, cO,0],[0,0,1]])
    R1_i = np.array([[1,0,0],[0,ci,-si],[0,si,ci]])
    R3_w = np.array([[ cw,-sw,0],[ sw, cw,0],[0,0,1]])
    C = R3_O @ R1_i @ R3_w
    r_ijk = C @ r_pqw
    v_ijk = C @ v_pqw
    return r_ijk, v_ijk

def main():
    print("=== Symplectic test for ds_2bpSTM (STM) ===")

    # 1) SPICE Sun–Mars state (optional realistic case)
    if USE_SPICE:
        print("\n[Case A] Sun–Mars heliocentric state (SPICE)")
        load_kernels()
        et0 = sp.utc2et("2025-01-01T00:00:00")
        r0, v0 = heliocentric_state("MARS BARYCENTER", et0)  # or "MARS"
        x0 = np.hstack([r0, v0])
        for days in [1, 7, 30, 90]:
            dt = days * 86400.0
            check_symplectic_at_state(x0, dt, MU_SUN)

    # 2) Simple circular test (sanity, Earth circular LEO-ish scale)
    print("\n[Case B] Circular synthetic state (Earth mu)")
    r0, v0 = make_circular_state(a_km=7000.0, mu=MU_EARTH)
    x0 = np.hstack([r0, v0])
    for minutes in [10, 45, 90]:
        dt = minutes * 60.0
        check_symplectic_at_state(x0, dt, MU_EARTH)

    # 3) Random synthetic Keplerian states (Sun mu), many dt samples
    print("\n[Case C] Random Keplerian states (Sun mu), dt sweep")
    rng = np.random.default_rng(42)
    for k in range(5):  # 5 random states
        r0, v0 = random_kepler_state(MU_SUN, seed=rng.integers(1, 10_000))
        x0 = np.hstack([r0, v0])
        # shorter to longer steps
        for days in [0.1, 1, 10, 50]:
            dt = days * 86400.0
            check_symplectic_at_state(x0, dt, MU_SUN)

    print("\nAll symplectic checks PASSED within tolerances.")

if __name__ == "__main__":
    main()

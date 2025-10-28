# ds_2bpSTM.py
# Python port of ds_2bpSTM.m (Battin/Vallado universal-variable solver with STM)
# Returns final Cartesian state and/or State Transition Matrix for a single time step.

from __future__ import annotations
import numpy as np

def ds_2bpSTM(x0_C: np.ndarray, dt: float, GM: float, x1_C: np.ndarray | None = None):
    """
    [x1_C, STM_M] = ds_2bpSTM(x0_C, dt, GM, x1_C=None)

    Compute the final state and/or state transition matrix for two-body motion
    using universal variables (Battin/Vallado), with the same scaling and logic
    as the original MATLAB routine.

    Parameters
    ----------
    x0_C : (6,) array_like
        Initial state [r; v] in km, km/s.
    dt : float
        Propagation interval in seconds.
    GM : float
        Gravitational parameter (km^3/s^2).
    x1_C : (6,) array_like, optional
        If provided, routine will *not* recompute the final state (uses x1_C to
        build the STM only). Units as x0_C.

    Returns
    -------
    x1_C : (6,) ndarray
        Final state (km, km/s).
    STM_M : (6,6) ndarray
        State transition matrix from x0_C to x1_C.
    """
    x0_C = np.asarray(x0_C, dtype=float).reshape(6)
    if x1_C is not None:
        x1_C = np.asarray(x1_C, dtype=float).reshape(6)

    # --- Early exit (optional, commented out in original)
    # if abs(dt) == 0:
    #     STM = np.eye(6)
    #     return x0_C.copy(), STM

    # ---------------- NORMALIZATION (same as original) ----------------
    LSF = np.linalg.norm(x0_C[0:3])        # length scale
    VSF = np.linalg.norm(x0_C[3:6])        # speed scale
    XSF = np.array([LSF, LSF, LSF, VSF, VSF, VSF])
    TSF = LSF / VSF                        # time scale
    GMSF = LSF**3 / TSF**2

    x0 = x0_C / XSF
    dt_n = dt / TSF
    GM_n = GM / GMSF

    # Flags
    compute_x1 = (x1_C is None)
    compute_STM = True  # This routine always returns STM when called from Python

    if not compute_x1:
        x1 = x1_C / XSF

    # Split state
    r0_C = x0[0:3]
    v0_C = x0[3:6]
    r0 = _work_norm(r0_C[None, :])[0]
    v0 = _work_norm(v0_C[None, :])[0]

    # -- sigma0 and alpha (Battin p.174/175)
    si0 = _work_dot(r0_C[None, :], v0_C[None, :])[0] / np.sqrt(GM_n)
    alpha = 2.0 / r0 - (v0**2) / GM_n

    # -- solve for universal anomaly xsi, and r1
    toll = 1e-15
    if compute_x1:
        r1, xsi = _aux_xsi(alpha, GM_n, si0, r0, v0, dt_n, toll)
    else:
        r1_C = x1[0:3]
        v1_C = x1[3:6]
        r1 = _work_norm(r1_C[None, :])[0]
        si1 = _work_dot(r1_C[None, :], v1_C[None, :])[0] / np.sqrt(GM_n)
        xsi = alpha * np.sqrt(GM_n) * dt_n + si1 - si0

    # U functions
    # NOTE: original code returns (U0,U1,U2,U3,U4,U5) but only uses U1,U2,U4,U5 here
    _, U1, U2, _, U4, U5 = _aux_univ(alpha, xsi)

    # --- F, G, Ft, Gt (Battin p.179)
    F = 1.0 - U2 / r0
    G = r0 * U1 / np.sqrt(GM_n) + U2 * si0 / np.sqrt(GM_n)
    Ft = -np.sqrt(GM_n) * U1 / (r1 * r0)
    Gt = 1.0 - U2 / r1

    # --- Build x1 if needed
    if compute_x1:
        r1_C = F * r0_C + G * v0_C
        v1_C = Ft * r0_C + Gt * v0_C
    else:
        r1_C = x1[0:3]
        v1_C = x1[3:6]

    if not compute_STM:
        x1_out = np.hstack([r1_C, v1_C]) * XSF
        return x1_out, None

    # --- STM (Battin pp. 466–467, eqns around 9.74)
    C = (3.0 * U5 - xsi * U4) / np.sqrt(GM_n) - dt_n * U2

    dv_C = v1_C - v0_C
    dr_C = r1_C - r0_C

    # Helpers to keep expressions close to the MATLAB
    r0n, r1n = r0, r1

    # dr/dr0  (R~)
    drdr0 = (r1n / GM_n) * np.outer(dv_C, dv_C) \
            + (r0n * (1.0 - F) * np.outer(r1_C, r0_C) + C * np.outer(v1_C, r0_C)) / (r0n**3) \
            + F * np.eye(3)

    # dr/dv0  (R)
    drdv0 = (r0n / GM_n) * (1.0 - F) * (np.outer(dr_C, v0_C) - np.outer(dv_C, r0_C)) \
            + (C / GM_n) * np.outer(v1_C, v0_C) \
            + G * np.eye(3)

    # dv/dr0  (V~)
    dvdr0 = -np.outer(dv_C, r0_C) / (r0n**2) \
                - np.outer(r1_C, dv_C) / (r1n**2) \
                - Ft * np.outer(r1_C, r1_C) / (r1n**2) \
                + Ft * (np.outer(r1_C, v1_C) - np.outer(v1_C, r1_C)) \
                    * (np.dot(r1_C, dv_C)) / (GM_n * r1n) \
                - (GM_n * C) / (r1n**3 * r0n**3) * np.outer(r1_C, r0_C)

    # finalize dv/dr0 (Battin adds +Ft*I at the end)
    dvdr0 = dvdr0 + Ft * np.eye(3)

    # dv/dv0  (V)
    dvdv0 = np.outer(dv_C, dv_C) * (r0n / GM_n) \
            + (r0n * (1.0 - F) * np.outer(r1_C, r0_C) - C * np.outer(r1_C, v0_C)) / (r1n**3) \
            + Gt * np.eye(3)

    # Pack (normalized)
    STM_n = np.block([[drdr0, drdv0],
                      [dvdr0, dvdv0]])

    # ------------------ SCALE BACK TO DIMENSIONED ------------------
    x1_out = np.hstack([r1_C, v1_C]) * XSF

    # dimensionalize STM exactly like the MATLAB
    S = np.block([
        [np.eye(3),                 (LSF / VSF) * np.eye(3)],
        [(VSF / LSF) * np.eye(3),   np.eye(3)]
    ])
    STM_M = STM_n.copy()
    STM_M = np.block([
        [drdr0,      drdv0 * (LSF / VSF)],
        [dvdr0 * (VSF / LSF), dvdv0]
    ])

    return x1_out, STM_M


# -------------------------- helpers (faithful to MATLAB) --------------------------

# in ds_2bpSTM.py

# robust Stumpff for small arguments
def _aux_univ(alpha: float, xsi: float):
    # threshold for switching to series
    eps = 1e-10
    if abs(alpha) < eps:
        # Parabolic limit via series (up to ξ^5 is fine in double)
        U0 = 1.0
        U1 = xsi
        U2 = 0.5 * xsi**2
        U3 = (1.0/6.0) * xsi**3
        U4 = (1.0/24.0) * xsi**4
        U5 = (1.0/120.0) * xsi**5
        return U0, U1, U2, U3, U4, U5

    if alpha > 0:
        sa = np.sqrt(alpha); z = sa * xsi
        c, s = np.cos(z), np.sin(z)
        U0 = c
        U1 = s / sa
    else:
        sa = np.sqrt(-alpha); z = sa * xsi
        ch, sh = np.cosh(z), np.sinh(z)
        U0 = ch
        U1 = sh / sa

    U2 = (1.0 - U0) / alpha
    U3 = (xsi - U1) / alpha
    U4 = (xsi**2 / 2.0 - U2) / alpha
    U5 = (xsi**3 / 6.0 - U3) / alpha
    return U0, U1, U2, U3, U4, U5



def _aux_xsi(alpha, GM, si0, r0, v0, dt, toll):
    """Newton solve for universal anomaly xsi (Battin p.179); also returns r1."""
    # First guess
    if abs(alpha) < 1e-7:
        # Parabola
        p = (r0**2) * (v0**2) / GM - si0**2
        s = np.arctan(np.sqrt(p**3 / GM) / 3.0 / dt)
        w = np.arctan(np.tan(s)**(1.0 / 3.0))
        xsi = 2.0 * np.sqrt(p) / np.tan(2.0 * w)
    elif alpha >= 1e-7:
        # Ellipse
        xsi = np.sqrt(GM) * dt * alpha
    else:
        # Hyperbola
        a = 1.0 / alpha
        xsi = np.sign(dt) * np.sqrt(-a) * np.log(
            -2.0 * alpha * dt * np.sqrt(GM) / (si0 + np.sign(dt) * np.sqrt(-a) * (1.0 - r0 * alpha))
        )

    xsiold = 1e6
    while abs(xsi - xsiold) > toll:
        U0, U1, U2, U3, _, _ = _aux_univ(alpha, xsi)
        r1 = r0 * U0 + si0 * U1 + U2           # = -dKepler/dxsi
        Kepler = np.sqrt(GM) * dt - (r0 * U1 + si0 * U2 + U3)
        xsiold = xsi
        xsi = xsiold + Kepler / r1             # Newton step
    return r1, xsi


def _work_norm(x_M: np.ndarray):
    """Row-wise vector norms (follows MATLAB helper)."""
    return np.sqrt(_work_dot(x_M, x_M))


def _work_dot(x_M: np.ndarray, y_M: np.ndarray):
    """Row-wise dot products (follows MATLAB helper)."""
    return np.sum(x_M * y_M, axis=1)

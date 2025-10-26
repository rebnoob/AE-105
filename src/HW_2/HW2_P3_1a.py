import numpy as np

# Utility helpers
def _wrap2pi(x):
    return (x + 2*np.pi) % (2*np.pi)

def _deg2rad(x, in_degrees):
    return np.radians(x) if in_degrees else x

def _rad2deg(x, out_degrees):
    return np.degrees(x) if out_degrees else x

# r,v  ->  classical elements (ELORB)
def rv2coe(r, v, mu, deg=True, atol=1e-10):
    r = np.asarray(r, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)

    rnorm = np.linalg.norm(r)
    vnorm = np.linalg.norm(v)
    rv_dot = float(np.dot(r, v))

    # h, n, e, xi
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)

    K = np.array([0.0, 0.0, 1.0])
    n = np.cross(K, h)
    nnorm = np.linalg.norm(n)

    e_vec = ((vnorm**2 - mu/rnorm)*r - rv_dot*v) / mu
    e = np.linalg.norm(e_vec)

    xi = 0.5*vnorm**2 - mu/rnorm

    # a, p
    if abs(e - 1.0) > 1e-12:
        a = -mu/(2.0*xi)
        p = a*(1.0 - e**2)
    else:
        a = np.inf
        p = hnorm**2/mu

    # i
    cos_i = np.clip(h[2]/hnorm, -1.0, 1.0)
    i = np.arccos(cos_i)

    # RAAN (Ω)
    if nnorm > atol:
        cos_Om = np.clip(n[0]/nnorm, -1.0, 1.0)
        Om = np.arccos(cos_Om)
        if n[1] < 0:
            Om = 2*np.pi - Om
        Om = _wrap2pi(Om)
    else:
        Om = 0.0

    # argument of periapsis (ω)
    if nnorm > atol and e > atol:
        cos_w = np.clip(np.dot(n, e_vec)/(nnorm*e), -1.0, 1.0)
        w = np.arccos(cos_w)
        if e_vec[2] < 0:
            w = 2*np.pi - w
        w = _wrap2pi(w)
    else:
        w = 0.0

    # true anomaly (ν)
    if e > atol:
        cos_nu = np.clip(np.dot(e_vec, r)/(e*rnorm), -1.0, 1.0)
        nu = np.arccos(cos_nu)
        if rv_dot < 0:
            nu = 2*np.pi - nu
        nu = _wrap2pi(nu)
    else:
        nu = 0.0

    # Special-case angles
    w_true_tilde = None
    if i < atol and e > atol:
        # elliptical equatorial
        cos_wtt = np.clip(e_vec[0]/e, -1.0, 1.0)
        wtt = np.arccos(cos_wtt)
        if e_vec[1] < 0:
            wtt = 2*np.pi - wtt
        w_true_tilde = _wrap2pi(wtt)

    u = None
    if e < atol and i >= atol:
        if nnorm > atol:
            cos_u = np.clip(np.dot(n, r)/(nnorm*np.linalg.norm(r)), -1.0, 1.0)
            u_ = np.arccos(cos_u)
            if r[2] < 0:
                u_ = 2*np.pi - u_
            u = _wrap2pi(u_)

    lambda_true = None
    if e < atol and i < atol:
        cos_lt = np.clip(r[0]/np.linalg.norm(r), -1.0, 1.0)
        lt = np.arccos(cos_lt)
        if r[1] < 0:
            lt = 2*np.pi - lt
        lambda_true = _wrap2pi(lt)

    coe = {
        "p": p,
        "a": a,
        "e": e,
        "i": _rad2deg(i, deg),
        "RAAN": _rad2deg(Om, deg),
        "argp": _rad2deg(w, deg),
        "nu": _rad2deg(nu, deg),
    }
    if u is not None:
        coe["u"] = _rad2deg(u, deg)
    if lambda_true is not None:
        coe["lambda_true"] = _rad2deg(lambda_true, deg)
    if w_true_tilde is not None:
        coe["w_true_tilde"] = _rad2deg(w_true_tilde, deg)
    return coe

# classical elements  ->  r,v  (RANDV)
def coe2rv(
    mu,
    p=None, a=None, e=None, i=None, RAAN=None, argp=None, nu=None,
    u=None, lambda_true=None, w_true_tilde=None,
    deg=True, atol=1e-10
):
    """
    Convert classical elements to inertial r,v following Algorithm 6 (RANDV).
    """
    # --- angle handling
    i_   = _deg2rad(i, deg) if i is not None else 0.0
    RAAN_= _deg2rad(RAAN, deg) if RAAN is not None else 0.0
    argp_= _deg2rad(argp, deg) if argp is not None else 0.0
    nu_  = _deg2rad(nu, deg) if nu is not None else None
    u_   = _deg2rad(u, deg) if u is not None else None
    lam_ = _deg2rad(lambda_true, deg) if lambda_true is not None else None
    wtt_ = _deg2rad(w_true_tilde, deg) if w_true_tilde is not None else None

    # --- choose special-case substitutions exactly as in the algorithm
    if (e is not None and abs(e) < atol) and (i_ < atol):
        # Circular Equatorial
        argp_ = 0.0
        RAAN_ = 0.0
        if lam_ is None:
            raise ValueError("Circular equatorial: provide lambda_true")
        nu_ = lam_
    elif (e is not None and abs(e) < atol) and (i_ >= atol):
        # Circular Inclined
        argp_ = 0.0
        if u_ is None:
            raise ValueError("Circular inclined: provide u (argument of latitude)")
        nu_ = u_
    elif (e is not None and e > atol) and (i_ < atol):
        # Elliptical Equatorial
        RAAN_ = 0.0
        if wtt_ is None:
            raise ValueError("Elliptical equatorial: provide w_true_tilde")
        argp_ = wtt_
        if nu_ is None:
            raise ValueError("Provide nu for elliptical equatorial")
    else:
        # General case: must have nu
        if nu_ is None:
            raise ValueError("Provide nu for the general (non-special) case")

    # --- determine p
    if p is None:
        if a is None or e is None:
            raise ValueError("Provide either p, or (a and e).")
        p = a*(1.0 - e**2)

    # --- PQW position & velocity
    cnu, snu = np.cos(nu_), np.sin(nu_)
    r_pqw = np.array([
        p*cnu/(1.0 + (e if e is not None else 0.0)*cnu),
        p*snu/(1.0 + (e if e is not None else 0.0)*cnu),
        0.0
    ])
    vp = np.sqrt(mu/p)
    v_pqw = np.array([
        -vp*snu,
        vp*((e if e is not None else 0.0) + cnu),
        0.0
    ])

    # --- DCM from PQW to IJK: R3(-Ω) R1(-i) R3(-ω)
    cO, sO = np.cos(-RAAN_), np.sin(-RAAN_)
    ci, si = np.cos(-i_),    np.sin(-i_)
    cw, sw = np.cos(-argp_), np.sin(-argp_)

    R3_O = np.array([[ cO, sO, 0.0],
                     [-sO, cO, 0.0],
                     [0.0, 0.0, 1.0]])
    R1_i = np.array([[1.0, 0.0, 0.0],
                     [0.0,  ci,  si],
                     [0.0, -si,  ci]])
    R3_w = np.array([[ cw, sw, 0.0],
                     [-sw, cw, 0.0],
                     [0.0, 0.0, 1.0]])

    C = R3_O @ R1_i @ R3_w   # [IJK<-PQW]

    r_ijk = C @ r_pqw
    v_ijk = C @ v_pqw
    return r_ijk, v_ijk

# One-call front-end: pick direction
def kepler_transform(mode, *, r=None, v=None, mu=None,
                     p=None, a=None, e=None, i=None, RAAN=None, argp=None, nu=None,
                     u=None, lambda_true=None, w_true_tilde=None,
                     deg=True):
    """
    mode='rv2coe'  -> pass r,v,mu ; returns COE dict
    mode='coe2rv'  -> pass mu and elements ; returns (r,v)
    """
    if mode == "rv2coe":
        if r is None or v is None or mu is None:
            raise ValueError("rv2coe: require r, v, mu")
        return rv2coe(r, v, mu, deg=deg)
    elif mode == "coe2rv":
        if mu is None:
            raise ValueError("coe2rv: require mu")
        return coe2rv(mu, p=p, a=a, e=e, i=i, RAAN=RAAN, argp=argp, nu=nu,
                      u=u, lambda_true=lambda_true, w_true_tilde=w_true_tilde,
                      deg=deg)
    else:
        raise ValueError("mode must be 'rv2coe' or 'coe2rv'")

#Quick test
if __name__ == "__main__":
    mu_earth = 398600.4418

    # Your test state
    r0 = np.array([6524.834, 6862.875, 6448.296])
    v0 = np.array([4.901327, 5.533756, -1.976341])

    coe = kepler_transform("rv2coe", r=r0, v=v0, mu=mu_earth, deg=True)
    print("ELORB (rv->coe):")
    for k, val in coe.items():
        print(f"  {k:>15}: {val}")

    # Reconstruct r,v from those elements (general case)
    r1, v1 = kepler_transform(
        "coe2rv",
        mu=mu_earth,
        p=coe["p"], e=coe["e"], i=coe["i"],
        RAAN=coe["RAAN"], argp=coe["argp"], nu=coe["nu"],
        deg=True
    )
    print("\nRANDV (coe->rv) reconstruction error:")
    print("  |r1 - r0| =", np.linalg.norm(r1 - r0), "km")
    print("  |v1 - v0| =", np.linalg.norm(v1 - v0), "km/s")
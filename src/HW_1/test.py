import spiceypy as sp
import numpy as np
from math import sin, cos, asin, atan2, sqrt, radians, degrees

# -------- Kernels (EDIT paths to match your machine) --------
KERNELS = [
    "/Users/donnylu/Documents/ae105/generic_kernels/lsk/naif0012.tls",          # leap seconds
    "/Users/donnylu/Documents/ae105/generic_kernels/spk/planets/de442.bsp",     # <-- planetary ephemeris
    "/Users/donnylu/Documents/ae105/generic_kernels/spk/planets/de430.bsp",     # <-- planetary ephemeris
    "/Users/donnylu/Documents/ae105/generic_kernels/pck/pck00010.tpc",          # IAU body constants
    "/Users/donnylu/Documents/ae105/generic_kernels/pck/gm_de431.tpc",
    "/Users/donnylu/Documents/ae105/generic_kernels/pck/earth_latest_high_prec.bpc",  # Earth orientation for ITRF
    "/Users/donnylu/Documents/ae105/generic_kernels/spk/satellites/jup310.bsp",
    # (Optional) frame kernels that define ITRF93; usually not needed if your BPC is recent.
]


def d2r(x): return np.deg2rad(x)
def r2d(x): return np.rad2deg(x)

def wrap_az_deg(az_deg):
    """Wrap azimuth to [0, 360) degrees."""
    return (az_deg + 360.0) % 360.0

def load_kernels(kernels):
    sp.kclear()
    for k in kernels:
        sp.furnsh(k)

load_kernels(KERNELS)

# --- 1) GM of Jupiter (km^3/s^2)
# Two equivalent ways: bodvcd (by ID) or bodvrd (by name)
gm_jup = sp.bodvcd(599, "GM", 1)[1][0]           # by NAIF ID (Jupiter = 599)
# gm_jup = sp.bodvrd("JUPITER", "GM", 1)[1][0]   # by name (also works)

# --- 2) Moon radius (equatorial)
radii_moon = sp.bodvcd(301, "RADII", 3)[1]       # returns [a, b, c] in km
r_moon_eq = radii_moon[0]                        # equatorial radius (a)

# --- 3) Jupiter prime meridian constants (deg, deg/day, …)
# PM = [W0, W_dot, (optional additional terms…)] depending on kernel
pm_jup = sp.bodvcd(599, "PM", 3)[1]
rot_jup_deg_per_day = pm_jup[1]                  # angular rotation rate (deg/day)

# (1) Start date (UTC string)
date0 = "2025 Apr 1 12:00:00 UTC"

# (2) Convert to ephemeris time (seconds past J2000)
et0 = sp.str2et(date0)

# (3) Make a row vector of times — every hour (3600s) for one day (86400s)
et_R = np.arange(et0, et0 + 86400 + 1, 3600)

# (4) Add 7 days worth of seconds (7 * 86400)
et1 = et0 + 7 * 86400

# (5) Convert back to UTC calendar string with 6 digits after seconds
date1 = sp.et2utc(et1, 'C', 6)

print("et0:", et0)
print("et_R shape:", et_R.shape)
print("et1:", et1)
print("date1:", date1)
lat_deg = 34.1470
lon_deg_east = -118.1440
alt_km = 0.25   # small altitude; you can set 0.0 for simplicity

# Earth radius from kernel (we'll use equatorial radius 'a')
radii = sp.bodvrd("EARTH", "RADII", 3)[1]  # [a,b,c] in km
a = radii[0]
f = 0.0  # sphere assumption (flattening = 0)

# Station vector in Earth-fixed (ITRF) using geodetic spherical model
lon_rad = d2r(lon_deg_east)
lat_rad = d2r(lat_deg)
site_itrf = sp.georec(lon_rad, lat_rad, alt_km, a, f)
print("Site vector in ITRF:", site_itrf)
site_itrf = np.array(site_itrf)
site_itrf

# ------------------------------------------------------------------
# [5] Retrieve planetary ephemerides
# Example: Callisto (504) w.r.t. Jupiter system barycenter (599)
# in the ECLIPJ2000 frame, with no aberration correction
# sp.spkezr takes a scalar ET; vectorize with a comprehension:
states = np.array([
    sp.spkezr("4", et, "J2000", "LT+S", "399")[0]  # 6-vector (pos, vel)
    for et in et_R
])
# positions (km) and velocities (km/s) vs time:
pos_km = states[:, :3]
vel_kmps = states[:, 3:]

# ------------------------------------------------------------------
# [6] Rotation matrix between reference frames
# Example: matrix from Mars body-fixed to ECLIPJ2000 at et0
A = sp.pxform("IAU_MARS", "ECLIPJ2000", et0)

# (3) Compute Mars north pole in the ecliptic reference frame:
Z_bf = np.array([0.0, 0.0, 1.0])   # +Z in body-fixed frame = north pole
Z_ecl = A @ Z_bf                   # direction in ECLIPJ2000

# --- quick prints
print("Callisto positions (first 3 rows) [km]:\n", pos_km[:3])
print("Rotation matrix IAU_MARS->ECLIPJ2000 @ et0:\n", A)
print("Mars north pole in ECLIPJ2000:\n", Z_ecl)

print(f"GM(Jupiter)           = {gm_jup:.10g} km^3/s^2")
print(f"Moon equatorial radius = {r_moon_eq:.3f} km")
print(f"Jupiter rotation rate  = {rot_jup_deg_per_day:.6f} deg/day")

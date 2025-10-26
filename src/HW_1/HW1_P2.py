# pip install spiceypy
import spiceypy as sp
import numpy as np
from math import atan2, sqrt

# ---------------------- Small helpers ----------------------
def d2r(x): return np.deg2rad(x)
def r2d(x): return np.rad2deg(x)
def wrap_az_deg(az_deg): return (az_deg + 360.0) % 360.0

def enu_basis(lat_rad, lon_rad):
    """Return East, North, Up unit vectors expressed in ITRF at the site."""
    e_hat = np.array([-np.sin(lon_rad),                  np.cos(lon_rad),                 0.0])
    n_hat = np.array([-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)])
    u_hat = np.array([ np.cos(lat_rad)*np.cos(lon_rad),  np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)])
    return e_hat, n_hat, u_hat

def az_el_from_los_itrf(los_itrf, E_hat, N_hat, U_hat):
    """
    Azimuth from North toward East, elevation above horizon (degrees),
    given a line-of-sight vector in ITRF and local ENU basis vectors (unit).
    """
    e = float(E_hat @ los_itrf)
    n = float(N_hat @ los_itrf)
    u = float(U_hat @ los_itrf)
    az = atan2(e, n)                          
    el = atan2(u, sqrt(e*e + n*n))            # radians
    return wrap_az_deg(r2d(az)), r2d(el)

def site_vector_itrf(lat_deg, lon_deg_east, alt_km=0.0, sphere=True):

    radii = sp.bodvrd("EARTH", "RADII", 3)[1]  # [a, b, c] km
    a = float(radii[0])
    f = 0.0 if sphere else (radii[0] - radii[2]) / radii[0]
    # NOTE: georec expects (lon, lat) in radians, altitude in km, a, f
    return np.array(sp.georec(d2r(lon_deg_east), d2r(lat_deg), alt_km, a, f))

def j2000_to_itrf_vec(vec_j2000, et):
    R = sp.pxform("J2000", "ITRF93", et)
    return R @ vec_j2000

def ecm_to_target_itrf(target, et, abcorr="LT+S", observer="EARTH"):
    st, _ = sp.spkezr(str(target), et, "J2000", abcorr, observer)  # ECM->target in J2000
    r_j2000 = np.array(st[:3])
    return j2000_to_itrf_vec(r_j2000, et)

def az_el_of_target_from_site(target, et, site_itrf, E_hat, N_hat, U_hat, abcorr="LT+S"):
    """Az/El (deg) of target as seen from the given site."""
    r_ecm_to_tgt_itrf = ecm_to_target_itrf(target, et, abcorr=abcorr, observer="EARTH")
    los_itrf = r_ecm_to_tgt_itrf - site_itrf   # site -> target (ITRF)
    return az_el_from_los_itrf(los_itrf, E_hat, N_hat, U_hat)

# ---------------------- Load kernels ----------------------
sp.kclear()
# Adjust these paths to yours
sp.furnsh("/Users/donnylu/Documents/ae105/generic_kernels/lsk/naif0012.tls")                 # leap seconds
sp.furnsh("/Users/donnylu/Documents/ae105/generic_kernels/pck/pck00010.tpc")                 # IAU constants
sp.furnsh("/Users/donnylu/Documents/ae105/generic_kernels/pck/earth_latest_high_prec.bpc")   # Earth orientation (ITRF)
sp.furnsh("/Users/donnylu/Documents/ae105/generic_kernels/spk/planets/de442.bsp")            # planets

# ---------------------- Problem settings ----------------------
# Time: Apr 5, 2025, 3 AM local Pasadena.
# Real clock is PDT (UTC-7) -> 10:00:00 UTC. If you must follow literal "PST", set use_pst=True.
use_pst = False
utc = "2025-04-05T11:00:00" if use_pst else "2025-04-05T10:00:00"
et = sp.utc2et(utc)

# Pasadena site (east-positive longitude). Altitude is optional; set 0 for the spherical-Earth assumption.
lat_deg = 34.1470
lon_deg_east = -118.1440
alt_km = 0.0   # spherical Earth; altitude not required
site_itrf = site_vector_itrf(lat_deg, lon_deg_east, alt_km=alt_km, sphere=True)

# Local ENU basis at the site (expressed in ITRF)
E_hat, N_hat, U_hat = enu_basis(d2r(lat_deg), d2r(lon_deg_east))

# Aberration/light-time option: "LT+S" (proper) or "NONE" (geometric)
abcorr = "LT+S"

# Choose whether to use barycenters (4/5) or planet centers (499/599)
use_barycenters = True
mars_target = 4 if use_barycenters else "MARS"      # 4 or 499/"MARS"
jup_target  = 5 if use_barycenters else "JUPITER"   # 5 or 599/"JUPITER"

# ---------------------- Compute Az/El ----------------------
az_mars, el_mars = az_el_of_target_from_site(mars_target, et, site_itrf, E_hat, N_hat, U_hat, abcorr=abcorr)
az_jup,  el_jup  = az_el_of_target_from_site(jup_target,  et, site_itrf, E_hat, N_hat, U_hat, abcorr=abcorr)

print(f"{sp.et2utc(et,'C',3)}   (Az: 0°=North, 90°=East; Elevation positive up)")
print(f"  Mars:    az = {az_mars:7.3f}°,  el = {el_mars:7.3f}°")
print(f"  Jupiter: az = {az_jup:7.3f}°,  el = {el_jup:7.3f}°")


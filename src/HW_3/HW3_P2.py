#!/usr/bin/env python3
import numpy as np
import spiceypy as spice
import csv

# ======================================================
# 1. YOUR KERNEL PATHS
# ======================================================

TLS = "/Users/rebnoob/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/rebnoob/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK_PLANETS = "/Users/rebnoob/Documents/ae105/generic_kernels/spk/planets/de442.bsp"
SPK_CLIPPER = "/Users/rebnoob/Documents/ae105/generic_kernels/spk/21F31_MEGA_L241010_A300411_LP01_V6_pad_scpse.bsp"
# ^ change filename above if your file name is slightly different

# ======================================================
# 2. CONSTANTS
# ======================================================

START_UTC = "2032-07-20T00:00:00"
END_UTC   = "2032-11-15T00:00:00"
STEP_SEC  = 3600.0  # 1 hour

M_SC = 4000.0      # kg
T_THRUST = 300.0   # N
A_SOLAR = 100.0    # m^2
C_R = 1.18

# Jupiter
MU_JUP = 1.26686534e17    # m^3/s^2
R_JUP  = 7.1492e7         # m
J2_JUP = 0.0147

# Sun
MU_SUN = 1.32712440018e20

# SRP
P_SRP_1AU = 4.56e-6       # N/m^2
AU = 1.495978707e11       # m

# NAIF names
SC_NAME   = "EUROPA CLIPPER"
JUP_NAME  = "JUPITER BARYCENTER"
SUN_NAME  = "SUN"
IO_NAME        = "IO"
EUROPA_NAME    = "EUROPA"
GANYMEDE_NAME  = "GANYMEDE"
CALLISTO_NAME  = "CALLISTO"
SATURN_NAME    = "SATURN BARYCENTER"

OUT_CSV = "europa_clipper_accels.csv"

# ------------------------------------------------------
# Hardcoded GMs (JPL/NAIF standard values)
# values originally in km^3/s^2, multiplied by 1e9 → m^3/s^2
# ------------------------------------------------------
GMS = {
    "IO":        5959.916e9,   # 5959.916 km^3/s^2
    "EUROPA":    3202.739e9,
    "GANYMEDE":  9887.834e9,
    "CALLISTO":  7179.289e9,
    # Saturn GM (from NAIF: 37940626.061137 km^3/s^2)
    "SATURN":    37940626.061137e9,
}

# ======================================================
# 3. HELPERS
# ======================================================

def load_kernels():
    spice.furnsh(TLS)
    spice.furnsh(PCK)
    spice.furnsh(SPK_PLANETS)
    spice.furnsh(SPK_CLIPPER)

def utc_range(start_utc, end_utc, step_sec):
    et_start = spice.str2et(start_utc)
    et_end   = spice.str2et(end_utc)
    t = et_start
    while t <= et_end + 1e-6:
        yield t
        t += step_sec

def get_pos(target, et, observer=JUP_NAME, frame="J2000", abcorr="NONE"):
    state, _ = spice.spkezr(target, et, frame, abcorr, observer)
    return np.array(state[:3]) * 1000.0

def grav_acc(mu, r_vec):
    r = np.linalg.norm(r_vec)
    return -mu * r_vec / r**3

def j2_acc_jupiter(r_vec):
    r = np.linalg.norm(r_vec)
    if r == 0.0:
        return np.zeros(3)
    base = (MU_JUP / r**2) * 1.5 * J2_JUP * (R_JUP / r)**2
    x, y, z = r_vec
    # J2 vector formula:
    # a_J2 = - (3/2) J2 (mu/r^2) (Re/r)^2  [ (1 - 5(z/r)^2) r_hat  +  2(z/r) k_hat ]
    # Let's match the 'base' variable which is (3/2) J2 (mu/r^2) (Re/r)^2
    
    # term1 = (1 - 5 * (z/r)**2) * (r_vec / r)
    # term2 = 2 * (z/r) * np.array([0, 0, 1])
    # return -base * (term1 + term2)
    
    # Optimized:
    z_sq_r_sq = (z / r)**2
    return -base * ( (r_vec/r) * (1 - 5*z_sq_r_sq) + np.array([0,0,1]) * (2*z/r) )

def srp_acc(sc_pos_sun):
    R = np.linalg.norm(sc_pos_sun)
    if R == 0.0:
        return np.zeros(3)
    P = P_SRP_1AU * (AU / R)**2
    F = P * C_R * A_SOLAR
    a = F / M_SC
    return a * (sc_pos_sun / R)  # away from sun

def thrust_acc(sc_vel_jup):
    a_mag = T_THRUST / M_SC  # 0.075 m/s^2
    v = np.linalg.norm(sc_vel_jup)
    if v == 0.0:
        return np.zeros(3)
    return a_mag * (sc_vel_jup / v)

# ======================================================
# 4. MAIN
# ======================================================

def main():
    load_kernels()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "utc",
            "a_jupiter_central",
            "a_jupiter_J2",
            "a_sun",
            "a_io",
            "a_europa",
            "a_ganymede",
            "a_callisto",
            "a_saturn",
            "a_srp",
            "a_thrust",
        ])

        for et in utc_range(START_UTC, END_UTC, STEP_SEC):
            utc_str = spice.et2utc(et, "ISOC", 0)

            # spacecraft wrt Jupiter
            sc_state_jup, _ = spice.spkezr(SC_NAME, et, "J2000", "NONE", JUP_NAME)
            sc_r_jup = np.array(sc_state_jup[:3]) * 1000.0
            sc_v_jup = np.array(sc_state_jup[3:]) * 1000.0

            # bodies wrt Jupiter
            sun_r_jup = get_pos(SUN_NAME, et)
            io_r_jup  = get_pos(IO_NAME, et)
            eu_r_jup  = get_pos(EUROPA_NAME, et)
            ga_r_jup  = get_pos(GANYMEDE_NAME, et)
            ca_r_jup  = get_pos(CALLISTO_NAME, et)
            sa_r_jup  = get_pos(SATURN_NAME, et)

            # relative vectors SC - body
            sc_r_sun_rel = sc_r_jup - sun_r_jup
            sc_r_io_rel  = sc_r_jup - io_r_jup
            sc_r_eu_rel  = sc_r_jup - eu_r_jup
            sc_r_ga_rel  = sc_r_jup - ga_r_jup
            sc_r_ca_rel  = sc_r_jup - ca_r_jup
            sc_r_sa_rel  = sc_r_jup - sa_r_jup

            # accelerations
    # accelerations
            a_jup  = grav_acc(MU_JUP, sc_r_jup)
            a_j2   = j2_acc_jupiter(sc_r_jup)
            
            # Third-body perturbations (Direct - Indirect)
            # Indirect = acceleration of Jupiter due to Body = grav_acc(GM, body_r_jup)
            # But grav_acc returns force on target towards attractor. 
            # Force on Jup due to Body is towards Body. Vector is body_r_jup.
            # a_indirect = G * M_body * (body_r_jup) / |body_r_jup|^3
            # This is exactly grav_acc(GM, body_r_jup) because grav_acc takes r_vec as pos of object relative to center?
            # Wait, grav_acc(mu, r_vec) returns -mu * r_vec / r^3.
            # If r_vec is position of SC wrt Body, then it's accel of SC towards Body.
            # Here sc_r_sun_rel is SC - Sun. So it is position of SC wrt Sun. Correct.
            # Indirect term: Acceleration of Jupiter due to Sun.
            # Position of Jupiter wrt Sun is (jup_r_sun) = -sun_r_jup.
            # So a_jup_due_to_sun = -mu_sun * (-sun_r_jup) / |sun_r_jup|^3 = mu_sun * sun_r_jup / r^3.
            # Alternatively: a_indirect = grav_acc(MU_SUN, -sun_r_jup).
            
            # Let's use the helper carefully.
            # a_sc_sun = a_direct - a_indirect
            # a_direct = grav_acc(MU_SUN, sc_r_sun_rel)  (Acc of SC due to Sun)
            # a_indirect = grav_acc(MU_SUN, -sun_r_jup)  (Acc of Jup due to Sun)
            
            a_sun  = grav_acc(MU_SUN, sc_r_sun_rel) - grav_acc(MU_SUN, -sun_r_jup)
            a_io   = grav_acc(GMS["IO"],       sc_r_io_rel) - grav_acc(GMS["IO"],       -io_r_jup)
            a_eur  = grav_acc(GMS["EUROPA"],   sc_r_eu_rel) - grav_acc(GMS["EUROPA"],   -eu_r_jup)
            a_gan  = grav_acc(GMS["GANYMEDE"], sc_r_ga_rel) - grav_acc(GMS["GANYMEDE"], -ga_r_jup)
            a_cal  = grav_acc(GMS["CALLISTO"], sc_r_ca_rel) - grav_acc(GMS["CALLISTO"], -ca_r_jup)
            a_sat  = grav_acc(GMS["SATURN"],   sc_r_sa_rel) - grav_acc(GMS["SATURN"],   -sa_r_jup)

            # SRP (need SC wrt Sun directly)
            sc_state_sun, _ = spice.spkezr(SC_NAME, et, "J2000", "NONE", "SUN")
            sc_r_sun_abs = np.array(sc_state_sun[:3]) * 1000.0
            a_srp = srp_acc(sc_r_sun_abs)

            # thrust
            a_thr = thrust_acc(sc_v_jup)

            writer.writerow([
                utc_str,
                np.linalg.norm(a_jup),
                np.linalg.norm(a_j2),
                np.linalg.norm(a_sun),
                np.linalg.norm(a_io),
                np.linalg.norm(a_eur),
                np.linalg.norm(a_gan),
                np.linalg.norm(a_cal),
                np.linalg.norm(a_sat),
                np.linalg.norm(a_srp),
                np.linalg.norm(a_thr),
            ])

    print(f"wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
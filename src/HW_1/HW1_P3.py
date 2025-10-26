# pip install spiceypy numpy matplotlib
import spiceypy as sp
import numpy as np
import matplotlib.pyplot as plt

# ---------- Kernels (adjust paths) ----------
TLS = "/Users/donnylu/Documents/ae105/generic_kernels/lsk/naif0012.tls"
PCK = "/Users/donnylu/Documents/ae105/generic_kernels/pck/pck00010.tpc"
SPK = "/Users/donnylu/Documents/ae105/generic_kernels/spk/planets/de442.bsp"

def load_kernels():
    sp.kclear()
    sp.furnsh(TLS)
    sp.furnsh(PCK)
    sp.furnsh(SPK)

def linspace_ets(utc_start: str, utc_end: str, n: int):
    et0 = sp.utc2et(utc_start); et1 = sp.utc2et(utc_end)
    return np.linspace(et0, et1, n)

def bary_heliocentric_xy_au(target_bary_name: str, ets):
    """
    Return Sun-centered ecliptic x,y [AU] for a SYSTEM BARYCENTER target.
    r_helio = r_target^SSB - r_sun^SSB, all in ECLIPJ2000.
    """
    AU = 149_597_870.700  # km
    r_tgt = np.array([sp.spkezr(target_bary_name, et, "ECLIPJ2000", "NONE", "SSB")[0][:3] for et in ets])
    r_sun = np.array([sp.spkezr("SUN",              et, "ECLIPJ2000", "NONE", "SSB")[0][:3] for et in ets])
    r_helio = (r_tgt - r_sun) / AU
    return r_helio[:,0], r_helio[:,1]

def main():
    load_kernels()

    # Time grid across 2025
    ets = linspace_ets("2025-01-01T00:00:00", "2025-12-31T23:59:59", 200)

    # Use NAIF barycenter names (1..9). "EARTH BARYCENTER" is the EMB.
    barycenters = [
        ("MERCURY BARYCENTER", "Mercury (bary)"),
        ("VENUS BARYCENTER",   "Venus (bary)"),
        ("EARTH BARYCENTER",   "Earth-Moon (EMB)"),
        ("MARS BARYCENTER",    "Mars (bary)"),
        ("JUPITER BARYCENTER", "Jupiter (bary)"),
        ("SATURN BARYCENTER",  "Saturn (bary)"),
        ("URANUS BARYCENTER",  "Uranus (bary)"),
        ("NEPTUNE BARYCENTER", "Neptune (bary)"),
        # Optional:
        ("PLUTO BARYCENTER",   "Pluto (bary)"),
    ]

    plt.figure(figsize=(8,8))
    for name, label in barycenters:
        x, y = bary_heliocentric_xy_au(name, ets)
        plt.plot(x, y, linewidth=1.6, label=label)
        plt.scatter(x[0],  y[0],  s=12)           # Jan 1 marker
        plt.scatter(x[-1], y[-1], s=12, marker="x")  # Dec 31 marker

    # Sun at origin
    plt.scatter(0, 0, s=60, label="Sun")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Planet System Barycenter Orbits in Ecliptic Frame (ECLIPJ2000), 2025")
    plt.xlabel("X [AU]  (Ecliptic)")
    plt.ylabel("Y [AU]  (Ecliptic)")
    plt.grid(True, alpha=0.4)
    plt.legend(loc="upper right", ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()  # add plt.savefig(...) if you want a file

if __name__ == "__main__":
    main()
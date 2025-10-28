import pandas as pd
df = pd.read_csv("porkchop_earth_to_mars_2035.csv")

best = df.loc[df["C3_km2s2"].idxmin()]
print(best)
import numpy as np
import matplotlib.pyplot as plt
from lamberthub import izzo2015
import spiceypy as sp

MU_SUN = 1.327124400419e11
FRAME = "ECLIPJ2000"

et_dep = sp.utc2et(best["dep_utc"])
et_arr = sp.utc2et(best["arr_utc"])

# get Sun-centered states
def bary_heliocentric_rv_km(target_bary_name, et):
    st_tgt, _ = sp.spkezr(target_bary_name, et, FRAME, "NONE", "SSB")
    st_sun, _ = sp.spkezr("SUN", et, FRAME, "NONE", "SSB")
    r = np.array(st_tgt[:3]) - np.array(st_sun[:3])
    v = np.array(st_tgt[3:]) - np.array(st_sun[:3])
    return r, v

r1, vE = bary_heliocentric_rv_km("EARTH BARYCENTER", et_dep)
r2, vM = bary_heliocentric_rv_km("MARS BARYCENTER", et_arr)

tof = (et_arr - et_dep)
v1, v2 = izzo2015(MU_SUN, r1, r2, tof)

# Plot in ecliptic XY
AU = 149_597_870.7
plt.figure(figsize=(7,7))
# Earth orbit preview (approx circle)
plt.scatter(0,0,marker='*',color='orange',label='Sun')
plt.plot([r1[0]/AU],[r1[1]/AU],'bo',label='Earth dep')
plt.plot([r2[0]/AU],[r2[1]/AU],'ro',label='Mars arr')
plt.legend(); plt.axis('equal')
plt.xlabel("X [AU]"); plt.ylabel("Y [AU]")
plt.title("Lambert transfer (Earth→Mars, min C₃ ≈ %.1f km²/s²)" % best["C3_km2s2"])
plt.show()


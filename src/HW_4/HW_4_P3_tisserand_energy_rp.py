#!/usr/bin/env python3
"""
HW 4 Problem 3: Tisserand Graph - Heliocentric Energy vs Periapsis
Plot specific orbital energy vs periapsis distance
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================================================
# CONSTANTS
# ======================================================

AU = 150e6  # km
GM_SUN = 1.32712440018e11  # km^3/s^2

# Planet data
PLANETS = {
    'Mercury': {'a': 0.39, 'R': 2439.7, 'color': '#8C7853'},
    'Venus':   {'a': 0.72, 'R': 6051.8, 'color': '#FFC649'},
    'Earth':   {'a': 1.00, 'R': 6371.0, 'color': '#4A90E2'},
    'Mars':    {'a': 1.52, 'R': 3389.5, 'color': '#E27B58'},
    'Jupiter': {'a': 5.20, 'R': 69911,  'color': '#D4A373'}
}

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def orbital_energy(a):
    """
    Calculate specific orbital energy (energy per unit mass) in km²/s²
    E = -GM_sun / (2a)
    """
    return -GM_SUN / (2 * a)

def a_e_to_rp_ra(a, e):
    """Convert (a, e) to (r_periapsis, r_apoapsis)"""
    r_p = a * (1 - e)
    r_a = a * (1 + e)
    return r_p, r_a

def tisserand_parameter(a, e, a_planet):
    """Calculate Tisserand parameter (planar orbits)"""
    T = (a_planet / a) + 2 * np.sqrt((a / a_planet) * (1 - e**2))
    return T

def resonant_orbit_sma(n, m, a_planet_km):
    """Calculate semi-major axis for n:m resonance"""
    a_res = a_planet_km * (n / m)**(2/3)
    return a_res

# ======================================================
# ENERGY VS PERIAPSIS PLOT
# ======================================================

def plot_energy_vs_periapsis():
    """
    Plot heliocentric specific energy vs periapsis
    Y-axis: E = -GM_sun/(2a) in km²/s²
    X-axis: r_p in AU
    """
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define range
    r_p_min = 0.1 * AU  # km
    r_p_max = 6.0 * AU  # km
    
    planet_order = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
    
    print("="*60)
    print("ENERGY VS PERIAPSIS TISSERAND GRAPH")
    print("="*60)
    
    # ===== PLOT PLANET TISSERAND CURVES =====
    
    for planet_name in planet_order:
        planet = PLANETS[planet_name]
        a_p = planet['a'] * AU  # km
        color = planet['color']
        
        print(f"\n{planet_name} (a = {planet['a']:.2f} AU):")
        
        # Plot multiple T-parameter curves
        T_values = [2.0, 2.3, 2.5, 2.7, 2.8, 2.9, 2.95, 3.0]
        
        for T in T_values:
            r_p_vals = []
            energy_vals = []
            
            # Generate orbits for this T value with log spacing
            a_range = np.logspace(np.log10(0.05 * a_p), np.log10(20 * a_p), 2000)
            
            for a in a_range:
                # Calculate eccentricity for this (a, T)
                term = T - a_p / a
                if term <= 0:
                    continue
                
                e_sq = 1 - (a_p / a) * (term / 2)**2
                
                if 0 <= e_sq <= 1:
                    e = np.sqrt(e_sq)
                    if e < 0.99:  # Avoid near-parabolic
                        r_p, r_a = a_e_to_rp_ra(a, e)
                        
                        # Check if r_p is in range
                        if r_p_min <= r_p <= r_p_max:
                            # Calculate specific energy
                            E = orbital_energy(a)
                            
                            r_p_vals.append(r_p / AU)  # Convert to AU
                            energy_vals.append(E)  # km²/s²
            
            if len(r_p_vals) > 10:
                # Line styling
                if abs(T - 3.0) < 0.01:
                    linestyle = '-'
                    linewidth = 3.0
                    alpha = 1.0
                    label = f'{planet_name} (T=3)'
                else:
                    linestyle = '--'
                    linewidth = 1.2
                    alpha = 0.3 + 0.5 * (T - 2.0) / 1.0
                    label = None
                
                ax.plot(r_p_vals, energy_vals, color=color,
                       linestyle=linestyle, linewidth=linewidth,
                       alpha=alpha, label=label, zorder=5)
        
        # Mark planet's circular orbit
        r_circ = planet['a']
        E_circ = orbital_energy(planet['a'] * AU)
        ax.plot(r_circ, E_circ, 'o', color=color, markersize=15,
               markeredgecolor='white', markeredgewidth=2.5, zorder=15)
        
        # Label with arrow
        offset_x = 1.15 if planet_name != 'Jupiter' else 1.25
        offset_y = 1.05 if planet_name not in ['Mercury', 'Jupiter'] else 1.1
        ax.annotate(planet_name, xy=(r_circ, E_circ),
                   xytext=(r_circ * offset_x, E_circ * offset_y),
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor=color, alpha=0.9, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        print(f"  Circular orbit energy: {E_circ:.2e} km²/s²")
    
    # ===== V∞ LEVEL SETS FOR EARTH =====
    
    print("\n" + "="*60)
    print("V∞ LEVEL SETS (EARTH)")
    print("="*60)
    
    a_earth = PLANETS['Earth']['a'] * AU
    v_inf_levels = [2, 3, 5, 10, 15, 20, 30]  # km/s
    
    for v_inf_kmps in v_inf_levels:
        T = 3 - (v_inf_kmps**2 * a_earth / GM_SUN)
        print(f"v∞ = {v_inf_kmps} km/s -> T = {T:.4f}")
        
        r_p_vals = []
        energy_vals = []
        
        a_range = np.logspace(np.log10(0.05 * AU), np.log10(15 * AU), 2000)
        
        for a in a_range:
            term = T - a_earth / a
            if term <= 0:
                continue
            
            e_sq = 1 - (a_earth / a) * (term / 2)**2
            
            if 0 <= e_sq <= 1:
                e = np.sqrt(e_sq)
                if e < 0.99:
                    r_p, r_a = a_e_to_rp_ra(a, e)
                    
                    if r_p_min <= r_p <= r_p_max:
                        E = orbital_energy(a)
                        r_p_vals.append(r_p / AU)
                        energy_vals.append(E)
        
        if len(r_p_vals) > 10:
            ax.plot(r_p_vals, energy_vals, 'k--', linewidth=1.3, alpha=0.5, zorder=3)
            
            # Label
            if len(r_p_vals) > 50:
                idx = len(r_p_vals) // 2
                # Position label slightly offset
                ax.text(r_p_vals[idx] * 1.05, energy_vals[idx] * 1.05,
                       f'{v_inf_kmps}',
                       fontsize=9, style='italic', color='black',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='yellow', alpha=0.7))
    
    # ===== EARTH RESONANCES =====
    
    print("\n" + "="*60)
    print("EARTH RESONANCES")
    print("="*60)
    
    a_earth_km = a_earth
    resonances = [(1, 1), (2, 1), (2, 3), (3, 1)]
    
    for n, m in resonances:
        a_res = resonant_orbit_sma(n, m, a_earth_km)
        E_res = orbital_energy(a_res)
        r_p_res = a_res  # Periapsis for circular orbit
        
        print(f"{n}:{m} resonance: a = {a_res/AU:.4f} AU, E = {E_res:.2e} km²/s²")
        
        # Mark resonance
        if r_p_min <= r_p_res <= r_p_max:
            ax.plot(r_p_res / AU, E_res, 's', color='limegreen', markersize=11,
                   markeredgecolor='darkgreen', markeredgewidth=2, zorder=10)
            ax.text(r_p_res / AU * 1.08, E_res * 1.08, f'{n}:{m}',
                   ha='left', va='bottom', fontsize=10,
                   color='darkgreen', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='lightgreen', alpha=0.8))
    
    # ===== CONSTANT APOAPSIS LINES =====
    
    # Add lines of constant apoapsis for reference
    r_a_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # AU
    
    for r_a_au in r_a_values:
        r_a = r_a_au * AU
        r_p_line = []
        E_line = []
        
        # For constant r_a, vary r_p
        r_p_range = np.logspace(np.log10(r_p_min), np.log10(min(r_a, r_p_max)), 100)
        
        for r_p in r_p_range:
            # Calculate a from r_p and r_a
            a = (r_p + r_a) / 2
            E = orbital_energy(a)
            
            r_p_line.append(r_p / AU)
            E_line.append(E)
        
        if len(r_p_line) > 5:
            ax.plot(r_p_line, E_line, ':', color='gray', linewidth=1, alpha=0.4, zorder=1)
            
            # Label at the rightmost point
            if len(r_p_line) > 0:
                ax.text(r_p_line[-1], E_line[-1], f'r_a={r_a_au}',
                       fontsize=8, color='gray', style='italic', alpha=0.6,
                       ha='right', va='bottom')
    
    # ===== FORMATTING =====
    
    ax.set_xlabel('Periapsis - $r_p$ (AU)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Heliocentric Specific Energy - $E$ (km²/s²)', fontsize=14, fontweight='bold')
    ax.set_title('Tisserand Graph: Orbital Energy vs Periapsis\n' +
                'Solar System - Log Scale',
                fontsize=16, fontweight='bold', pad=20)
    
    # Set log scale (magnitude, so we need to handle negative energies)
    # Energy is always negative for bound orbits, so plot abs value
    ax.set_xscale('log')
    
    # For y-axis, energies are negative. 
    # Use symlog to handle negative values with logarithmic scaling
    ax.set_yscale('symlog')
    
    ax.set_xlim(r_p_min/AU, r_p_max/AU)
    
    # Energy limits
    # E_min corresponds to large a (energy close to 0, e.g. -100)
    # E_max corresponds to small a (energy very negative, e.g. -2000)
    E_min = orbital_energy(r_p_max * 2)  
    E_max = orbital_energy(r_p_min / 2)
    
    # We want increasing energy as we go up:
    # Bottom limit: More negative (E_max * 1.5)
    # Top limit: Less negative (E_min * 0.5)
    ax.set_ylim(E_max * 1.5, E_min * 0.5)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, which='minor')
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    # Add info box
    textstr = ('Specific energy: E = -GM_sun/(2a)\n'
              'More negative = more bound orbit\n'
              'Gray lines: constant apoapsis (r_a)\n'
              'Black dashed: v∞ contours (km/s)')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           horizontalalignment='left', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/tisserand_energy_rp.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Energy vs periapsis graph saved to: {output_file}")
    plt.close()
    
    return output_file

# ======================================================
# MAIN
# ======================================================

def main():
    print("="*60)
    print("TISSERAND GRAPH: ENERGY vs PERIAPSIS")
    print("="*60)
    
    plot_energy_vs_periapsis()
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nThis format shows:")
    print("  • Y-axis: Specific orbital energy (km²/s²)")
    print("  • X-axis: Periapsis distance (AU)")
    print("  • More negative energy = more tightly bound")
    print("  • Tisserand curves show accessible orbits")
    print("  • Gray lines: constant apoapsis")
    print("="*60)

if __name__ == "__main__":
    main()

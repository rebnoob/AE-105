#!/usr/bin/env python3
"""
HW 4 Problem 3: Tisserand Graph - Periapsis vs Apoapsis Format
Alternative visualization with r_p (periapsis) on y-axis and r_a (apoapsis) on x-axis
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
# CONVERSION FUNCTIONS
# ======================================================

def a_e_to_rp_ra(a, e):
    """Convert (a, e) to (r_periapsis, r_apoapsis)"""
    r_p = a * (1 - e)
    r_a = a * (1 + e)
    return r_p, r_a

def rp_ra_to_a_e(r_p, r_a):
    """Convert (r_periapsis, r_apoapsis) to (a, e)"""
    a = (r_a + r_p) / 2
    e = (r_a - r_p) / (r_a + r_p)
    return a, e

def tisserand_parameter(a, e, a_planet):
    """Calculate Tisserand parameter (planar orbits)"""
    T = (a_planet / a) + 2 * np.sqrt((a / a_planet) * (1 - e**2))
    return T

def resonant_orbit_sma(n, m, a_planet_km):
    """Calculate semi-major axis for n:m resonance"""
    a_res = a_planet_km * (n / m)**(2/3)
    return a_res

# ======================================================
# TISSERAND GRAPH: RP VS RA
# ======================================================

def plot_tisserand_rp_ra():
    """
    Create Tisserand graph in r_p vs r_a space
    This format directly shows orbit shapes
    """
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define range
    r_min = 0.2 * AU  # km
    r_max = 6.0 * AU  # km
    
    planet_order = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
    
    print("="*60)
    print("TISSERAND GRAPH: PERIAPSIS vs APOAPSIS")
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
            r_a_vals = []
            
            # Generate orbits for this T value
            # Vary semi-major axis
            a_range = np.linspace(0.1 * a_p, 10 * a_p, 2000)
            
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
                        r_p_vals.append(r_p / AU)  # Convert to AU
                        r_a_vals.append(r_a / AU)
            
            if len(r_p_vals) > 10:
                # Line styling
                if abs(T - 3.0) < 0.01:
                    linestyle = '-'
                    linewidth = 3.0
                    alpha = 1.0
                    label = f'{planet_name} (T=3)'
                else:
                    linestyle = '--'
                    linewidth = 1.0
                    alpha = 0.3 + 0.4 * (T - 2.0) / 1.0
                    label = None
                
                ax.plot(r_a_vals, r_p_vals, color=color,
                       linestyle=linestyle, linewidth=linewidth,
                       alpha=alpha, label=label, zorder=5)
        
        # Mark planet's circular orbit
        r_circ = planet['a']
        ax.plot(r_circ, r_circ, 'o', color=color, markersize=14,
               markeredgecolor='white', markeredgewidth=2, zorder=15)
        
        # Label
        ax.annotate(planet_name, xy=(r_circ, r_circ),
                   xytext=(0.1, -0.15), textcoords='offset fontsize',
                   ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor=color, alpha=0.8, edgecolor='black'))
    
    # ===== V∞ LEVEL SETS FOR EARTH =====
    
    print("\n" + "="*60)
    print("V∞ LEVEL SETS (EARTH)")
    print("="*60)
    
    a_earth = PLANETS['Earth']['a'] * AU
    v_inf_levels = [3, 5, 10, 15, 20]  # km/s
    
    for v_inf_kmps in v_inf_levels:
        T = 3 - (v_inf_kmps**2 * a_earth / GM_SUN)
        print(f"v∞ = {v_inf_kmps} km/s -> T = {T:.4f}")
        
        r_p_vals = []
        r_a_vals = []
        
        a_range = np.linspace(0.1 * AU, 10 * AU, 2000)
        
        for a in a_range:
            term = T - a_earth / a
            if term <= 0:
                continue
            
            e_sq = 1 - (a_earth / a) * (term / 2)**2
            
            if 0 <= e_sq <= 1:
                e = np.sqrt(e_sq)
                if e < 0.99:
                    r_p, r_a = a_e_to_rp_ra(a, e)
                    r_p_vals.append(r_p / AU)
                    r_a_vals.append(r_a / AU)
        
        if len(r_p_vals) > 10:
            ax.plot(r_a_vals, r_p_vals, 'k--', linewidth=1.2, alpha=0.4, zorder=3)
            
            # Label
            if len(r_p_vals) > 50:
                idx = len(r_p_vals) // 3
                ax.annotate(f'v∞={v_inf_kmps} km/s',
                           xy=(r_a_vals[idx], r_p_vals[idx]),
                           xytext=(10, 5), textcoords='offset points',
                           fontsize=9, style='italic',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='yellow', alpha=0.6))
    
    # ===== EARTH RESONANCES =====
    
    print("\n" + "="*60)
    print("EARTH RESONANCES")
    print("="*60)
    
    a_earth_km = a_earth
    resonances = [(1, 1), (2, 1), (2, 3), (3, 1)]
    
    for n, m in resonances:
        a_res = resonant_orbit_sma(n, m, a_earth_km)
        
        print(f"{n}:{m} resonance: a = {a_res/AU:.4f} AU")
        
        # For circular orbit at this resonance
        r_res = a_res / AU
        ax.plot(r_res, r_res, 's', color='green', markersize=10,
               markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        ax.text(r_res, r_res + 0.1, f'{n}:{m}',
               ha='center', va='bottom', fontsize=9,
               color='darkgreen', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='lightgreen', alpha=0.7))
    
    # ===== ADDITIONAL FEATURES =====
    
    # Diagonal line (circular orbits)
    r_diag = np.linspace(r_min/AU, r_max/AU, 100)
    ax.plot(r_diag, r_diag, 'k:', linewidth=2, alpha=0.5,
           label='Circular Orbits (e=0)', zorder=2)
    
    # Sun's Hill sphere approximation lines
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.3)
    
    # ===== FORMATTING =====
    
    ax.set_xlabel('Apoapsis - $r_a$ (AU)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Periapsis - $r_p$ (AU)', fontsize=14, fontweight='bold')
    ax.set_title('Tisserand Graph: Periapsis vs Apoapsis\n' +
                'Solar System Planet Domains and V∞ Contours',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(r_min/AU, r_max/AU)
    ax.set_ylim(r_min/AU, r_max/AU)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    # Add info box
    textstr = ('Each point (r_a, r_p) represents an orbit\n'
              'Diagonal: circular orbits (r_p = r_a)\n'
              'Above diagonal: elliptical orbits\n'
              'Curves: constant Tisserand parameter T')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           horizontalalignment='right', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/tisserand_rp_ra.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Tisserand r_p vs r_a graph saved to: {output_file}")
    plt.close()
    
    return output_file

# ======================================================
# ZOOMED VERSION FOR INNER PLANETS
# ======================================================

def plot_tisserand_rp_ra_inner():
    """
    Zoomed version focusing on inner Solar System
    """
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Narrower range for inner planets
    r_min = 0.3 * AU
    r_max = 2.5 * AU
    
    planet_order = ['Mercury', 'Venus', 'Earth', 'Mars']
    
    print("\n" + "="*60)
    print("INNER SOLAR SYSTEM DETAIL")
    print("="*60)
    
    # Plot planets
    for planet_name in planet_order:
        planet = PLANETS[planet_name]
        a_p = planet['a'] * AU
        color = planet['color']
        
        # Denser T-value grid for detail
        T_values = np.linspace(2.0, 3.0, 20)
        
        for T in T_values:
            r_p_vals = []
            r_a_vals = []
            
            a_range = np.linspace(0.05 * a_p, 5 * a_p, 1500)
            
            for a in a_range:
                term = T - a_p / a
                if term <= 0:
                    continue
                
                e_sq = 1 - (a_p / a) * (term / 2)**2
                
                if 0 <= e_sq <= 1:
                    e = np.sqrt(e_sq)
                    if e < 0.98:
                        r_p, r_a = a_e_to_rp_ra(a, e)
                        if r_min <= r_p <= r_max and r_min <= r_a <= r_max:
                            r_p_vals.append(r_p / AU)
                            r_a_vals.append(r_a / AU)
            
            if len(r_p_vals) > 5:
                alpha = 0.2 + 0.5 * (T - 2.0) / 1.0
                linewidth = 0.5 if T < 2.95 else 2.5
                
                if abs(T - 3.0) < 0.01:
                    label = f'{planet_name}'
                    alpha = 1.0
                else:
                    label = None
                
                ax.plot(r_a_vals, r_p_vals, color=color,
                       linewidth=linewidth, alpha=alpha, label=label, zorder=5)
        
        # Mark planet
        r_circ = planet['a']
        ax.plot(r_circ, r_circ, 'o', color=color, markersize=16,
               markeredgecolor='white', markeredgewidth=2, zorder=15)
        ax.text(r_circ, r_circ - 0.08, planet_name,
               ha='center', va='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5',
                       facecolor=color, alpha=0.9, edgecolor='black'))
    
    # V∞ contours
    a_earth = PLANETS['Earth']['a'] * AU
    v_inf_levels = [2, 3, 5, 8, 10, 15]
    
    for v_inf_kmps in v_inf_levels:
        T = 3 - (v_inf_kmps**2 * a_earth / GM_SUN)
        
        r_p_vals = []
        r_a_vals = []
        
        a_range = np.linspace(0.1 * AU, 3 * AU, 1500)
        
        for a in a_range:
            term = T - a_earth / a
            if term <= 0:
                continue
            
            e_sq = 1 - (a_earth / a) * (term / 2)**2
            
            if 0 <= e_sq <= 1:
                e = np.sqrt(e_sq)
                if e < 0.98:
                    r_p, r_a = a_e_to_rp_ra(a, e)
                    if r_min <= r_p <= r_max and r_min <= r_a <= r_max:
                        r_p_vals.append(r_p / AU)
                        r_a_vals.append(r_a / AU)
        
        if len(r_p_vals) > 10:
            ax.plot(r_a_vals, r_p_vals, 'k--', linewidth=1.5, alpha=0.5, zorder=3)
            
            if len(r_p_vals) > 30:
                idx = len(r_p_vals) // 2
                ax.text(r_a_vals[idx] + 0.05, r_p_vals[idx],
                       f'{v_inf_kmps}',
                       fontsize=10, style='italic', color='black',
                       bbox=dict(boxstyle='round,pad=0.2',
                               facecolor='yellow', alpha=0.7))
    
    # Earth resonances
    a_earth_km = a_earth
    resonances = [(1, 1), (2, 3), (2, 1)]
    
    for n, m in resonances:
        a_res = resonant_orbit_sma(n, m, a_earth_km) / AU
        if r_min/AU <= a_res <= r_max/AU:
            ax.plot(a_res, a_res, 's', color='lime', markersize=12,
                   markeredgecolor='darkgreen', markeredgewidth=2, zorder=10)
            ax.text(a_res + 0.05, a_res + 0.05, f'{n}:{m}',
                   fontsize=10, color='darkgreen', fontweight='bold')
    
    # Diagonal
    r_diag = np.linspace(r_min/AU, r_max/AU, 100)
    ax.plot(r_diag, r_diag, 'k:', linewidth=2.5, alpha=0.6,
           label='Circular Orbits', zorder=2)
    
    # Formatting
    ax.set_xlabel('Apoapsis - $r_a$ (AU)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Periapsis - $r_p$ (AU)', fontsize=14, fontweight='bold')
    ax.set_title('Inner Solar System: Periapsis vs Apoapsis\n' +
                'Detailed View with V∞ Contours (km/s)',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(r_min/AU, r_max/AU)
    ax.set_ylim(r_min/AU, r_max/AU)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/tisserand_rp_ra_inner.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Inner planet detail saved to: {output_file}")
    plt.close()
    
    return output_file

# ======================================================
# LOG-SCALE VERSION
# ======================================================

def plot_tisserand_rp_ra_loglog():
    """
    Log-log scale version for better visualization across full range
    """
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define range
    r_min = 0.2 * AU  # km
    r_max = 6.0 * AU  # km
    
    planet_order = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
    
    print("\n" + "="*60)
    print("LOG-LOG SCALE VERSION")
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
            r_a_vals = []
            
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
                        if r_p > r_min and r_a < r_max * 2:
                            r_p_vals.append(r_p / AU)  # Convert to AU
                            r_a_vals.append(r_a / AU)
            
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
                
                ax.plot(r_a_vals, r_p_vals, color=color,
                       linestyle=linestyle, linewidth=linewidth,
                       alpha=alpha, label=label, zorder=5)
        
        # Mark planet's circular orbit
        r_circ = planet['a']
        ax.plot(r_circ, r_circ, 'o', color=color, markersize=15,
               markeredgecolor='white', markeredgewidth=2.5, zorder=15)
        
        # Label
        offset_x = 1.15 if planet_name != 'Jupiter' else 1.2
        offset_y = 0.85 if planet_name != 'Mercury' else 0.75
        ax.annotate(planet_name, xy=(r_circ, r_circ),
                   xytext=(r_circ * offset_x, r_circ * offset_y),
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor=color, alpha=0.9, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
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
        r_a_vals = []
        
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
                    if r_p > r_min and r_a < r_max * 2:
                        r_p_vals.append(r_p / AU)
                        r_a_vals.append(r_a / AU)
        
        if len(r_p_vals) > 10:
            ax.plot(r_a_vals, r_p_vals, 'k--', linewidth=1.3, alpha=0.5, zorder=3)
            
            # Label
            if len(r_p_vals) > 50:
                idx = len(r_p_vals) // 2
                ax.text(r_a_vals[idx] * 1.05, r_p_vals[idx] * 1.05,
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
        
        print(f"{n}:{m} resonance: a = {a_res/AU:.4f} AU")
        
        # For circular orbit at this resonance
        r_res = a_res / AU
        ax.plot(r_res, r_res, 's', color='limegreen', markersize=11,
               markeredgecolor='darkgreen', markeredgewidth=2, zorder=10)
        ax.text(r_res * 1.12, r_res * 1.12, f'{n}:{m}',
               ha='left', va='bottom', fontsize=10,
               color='darkgreen', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='lightgreen', alpha=0.8))
    
    # ===== ADDITIONAL FEATURES =====
    
    # Diagonal line (circular orbits)
    r_diag = np.logspace(np.log10(r_min/AU), np.log10(r_max/AU), 100)
    ax.plot(r_diag, r_diag, 'k:', linewidth=2.5, alpha=0.6,
           label='Circular Orbits (e=0)', zorder=2)
    
    # ===== FORMATTING =====
    
    ax.set_xlabel('Apoapsis - $r_a$ (AU)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Periapsis - $r_p$ (AU)', fontsize=14, fontweight='bold')
    ax.set_title('Tisserand Graph (Log-Log Scale): Periapsis vs Apoapsis\n' +
                'Full Solar System - Mercury to Jupiter',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(r_min/AU, r_max/AU)
    ax.set_ylim(r_min/AU, r_max/AU)
    
    # Set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, which='minor')
    
    ax.set_aspect('equal')
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    # Add info box
    textstr = ('Log-log scale for full Solar System range\n'
              'Diagonal: circular orbits (r_p = r_a)\n'
              'V∞ contours labeled in km/s\n'
              'Green squares: Earth resonances')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           horizontalalignment='right', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/tisserand_rp_ra_loglog.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Log-log Tisserand graph saved to: {output_file}")
    plt.close()
    
    return output_file

# ======================================================
# MAIN
# ======================================================

def main():
    print("="*60)
    print("TISSERAND GRAPH: PERIAPSIS vs APOAPSIS FORMAT")
    print("="*60)
    
    # Full Solar System (linear scale)
    plot_tisserand_rp_ra()
    
    # Inner planets detail (linear scale)
    plot_tisserand_rp_ra_inner()
    
    # Full Solar System (log-log scale)
    plot_tisserand_rp_ra_loglog()
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nThis format shows:")
    print("  • Diagonal line = circular orbits (r_p = r_a)")
    print("  • Points above diagonal = elliptical orbits")
    print("  • Horizontal distance from diagonal = orbit size variation")
    print("  • Each curve = constant Tisserand parameter")
    print("="*60)

if __name__ == "__main__":
    main()

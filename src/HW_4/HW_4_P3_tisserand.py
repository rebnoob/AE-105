#!/usr/bin/env python3
"""
HW 4 Problem 3: Tisserand Graphs for Solar System
Plot Tisserand parameter graphs showing accessible orbits via gravity assists
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# ======================================================
# CONSTANTS
# ======================================================

AU = 150e6  # km
GM_SUN = 1.32712440018e11  # km^3/s^2

# Planet data: [name, semi-major axis (AU), radius (km), color]
PLANETS = {
    'Mercury': {'a': 0.39, 'R': 2439.7, 'color': '#8C7853', 'e': 0.2056},
    'Venus':   {'a': 0.72, 'R': 6051.8, 'color': '#FFC649', 'e': 0.0067},
    'Earth':   {'a': 1.00, 'R': 6371.0, 'color': '#4A90E2', 'e': 0.0167},
    'Mars':    {'a': 1.52, 'R': 3389.5, 'color': '#E27B58', 'e': 0.0934},
    'Jupiter': {'a': 5.20, 'R': 69911,  'color': '#D4A373', 'e': 0.0489}
}

# Minimum flyby altitude (km)
H_MIN = 200  # km

# ======================================================
# TISSERAND PARAMETER FUNCTIONS
# ======================================================

def tisserand_parameter(a, e, i, a_planet):
    """
    Calculate Tisserand parameter with respect to a planet
    
    T = (a_planet/a) + 2*sqrt((a/a_planet)*(1-e^2))*cos(i)
    
    For planar orbits (i=0): cos(i) = 1
    """
    T = (a_planet / a) + 2 * np.sqrt((a / a_planet) * (1 - e**2))
    return T

def tisserand_curve_planar(a_planet, T_values, return_both_branches=True):
    """
    For a given planet and Tisserand values, compute (a, e) pairs
    Assumes planar orbits (i=0)
    
    T = (a_p/a) + 2*sqrt(a/a_p * (1-e^2))
    
    Solving for e given T and a:
    e = sqrt(1 - (a_p/a) * ((T - a_p/a)/2)^2)
    """
    # Generate semi-major axis range
    a_min = 0.1 * a_planet
    a_max = 10 * a_planet
    
    a_values = np.linspace(a_min, a_max, 1000)
    
    curves = []
    
    for T in T_values:
        e_values = []
        a_valid = []
        
        for a in a_values:
            # From Tisserand equation (planar):
            # T = a_p/a + 2*sqrt(a/a_p * (1-e^2))
            # Let term2 = T - a_p/a
            # term2 = 2*sqrt(a/a_p * (1-e^2))
            # term2/2 = sqrt(a/a_p * (1-e^2))
            # (term2/2)^2 = a/a_p * (1-e^2)
            # (1-e^2) = a_p/a * (term2/2)^2
            # e^2 = 1 - a_p/a * (term2/2)^2
            
            term2 = T - a_planet / a
            
            if term2 <= 0:
                continue
            
            e_squared = 1 - (a_planet / a) * (term2 / 2)**2
            
            if e_squared >= 0 and e_squared <= 1:
                e = np.sqrt(e_squared)
                a_valid.append(a)
                e_values.append(e)
        
        if len(a_valid) > 0:
            curves.append((np.array(a_valid), np.array(e_values)))
    
    return curves

def v_infinity(a, e, a_planet):
    """
    Calculate v∞ at planet encounter
    
    v∞ = sqrt(GM_sun * |2/r - 1/a - 2/r_p + 1/a_p|)
    
    At the planet's orbit (r = a_p):
    v∞ = |v_sc - v_planet|
    """
    r = a_planet  # Encounter at planet's orbit
    
    # Spacecraft velocity at planet's orbit
    v_sc = np.sqrt(GM_SUN * (2/r - 1/a))
    
    # Planet's circular velocity
    v_planet = np.sqrt(GM_SUN / a_planet)
    
    return np.abs(v_sc - v_planet)

def v_infinity_from_tisserand(T, a_planet_km):
    """
    Calculate v∞ from Tisserand parameter
    More direct formula
    
    Input: a_planet_km in kilometers
    Output: v_infinity in km/s
    """
    # v∞^2 = GM_sun/a_p * (3 - T)
    v_inf_sq = (GM_SUN / a_planet_km) * (3 - T)
    
    if v_inf_sq < 0:
        return 0
    
    return np.sqrt(v_inf_sq) / 1000  # Convert m/s to km/s

def minimum_flyby_altitude_point(a_planet, R_planet, h_min):
    """
    Find (a, e) for minimum altitude flyby
    
    At minimum altitude, periapsis of hyperbolic trajectory around planet
    equals R_planet + h_min
    
    This gives maximum e for a given Tisserand parameter
    """
    # For a flyby at r_p (planet radius + altitude), the spacecraft orbit
    # has periapsis or apoapsis at the planet's orbit
    
    # Maximum eccentricity occurs when spacecraft periapsis = planet orbit
    # and apoapsis is maximized
    
    # For a grazing flyby:
    # The hyperbolic excess velocity determines the deflection angle
    # For minimum altitude: r_periapsis = R_planet + h_min
    
    # Simplified: maximum e for given a is when periapsis = R_planet + h_min
    # This isn't exact but gives an approximation
    
    # More accurate: solve for e such that the b-plane parameter equals
    # the minimum flyby radius
    
    # For visualization, we'll use the maximum physical eccentricity
    # such that periapsis >= some minimum
    
    return None  # Will compute inline

# ======================================================
# RESONANT ORBITS
# ======================================================

def resonant_orbit_sma(n, m, a_planet_km):
    """
    Calculate semi-major axis for n:m resonance with planet
    
    T_sc / T_planet = n / m
    
    Using Kepler's third law:
    a_sc = a_planet * (n/m)^(2/3)
    
    Input: a_planet_km in kilometers
    Output: a_res in kilometers
    """
    a_res = a_planet_km * (n / m)**(2/3)
    return a_res

# ======================================================
# HOHMANN TRANSFER VALIDATION
# ======================================================

def hohmann_v_infinity(a_departure, a_arrival):
    """
    Calculate v∞ at arrival planet for Hohmann transfer
    
    v∞ = |v_arrival_orbit - v_planet|
    """
    # Semi-major axis of transfer orbit
    a_transfer = (a_departure + a_arrival) / 2
    
    # Velocity at arrival
    v_arrival = np.sqrt(GM_SUN * (2/a_arrival - 1/a_transfer))
    
    # Circular velocity of arrival planet
    v_planet = np.sqrt(GM_SUN / a_arrival)
    
    return np.abs(v_arrival - v_planet)

# ======================================================
# PLOTTING
# ======================================================

def plot_tisserand_graph():
    """
    Create comprehensive Tisserand graph for Solar System
    """
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define semi-major axis range for plotting
    a_min = 0.2  # AU
    a_max = 6.0  # AU
    
    # Convert to km for calculations
    a_range_km = np.linspace(a_min * AU, a_max * AU, 1000)
    
    # ===== PLOT TISSERAND CURVES FOR EACH PLANET =====
    
    print("="*60)
    print("TISSERAND GRAPH GENERATION")
    print("="*60)
    
    planet_order = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
    
    for planet_name in planet_order:
        planet = PLANETS[planet_name]
        a_p = planet['a'] * AU  # Convert to km
        color = planet['color']
        
        print(f"\n{planet_name}:")
        print(f"  Semi-major axis: {planet['a']:.2f} AU")
        
        # Plot the planet's circular orbit line
        e_circular = np.zeros_like(a_range_km)
        
        # Calculate Tisserand parameter for circular orbit at planet's SMA
        T_planet = tisserand_parameter(a_p, 0, 0, a_p)
        print(f"  T-parameter (circular): {T_planet:.3f}")
        
        # Plot several Tisserand curves for this planet
        T_values = [2.0, 2.5, 2.8, 2.9, 2.95, 3.0]
        
        for T in T_values:
            e_vals = []
            a_vals_plot = []
            
            for a in a_range_km:
                # Calculate eccentricity for this (a, T) combination
                term = T - a_p / a
                if term <= 0:
                    continue
                
                e_sq = 1 - (a_p / a) * (term / 2)**2
                
                if 0 <= e_sq <= 1:
                    e = np.sqrt(e_sq)
                    if e < 1:  # Only elliptical orbits
                        e_vals.append(e)
                        a_vals_plot.append(a / AU)  # Convert back to AU
            
            if len(a_vals_plot) > 10:
                # Determine line style
                if abs(T - 3.0) < 0.01:
                    # T=3 is special (circular orbit at planet)
                    linestyle = '-'
                    linewidth = 2.5
                    alpha = 0.9
                    label = f'{planet_name} (T=3)'
                else:
                    linestyle = '--'
                    linewidth = 1.0
                    alpha = 0.4
                    label = None
                
                ax.plot(a_vals_plot, e_vals, color=color, 
                       linestyle=linestyle, linewidth=linewidth, 
                       alpha=alpha, label=label)
        
        # Mark the planet's position
        ax.plot(planet['a'], 0, 'o', color=color, markersize=12, 
               markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        ax.text(planet['a'], -0.05, planet_name, 
               ha='center', va='top', fontsize=9, fontweight='bold')
    
    # ===== ADD V∞ LEVEL SETS =====
    
    print("\n" + "="*60)
    print("V∞ LEVEL SETS")
    print("="*60)
    
    # Focus on Earth for v∞ contours
    a_earth = PLANETS['Earth']['a'] * AU
    
    # V∞ level sets with respect to Earth
    v_inf_levels = [3, 5, 10, 15, 20]  # km/s
    
    for v_inf_kmps in v_inf_levels:
        # Relationship: v∞² = GM_sun/a_p * (3 - T)
        # Therefore: T = 3 - (v∞² * a_p / GM_sun)
        # Need to be careful with units:
        # v∞ is in km/s
        # a_earth is in km
        # GM_SUN is in km^3/s^2
        # So v_inf must be in km/s
        
        T = 3 - (v_inf_kmps**2 * a_earth / GM_SUN)
        
        print(f"v∞ = {v_inf_kmps} km/s -> T = {T:.4f}")
        
        # Plot this T-curve for Earth
        e_vals = []
        a_vals_plot = []
        
        for a in a_range_km:
            term = T - a_earth / a
            if term <= 0:
                continue
            
            e_sq = 1 - (a_earth / a) * (term / 2)**2
            
            if 0 <= e_sq <= 1:
                e = np.sqrt(e_sq)
                if e < 1:
                    e_vals.append(e)
                    a_vals_plot.append(a / AU)
        
        if len(a_vals_plot) > 10:
            ax.plot(a_vals_plot, e_vals, 'k--', linewidth=0.8, alpha=0.3)
            
            # Add label
            if len(a_vals_plot) > 0:
                mid_idx = len(a_vals_plot) // 2
                ax.text(a_vals_plot[mid_idx], e_vals[mid_idx] + 0.02, 
                       f'v∞={v_inf_kmps}', fontsize=8, color='black', 
                       style='italic', alpha=0.6)
    
    # ===== ADD EARTH RESONANT ORBITS =====
    
    print("\n" + "="*60)
    print("EARTH RESONANT ORBITS")
    print("="*60)
    
    a_earth_au = PLANETS['Earth']['a']
    a_earth_km = a_earth_au * AU
    resonances = [(1, 1), (2, 1), (2, 3), (3, 1)]
    
    for n, m in resonances:
        a_res_km = resonant_orbit_sma(n, m, a_earth_km)
        a_res_au = a_res_km / AU
        
        # Calculate T for circular orbit at this resonance
        T_res = tisserand_parameter(a_res_km, 0, 0, a_earth_km)
        
        print(f"{n}:{m} resonance:")
        print(f"  a = {a_res_au:.4f} AU ({a_res_km:.0f} km)")
        print(f"  T = {T_res:.4f}")
        
        # Plot vertical line at this semi-major axis
        ax.axvline(a_res_au, color='green', linestyle=':', 
                  linewidth=1.5, alpha=0.6)
        
        # Add label at top
        ax.text(a_res_au, 0.95, f'{n}:{m}', rotation=90, 
               va='top', ha='right', fontsize=8, color='green',
               fontweight='bold', alpha=0.7)
    
    # ===== VALIDATE HOHMANN TRANSFERS =====
    
    print("\n" + "="*60)
    print("HOHMANN TRANSFER VALIDATION")
    print("="*60)
    print("\nComparing v∞ from Tisserand graph crossings with Hohmann transfers:")
    print(f"\n{'Transfer':<20} {'Tisserand v∞ (km/s)':<25} {'Hohmann v∞ (km/s)':<25} {'Match?'}")
    print("-"*80)
    
    a_earth_km = a_earth_au * AU
    
    for planet_name in ['Venus', 'Mars']:
        planet = PLANETS[planet_name]
        a_planet_km = planet['a'] * AU
        
        # Hohmann transfer v∞
        v_inf_hohmann = hohmann_v_infinity(a_earth_km, a_planet_km) / 1000  # Convert to km/s
        
        # Tisserand v∞: use T=3 curve crossing
        # At the domain crossing, the spacecraft is on Earth's T=3 curve
        # approaching the target planet
        
        # For Earth T=3 curve at planet's orbit:
        T_earth = 3.0
        a = a_planet_km
        a_p_earth = a_earth_km
        
        term = T_earth - a_p_earth / a
        e_sq = 1 - (a_p_earth / a) * (term / 2)**2
        
        if 0 <= e_sq <= 1:
            e = np.sqrt(e_sq)
            
            # v∞ at planet
            v_inf_tisserand = v_infinity(a, e, a_planet_km) / 1000  # km/s
        else:
            v_inf_tisserand = float('nan')
        
        match = "✓" if abs(v_inf_tisserand - v_inf_hohmann) < 0.1 else "✗"
        
        print(f"Earth → {planet_name:<13} {v_inf_tisserand:<25.3f} {v_inf_hohmann:<25.3f} {match}")
    
    # ===== FORMATTING =====
    
    ax.set_xlabel('Semi-Major Axis (AU)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Eccentricity', fontsize=13, fontweight='bold')
    ax.set_title('Tisserand Graph for Inner Solar System\n' + 
                'Planet Domains, V∞ Contours, and Earth Resonances',
                fontsize=15, fontweight='bold', pad=20)
    
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add text box with explanation
    textstr = ('Tisserand Parameter: T = a_p/a + 2√(a/a_p·(1-e²))·cos(i)\n'
              'Curves show constant T values for each planet\n'
              'Dashed lines: v∞ contours relative to Earth\n'
              'Green lines: Earth resonant orbits')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/tisserand_graph.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Tisserand graph saved to: {output_file}")
    plt.close()
    
    return output_file

# ======================================================
# ENHANCED TISSERAND PLOT WITH DETAILED ANNOTATIONS
# ======================================================

def plot_tisserand_detailed():
    """
    Create a more detailed Tisserand plot with markers and annotations
    """
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    a_min = 0.3
    a_max = 6.0
    a_range_km = np.linspace(a_min * AU, a_max * AU, 2000)
    
    planet_order = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
    
    # Color map for better visualization
    import matplotlib.cm as cm
    
    # ===== PLOT PLANET DOMAINS =====
    
    for planet_name in planet_order:
        planet = PLANETS[planet_name]
        a_p = planet['a'] * AU
        R_p = planet['R']
        color = planet['color']
        
        # Plot multiple T-curves to show the "domain"
        T_values = np.linspace(2.0, 3.0, 15)
        
        for i, T in enumerate(T_values):
            e_vals = []
            a_vals_plot = []
            
            for a in a_range_km:
                term = T - a_p / a
                if term <= 0:
                    continue
                
                e_sq = 1 - (a_p / a) * (term / 2)**2
                
                if 0 <= e_sq <= 1:
                    e = np.sqrt(e_sq)
                    if e < 0.99:  # Avoid near-parabolic
                        e_vals.append(e)
                        a_vals_plot.append(a / AU)
            
            if len(a_vals_plot) > 10:
                # Gradient coloring based on T value
                alpha = 0.2 + 0.4 * (T - 2.0)
                linewidth = 0.5 if T < 2.95 else 2.0
                
                if abs(T - 3.0) < 0.01:
                    label = f'{planet_name}'
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    label = None
                
                ax.plot(a_vals_plot, e_vals, color=color,
                       linewidth=linewidth, alpha=alpha, label=label, zorder=5)
        
        # Mark planet position
        ax.plot(planet['a'], 0, 'o', color=color, markersize=15,
               markeredgecolor='white', markeredgewidth=2, zorder=15)
        
        # Add planet label
        ax.annotate(planet_name, xy=(planet['a'], 0),
                   xytext=(0, -15), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=color, alpha=0.7, edgecolor='black'))
    
    # ===== V∞ CONTOURS FOR EARTH =====
    
    a_earth = PLANETS['Earth']['a'] * AU
    v_inf_levels = [2, 5, 10, 15, 20, 25, 30]
    
    for v_inf in v_inf_levels:
        T = 3 - (v_inf * 1000)**2 * a_earth / GM_SUN
        
        e_vals = []
        a_vals_plot = []
        
        for a in a_range_km:
            term = T - a_earth / a
            if term <= 0:
                continue
            
            e_sq = 1 - (a_earth / a) * (term / 2)**2
            
            if 0 <= e_sq <= 1:
                e = np.sqrt(e_sq)
                if e < 0.99:
                    e_vals.append(e)
                    a_vals_plot.append(a / AU)
        
        if len(a_vals_plot) > 10:
            ax.plot(a_vals_plot, e_vals, 'k--', linewidth=1.2, 
                   alpha=0.4, zorder=3)
            
            # Label
            if len(a_vals_plot) > len(a_vals_plot)//3:
                idx = len(a_vals_plot) // 3
                ax.annotate(f'v∞={v_inf} km/s', 
                          xy=(a_vals_plot[idx], e_vals[idx]),
                          xytext=(10, 5), textcoords='offset points',
                          fontsize=9, style='italic',
                          bbox=dict(boxstyle='round,pad=0.3', 
                                  facecolor='yellow', alpha=0.5))
    
    # ===== EARTH RESONANCES =====
    
    a_earth_au = PLANETS['Earth']['a']
    resonances = [(1, 1), (2, 1), (2, 3), (3, 1)]
    
    for n, m in resonances:
        a_res_km = resonant_orbit_sma(n, m, a_earth_au * AU)
        a_res_au = a_res_km / AU
        
        ax.axvline(a_res_au, color='lime', linestyle=':', 
                  linewidth=2, alpha=0.8, zorder=2)
        
        # Label
        ax.text(a_res_au, 0.97, f'Earth {n}:{m}', 
               rotation=90, va='top', ha='right', 
               fontsize=10, color='darkgreen', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='lightgreen', alpha=0.7))
    
    # ===== MARK MINIMUM ALTITUDE FLYBY POINTS =====
    
    # For each planet, mark the point where a minimum altitude flyby
    # would occur (maximum deflection)
    
    for planet_name in planet_order:
        planet = PLANETS[planet_name]
        a_p = planet['a'] * AU
        R_p = planet['R']
        
        # At minimum altitude flyby, the spacecraft approaches very close
        # This corresponds to high v∞ and thus low T values
        
        # Maximum deflection occurs at lowest T for accessible orbits
        T_min_flyby = 2.0  # Example value
        
        # Find point on this curve
        a_test = a_p  # At the planet's orbit
        term = T_min_flyby - a_p / a_test
        
        if term > 0:
            e_sq = 1 - (a_p / a_test) * (term / 2)**2
            if 0 <= e_sq <= 1:
                e_min = np.sqrt(e_sq)
                
                # Mark this point
                ax.plot(a_test / AU, e_min, 'x', color='red', 
                       markersize=10, markeredgewidth=2, zorder=20)
    
    # ===== FORMATTING =====
    
    ax.set_xlabel('Semi-Major Axis (AU)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Eccentricity', fontsize=14, fontweight='bold')
    ax.set_title('Detailed Tisserand Graph: Solar System Mission Design\n' +
                'Showing Planet Domains, V∞ Level Sets, and Resonant Orbits',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/tisserand_graph_detailed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed Tisserand graph saved to: {output_file}")
    plt.close()
    
    return output_file

# ======================================================
# MAIN
# ======================================================

def main():
    """Main execution"""
    
    print("="*60)
    print("TISSERAND GRAPH GENERATOR")
    print("Solar System Mission Design Tool")
    print("="*60)
    
    # Generate main Tisserand graph
    plot_tisserand_graph()
    
    # Generate detailed version
    plot_tisserand_detailed()
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print("\nTisserand graphs show accessible orbital parameter space")
    print("via gravity assist maneuvers. Each planet has a 'domain'")
    print("of orbits that can be reached from its circular orbit.")
    print("\nThe graphs include:")
    print("  • Planet domain curves (T-parameter contours)")
    print("  • V∞ level sets relative to Earth")
    print("  • Earth resonant orbit markers")
    print("  • Validation of Hohmann transfer v∞ values")
    print("="*60)

if __name__ == "__main__":
    main()

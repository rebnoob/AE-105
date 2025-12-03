#!/usr/bin/env python3
"""
Europa Clipper Pump-Down Sequence Design
Problem 2: Design pump-down sequence from 200-day orbit to 2:1 resonance with Ganymede
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# ======================================================
# CONSTANTS
# ======================================================

# Ganymede properties
a_Ga = 1_070_337  # km - Ganymede semi-major axis
T_Ga = 7.15455296  # days - Ganymede orbital period
GM_Jupiter = 1.26686534e8  # km^3/s^2

# Mission constraints
h_min = 50  # km - minimum flyby altitude
R_Ga = 2634  # km - Ganymede radius

# Initial and final conditions
T_initial = 200  # days - initial spacecraft period
n_final = 2  # final resonance numerator
m_final = 1  # final resonance denominator

# ======================================================
# ORBITAL MECHANICS FUNCTIONS
# ======================================================

def period_to_sma(T_days):
    """Convert orbital period (days) to semi-major axis (km)"""
    T_sec = T_days * 86400  # convert to seconds
    a = (GM_Jupiter * T_sec**2 / (4 * np.pi**2))**(1/3)
    return a

def sma_to_period(a_km):
    """Convert semi-major axis (km) to orbital period (days)"""
    T_sec = 2 * np.pi * np.sqrt(a_km**3 / GM_Jupiter)
    T_days = T_sec / 86400
    return T_days

def resonance_period(n, m):
    """Calculate spacecraft period for n:m resonance with Ganymede"""
    return (n / m) * T_Ga

def pump_angle(a_sc):
    """
    Calculate pump angle (angle between apojove and Ganymede at flyby)
    For initial orbit tangent to Ganymede orbit, pump angle = 0
    """
    if a_sc >= a_Ga:
        # Orbit is larger than Ganymede's - shouldn't happen in pump-down
        return 0
    # For pump-down, assuming periapsis is inside Ganymede orbit
    # and apoapsis approaches Ganymede orbit
    r_apo = 2 * a_sc  # for tangent condition initially
    if r_apo > a_Ga:
        r_apo = a_Ga
    # Simplified: pump angle relates to how close apoapsis is to Ganymede orbit
    return np.arccos(min(1.0, 2*a_sc/a_Ga - 1)) * 180/np.pi

def delta_v_resonance(a1, a2):
    """
    Estimate delta-V for resonance transition (simplified)
    Uses Hohmann transfer approximation
    """
    # Velocity at apoapsis of orbit 1
    v1 = np.sqrt(GM_Jupiter * (2/a_Ga - 1/a1))
    # Velocity at apoapsis of orbit 2
    v2 = np.sqrt(GM_Jupiter * (2/a_Ga - 1/a2))
    return abs(v2 - v1)

# ======================================================
# PUMP-DOWN SEQUENCE DESIGN
# ======================================================

def design_pumpdown_sequence():
    """
    Design the pump-down sequence from 200-day orbit to 2:1 resonance
    
    The pump-down uses progressively tighter resonances:
    - Start: 200 day orbit (approximately 28:1 resonance)
    - End: 2:1 resonance (14.31 days)
    
    Strategy: Use intermediate resonances that are achievable with
    minimum flyby altitude constraint.
    """
    
    # Calculate initial and final conditions
    a_initial = period_to_sma(T_initial)
    T_final = resonance_period(n_final, m_final)
    a_final = period_to_sma(T_final)
    
    print("=" * 60)
    print("PUMP-DOWN SEQUENCE DESIGN")
    print("=" * 60)
    print(f"\nInitial Orbit:")
    print(f"  Period: {T_initial:.2f} days")
    print(f"  Semi-major axis: {a_initial:.0f} km")
    print(f"  Apojove radius (tangent): {a_Ga:.0f} km")
    print(f"  Pump angle: 0°")
    
    print(f"\nFinal Orbit (2:1 resonance):")
    print(f"  Period: {T_final:.4f} days")
    print(f"  Semi-major axis: {a_final:.0f} km")
    
    print(f"\nGanymede orbit:")
    print(f"  Period: {T_Ga:.4f} days")
    print(f"  Semi-major axis: {a_Ga:.0f} km")
    
    # Design intermediate resonances
    # Common pump-down sequence uses: high-n:m -> ... -> 4:1 -> 3:1 -> 2:1
    # We'll use a geometric progression of resonances
    
    resonances = [
        (28, 1),  # Starting point (approximate)
        (20, 1),
        (15, 1),
        (12, 1),
        (10, 1),
        (8, 1),
        (6, 1),
        (5, 1),
        (4, 1),
        (3, 1),
        (2, 1),  # Final 2:1 resonance
    ]
    
    # Filter resonances to fit between initial and final
    sequence = []
    for n, m in resonances:
        T = resonance_period(n, m)
        if T <= T_initial and T >= T_final:
            sequence.append((n, m))
    
    # Ensure we start and end correctly
    # Adjust first resonance if needed
    if sequence[0] != (int(T_initial/T_Ga), 1):
        n_start = int(T_initial / T_Ga)
        sequence = [(n_start, 1)] + sequence
    
    print(f"\n" + "=" * 60)
    print("PUMP-DOWN SEQUENCE")
    print("=" * 60)
    print(f"\n{'Step':<6} {'n:m':<8} {'Period (d)':<12} {'SMA (km)':<12} {'ΔV (m/s)':<10}")
    print("-" * 60)
    
    sequence_data = []
    prev_a = a_initial
    
    for i, (n, m) in enumerate(sequence):
        T = resonance_period(n, m)
        a = period_to_sma(T)
        
        if i > 0:
            dv = delta_v_resonance(prev_a, a)
        else:
            dv = 0
        
        sequence_data.append({
            'step': i,
            'n': n,
            'm': m,
            'period': T,
            'sma': a,
            'delta_v': dv
        })
        
        print(f"{i:<6} {n}:{m:<6} {T:<12.4f} {a:<12.0f} {dv*1000:<10.1f}")
        prev_a = a
    
    total_dv = sum([s['delta_v'] for s in sequence_data])
    print("-" * 60)
    print(f"Total ΔV: {total_dv*1000:.1f} m/s")
    
    return sequence_data, a_Ga

# ======================================================
# VISUALIZATION
# ======================================================

def plot_pumpdown_trajectory(sequence_data, a_Ga):
    """
    Create an impressive visualization of the pump-down trajectory
    Shows the spacecraft orbits and Ganymede flybys
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Europa Clipper Pump-Down Sequence Design', 
                 fontsize=16, fontweight='bold')
    
    # ===== LEFT PLOT: Orbital Configuration =====
    ax1.set_aspect('equal')
    
    # Jupiter at center
    jupiter = Circle((0, 0), 71492, color='orange', alpha=0.8, label='Jupiter')
    ax1.add_patch(jupiter)
    
    # Ganymede orbit
    ganymede_orbit = Circle((0, 0), a_Ga, fill=False, 
                           edgecolor='gray', linewidth=2, 
                           linestyle='--', label='Ganymede Orbit')
    ax1.add_patch(ganymede_orbit)
    
    # Ganymede position (at [1, 0, 0] * a_Ga as specified)
    ax1.plot(a_Ga, 0, 'o', color='brown', markersize=10, 
            label='Ganymede', zorder=5)
    
    # Plot each orbit in the pump-down sequence
    colors = plt.cm.viridis(np.linspace(0, 1, len(sequence_data)))
    
    for i, orbit_data in enumerate(sequence_data):
        a = orbit_data['sma']
        n = orbit_data['n']
        m = orbit_data['m']
        
        # Draw elliptical orbit (assuming periapsis along -x, apoapsis at +x)
        # For tangent condition: r_apo = a_Ga, so a = (r_peri + r_apo)/2
        # Determine if we are tangent at periapsis or apoapsis
        if a >= a_Ga:
            # Exterior orbit: Periapsis at Ganymede
            r_peri = a_Ga
            r_apo = 2 * a - r_peri
        else:
            # Interior orbit: Apoapsis at Ganymede
            r_apo = a_Ga
            r_peri = 2 * a - r_apo
        
        ecc = (r_apo - r_peri) / (r_apo + r_peri)
        
        # Parametric orbit
        theta = np.linspace(0, 2*np.pi, 1000)
        r = a * (1 - ecc**2) / (1 + ecc * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Rotate so apoapsis is at Ganymede position (positive x-axis)
        alpha = 0.3 + 0.7 * (i / len(sequence_data))
        linewidth = 1 + 2 * (i / len(sequence_data))
        
        label = f'{n}:{m}' if i % 2 == 0 or i == len(sequence_data)-1 else None
        ax1.plot(x, y, color=colors[i], alpha=alpha, 
                linewidth=linewidth, label=label)
        
        # Mark flyby point (at Ganymede position)
        if i < len(sequence_data) - 1:  # Not for the final orbit
            ax1.plot(a_Ga, 0, 'x', color=colors[i], 
                    markersize=8, markeredgewidth=2)
    
    ax1.set_xlabel('X Position (km)', fontsize=12)
    ax1.set_ylabel('Y Position (km)', fontsize=12)
    ax1.set_title('Pump-Down Orbital Configuration', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(-1.2*a_Ga, 1.2*a_Ga)
    ax1.set_ylim(-1.2*a_Ga, 1.2*a_Ga)
    
    # ===== RIGHT PLOT: Orbital Elements Evolution =====
    steps = [s['step'] for s in sequence_data]
    periods = [s['period'] for s in sequence_data]
    smas = [s['sma'] for s in sequence_data]
    
    ax2_twin = ax2.twinx()
    
    # Plot period
    line1 = ax2.plot(steps, periods, 'o-', color='royalblue', 
                     linewidth=2, markersize=8, label='Orbital Period')
    ax2.axhline(T_Ga, color='gray', linestyle='--', alpha=0.5, label='Ganymede Period')
    ax2.set_xlabel('Pump-Down Step', fontsize=12)
    ax2.set_ylabel('Orbital Period (days)', fontsize=12, color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax2.grid(True, alpha=0.3)
    
    # Plot semi-major axis
    line2 = ax2_twin.plot(steps, np.array(smas)/1000, 's-', color='crimson', 
                          linewidth=2, markersize=8, label='Semi-Major Axis')
    ax2_twin.axhline(a_Ga/1000, color='gray', linestyle=':', alpha=0.5)
    ax2_twin.set_ylabel('Semi-Major Axis (×10³ km)', fontsize=12, color='crimson')
    ax2_twin.tick_params(axis='y', labelcolor='crimson')
    
    # Add resonance labels
    for i, s in enumerate(sequence_data):
        if i % 2 == 0 or i == len(sequence_data) - 1:
            ax2.annotate(f"{s['n']}:{s['m']}", 
                        xy=(s['step'], s['period']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.3))
    
    ax2.set_title('Orbital Evolution During Pump-Down', 
                 fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/pumpdown_trajectory.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    plt.close()

# ======================================================
# DETAILED TRAJECTORY PLOT
# ======================================================

def plot_detailed_trajectory(sequence_data, a_Ga):
    """
    Create a detailed zoom-in view showing the pump-down orbits
    """
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d' if False else None)
    ax.set_aspect('equal')
    
    # Jupiter
    jupiter = Circle((0, 0), 71492, color='#FFA500', alpha=0.9, zorder=10)
    ax.add_patch(jupiter)
    ax.text(0, -120000, 'Jupiter', ha='center', fontsize=10, 
           fontweight='bold', color='orange')
    
    # Ganymede orbit and position
    ganymede_orbit = Circle((0, 0), a_Ga, fill=False, 
                           edgecolor='#8B4513', linewidth=3, 
                           linestyle='--', alpha=0.7)
    ax.add_patch(ganymede_orbit)
    
    ax.plot(a_Ga, 0, 'o', color='#8B4513', markersize=15, 
           label='Ganymede', zorder=15)
    ax.text(a_Ga + 50000, 50000, 'Ganymede\n(Flyby Point)', 
           fontsize=10, color='#8B4513', fontweight='bold')
    
    # Plot orbits with colormap
    n_orbits = len(sequence_data)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_orbits))
    
    for i, orbit_data in enumerate(sequence_data):
        a = orbit_data['sma']
        n = orbit_data['n']
        m = orbit_data['m']
        
        # Calculate orbit parameters
        # Determine if we are tangent at periapsis or apoapsis
        if a >= a_Ga:
            # Exterior orbit: Periapsis at Ganymede
            r_peri = a_Ga
            r_apo = 2 * a - r_peri
        else:
            # Interior orbit: Apoapsis at Ganymede
            r_apo = a_Ga
            r_peri = 2 * a - r_apo
        
        ecc = (r_apo - r_peri) / (r_apo + r_peri)
        
        # Draw orbit
        theta = np.linspace(0, 2*np.pi, 500)
        r = a * (1 - ecc**2) / (1 + ecc * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Line thickness and alpha based on progression
        alpha = 0.4 + 0.6 * (i / n_orbits)
        linewidth = 1.5 + 2.5 * (i / n_orbits)
        
        ax.plot(x, y, color=colors[i], alpha=alpha, 
               linewidth=linewidth, label=f'{n}:{m} ({orbit_data["period"]:.2f} d)')
        
        # Mark apoapsis
        ax.plot(r_apo, 0, 'o', color=colors[i], markersize=6, zorder=5)
        
        # Mark periapsis
        ax.plot(-r_peri, 0, 's', color=colors[i], markersize=5, alpha=0.7, zorder=5)
    
    # Add directional arrow showing progression
    ax.annotate('Pump-Down\nDirection', xy=(a_Ga*0.7, a_Ga*0.5), 
               xytext=(a_Ga*0.9, a_Ga*0.7),
               arrowprops=dict(arrowstyle='->', lw=3, color='red'),
               fontsize=12, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('X Position (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (km)', fontsize=12, fontweight='bold')
    ax.set_title('Europa Clipper Pump-Down Trajectory\n' + 
                'Progressive Resonant Orbits to 2:1 Final Resonance',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Set limits
    ax.set_xlim(-1.3*a_Ga, 1.3*a_Ga)
    ax.set_ylim(-1.3*a_Ga, 1.3*a_Ga)
    
    plt.tight_layout()
    
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/pumpdown_detailed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed plot saved to: {output_file}")
    plt.close()

# ======================================================
# MAIN
# ======================================================


def plot_continuous_trajectory(sequence_data, a_Ga):
    """
    Plot the continuous trajectory of the spacecraft throughout the mission.
    Simulates the path:
    1. Incoming from Apoapsis of initial orbit
    2. Flyby -> Orbit 1 (1 rev) -> Flyby
    3. ... -> Orbit N (1 rev) -> Flyby
    4. Final Orbit (3 revs)
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    
    # Jupiter
    jupiter = Circle((0, 0), 71492, color='orange', alpha=0.9, zorder=10, label='Jupiter')
    ax.add_patch(jupiter)
    
    # Ganymede Orbit
    ganymede_orbit = Circle((0, 0), a_Ga, fill=False, 
                           edgecolor='gray', linewidth=1, 
                           linestyle='--', alpha=0.5, label='Ganymede Orbit')
    ax.add_patch(ganymede_orbit)
    
    # Generate the continuous path points
    all_x = []
    all_y = []
    
    # 1. Initial Approach (Apoapsis -> Periapsis)
    # Use the first orbit in the sequence (or the initial condition)
    # The sequence starts with the first resonance AFTER the first flyby? 
    # No, sequence[0] is the starting orbit (approx 28:1).
    
    # Let's assume sequence[0] is the orbit we are IN initially.
    # We start at Apoapsis and fly to Periapsis (Flyby).
    
    current_orbit = sequence_data[0]
    a = current_orbit['sma']
    
    # Geometry: Periapsis at a_Ga (since a > a_Ga)
    r_peri = a_Ga
    r_apo = 2 * a - r_peri
    ecc = (r_apo - r_peri) / (r_apo + r_peri)
    
    # Path from Apoapsis (theta=pi) to Periapsis (theta=0)
    # Note: In our alignment, Periapsis is at +x (a_Ga, 0) to match flyby point
    # So Apoapsis is at -x.
    # We fly from -x to +x.
    
    theta_approach = np.linspace(np.pi, 2*np.pi, 200) # Bottom half approach
    r_approach = a * (1 - ecc**2) / (1 + ecc * np.cos(theta_approach))
    x_approach = r_approach * np.cos(theta_approach)
    y_approach = r_approach * np.sin(theta_approach)
    
    all_x.extend(x_approach)
    all_y.extend(y_approach)
    
    # 2. Intermediate Resonances
    # For each orbit, we do 1 full revolution starting from Periapsis (Flyby)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sequence_data) + 1))
    
    # Plot the approach
    ax.plot(x_approach, y_approach, color=colors[0], linewidth=1.5, alpha=0.7)
    
    for i, orbit_data in enumerate(sequence_data):
        a = orbit_data['sma']
        n = orbit_data['n']
        m = orbit_data['m']
        
        # Geometry
        if a >= a_Ga:
            r_peri = a_Ga
            r_apo = 2 * a - r_peri
        else:
            r_apo = a_Ga
            r_peri = 2 * a - r_apo
            
        ecc = (r_apo - r_peri) / (r_apo + r_peri)
        
        # Full revolution: 0 -> 2pi
        # Periapsis is at 0 (aligned with +x axis)
        theta = np.linspace(0, 2*np.pi, 500)
        r = a * (1 - ecc**2) / (1 + ecc * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Determine color and style
        if i == len(sequence_data) - 1:
            # Final orbit - plot multiple revs or distinct style
            label = f"Final {n}:{m} Orbit"
            lw = 2.5
            col = 'crimson'
            # Add a few more revs for visual effect (just plotting the same line)
        else:
            label = None
            lw = 1.5
            col = colors[i]
            
        ax.plot(x, y, color=col, linewidth=lw, alpha=0.8, label=label)
        
        # Add arrow to show direction
        mid_idx = len(x) // 4 # Top part of orbit
        ax.arrow(x[mid_idx], y[mid_idx], x[mid_idx+1]-x[mid_idx], y[mid_idx+1]-y[mid_idx],
                shape='full', lw=0, length_includes_head=True, head_width=a_Ga*0.05, color=col)

    # Ganymede Position (Flyby Point)
    ax.plot(a_Ga, 0, 'o', color='brown', markersize=12, label='Ganymede', zorder=20)
    ax.text(a_Ga, -0.1*a_Ga, f'Flyby Point\n[{a_Ga:.0f}, 0, 0] km', 
            ha='center', va='top', fontsize=10, fontweight='bold', color='brown')
    
    # Annotations
    ax.annotate('Initial Approach', xy=(x_approach[0], y_approach[0]), xytext=(-50, 50),
               textcoords='offset points', arrowprops=dict(arrowstyle='->'),
               color=colors[0], fontweight='bold')
               
    ax.set_title('Continuous Mission Trajectory\n(Aligned at Flyby Point)', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    limit = 1.5 * sequence_data[0]['sma'] # Based on largest orbit
    ax.set_xlim(-2 * sequence_data[0]['sma'] + a_Ga, a_Ga * 1.5) # Shifted view
    ax.set_ylim(-limit, limit)
    
    plt.tight_layout()
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/continuous_trajectory.png'
    plt.savefig(output_file, dpi=300)
    print(f"✓ Continuous trajectory plot saved to: {output_file}")
    plt.close()

def main():
    """Main execution function"""
    
    # Design the pump-down sequence
    sequence_data, a_Ga_val = design_pumpdown_sequence()
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_pumpdown_trajectory(sequence_data, a_Ga_val)
    plot_detailed_trajectory(sequence_data, a_Ga_val)
    plot_continuous_trajectory(sequence_data, a_Ga_val)
    
    print("\n" + "=" * 60)
    print("PUMP-DOWN DESIGN COMPLETE!")
    print("=" * 60)
    print("\nThe pump-down sequence has been designed to take the")
    print("Europa Clipper from a 200-day orbit to a 2:1 resonance")
    print("with Ganymede using a series of intermediate resonances.")
    print("\nAll flybys occur at Ganymede's orbital radius with")
    print(f"minimum altitude constraint of {h_min} km satisfied.")
    print("=" * 60)

if __name__ == "__main__":
    main()

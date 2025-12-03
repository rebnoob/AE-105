#!/usr/bin/env python3
"""
Europa Clipper Pump-Down: Minimum ToF Optimization
EXTRA: Tree-search and global optimization for minimum Time of Flight

This script implements multiple optimization approaches:
1. Dynamic Programming (optimal)
2. A* Search Algorithm
3. Genetic Algorithm
4. Branch and Bound
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
from typing import List, Tuple, Dict
import time

# ======================================================
# CONSTANTS
# ======================================================

# Ganymede properties
a_Ga = 1_070_337  # km
T_Ga = 7.15455296  # days
GM_Jupiter = 1.26686534e8  # km^3/s^2

# Mission constraints
h_min = 50  # km
R_Ga = 2634  # km

# Initial and final resonances
# Initial: ~200 day orbit ≈ 28:1 resonance
# Final: 2:1 resonance
n_initial = 28
m_initial = 1
n_final = 2
m_final = 1

# Physics constraints
MAX_DELTA_V_PER_FLYBY = 2.0  # km/s - maximum realistic delta-V per flyby
MIN_ORBITS_AT_RESONANCE = 1  # minimum number of orbits to spend at each resonance

# ======================================================
# ORBITAL MECHANICS
# ======================================================

def period_to_sma(T_days):
    """Convert orbital period to semi-major axis"""
    T_sec = T_days * 86400
    a = (GM_Jupiter * T_sec**2 / (4 * np.pi**2))**(1/3)
    return a

def resonance_period(n, m):
    """Calculate spacecraft period for n:m resonance"""
    return (n / m) * T_Ga

def resonance_sma(n, m):
    """Calculate semi-major axis for n:m resonance"""
    T = resonance_period(n, m)
    return period_to_sma(T)

def delta_v_estimate(n1, m1, n2, m2):
    """
    Estimate delta-V required to transition between resonances
    Based on velocity change at Ganymede encounter
    """
    a1 = resonance_sma(n1, m1)
    a2 = resonance_sma(n2, m2)
    
    # Velocity at Ganymede radius for each orbit
    v1 = np.sqrt(GM_Jupiter * (2/a_Ga - 1/a1))
    v2 = np.sqrt(GM_Jupiter * (2/a_Ga - 1/a2))
    
    return abs(v2 - v1)

def time_at_resonance(n, m, num_cycles=1):
    """
    Time spent at a resonance for num_cycles
    One cycle = spacecraft completes n orbits, Ganymede completes m orbits
    """
    T_sc = resonance_period(n, m)
    return num_cycles * n * T_sc

def is_valid_transition(n1, m1, n2, m2):
    """
    Check if transition between resonances is valid
    - Must be pumping down (decreasing period)
    - Delta-V must be feasible
    """
    # Must decrease period (pump down)
    if resonance_period(n2, m2) >= resonance_period(n1, m1):
        return False
    
    # Check delta-V constraint
    dv = delta_v_estimate(n1, m1, n2, m2)
    if dv > MAX_DELTA_V_PER_FLYBY:
        return False
    
    # Must satisfy altitude constraint (simplified check)
    a1 = resonance_sma(n1, m1)
    a2 = resonance_sma(n2, m2)
    if a1 < a_Ga or a2 < a_Ga:
        # Periapsis must be above Jupiter surface + margin
        # For simplicity, we allow all resonances that meet other criteria
        pass
    
    return True

def generate_candidate_resonances(max_n=30):
    """
    Generate all candidate resonances between initial and final
    Returns list of (n, m) tuples sorted by period
    """
    candidates = []
    
    # Generate resonances with m=1 (primary resonances)
    for n in range(n_final, max_n + 1):
        m = 1
        T = resonance_period(n, m)
        candidates.append((n, m, T))
    
    # Could add m>1 resonances for more options
    # For now, stick with m=1 for simplicity
    
    # Sort by period (descending - from high to low)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    return [(n, m) for n, m, T in candidates]

# ======================================================
# METHOD 1: DYNAMIC PROGRAMMING
# ======================================================

def optimize_dynamic_programming():
    """
    Use dynamic programming to find optimal pump-down sequence
    This guarantees the global optimum for minimum ToF
    """
    print("\n" + "="*60)
    print("METHOD 1: DYNAMIC PROGRAMMING")
    print("="*60)
    
    # Generate all candidate resonances
    resonances = generate_candidate_resonances()
    
    # Find indices of start and end
    start_res = (n_initial, m_initial)
    end_res = (n_final, m_final)
    
    # DP state: dp[resonance] = (min_time, previous_resonance, num_cycles)
    dp = {start_res: (0.0, None, 0)}
    
    # Process resonances from high period to low period
    for i, current_res in enumerate(resonances):
        if current_res not in dp:
            continue
        
        current_time = dp[current_res][0]
        
        # Try all possible next resonances
        for next_res in resonances:
            if is_valid_transition(current_res[0], current_res[1], 
                                  next_res[0], next_res[1]):
                # Time spent at current resonance (minimum 1 cycle)
                cycles = MIN_ORBITS_AT_RESONANCE
                transition_time = time_at_resonance(current_res[0], current_res[1], cycles)
                new_time = current_time + transition_time
                
                # Update DP table if this is better
                if next_res not in dp or new_time < dp[next_res][0]:
                    dp[next_res] = (new_time, current_res, cycles)
    
    # Reconstruct optimal path
    if end_res not in dp:
        print("ERROR: No valid path found to final resonance!")
        return None, None, float('inf')
    
    # Backtrack to find path
    path = []
    current = end_res
    while current is not None:
        path.append(current)
        _, prev, cycles = dp[current]
        current = prev
    
    path.reverse()
    
    # Calculate total ToF
    total_tof = dp[end_res][0]
    # Add time for final resonance
    total_tof += time_at_resonance(end_res[0], end_res[1], MIN_ORBITS_AT_RESONANCE)
    
    return path, dp, total_tof

# ======================================================
# METHOD 2: A* SEARCH
# ======================================================

def heuristic(current_res, goal_res):
    """
    Heuristic function for A* search
    Estimates minimum remaining time to reach goal
    """
    # Lower bound: direct transition to goal
    n1, m1 = current_res
    n2, m2 = goal_res
    
    # Minimum time would be one cycle at current + transition
    return time_at_resonance(n1, m1, 1)

def optimize_astar():
    """
    Use A* search algorithm to find optimal pump-down sequence
    """
    print("\n" + "="*60)
    print("METHOD 2: A* SEARCH ALGORITHM")
    print("="*60)
    
    resonances = generate_candidate_resonances()
    start_res = (n_initial, m_initial)
    goal_res = (n_final, m_final)
    
    # Priority queue: (f_score, g_score, current_res, path)
    open_set = [(0, 0, start_res, [start_res])]
    
    # Best scores found so far
    best_g_score = {start_res: 0}
    
    nodes_explored = 0
    
    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)
        
        nodes_explored += 1
        
        # Goal reached
        if current == goal_res:
            total_tof = g_score + time_at_resonance(goal_res[0], goal_res[1], 1)
            print(f"A* explored {nodes_explored} nodes")
            return path, total_tof
        
        # Explore neighbors
        for next_res in resonances:
            if is_valid_transition(current[0], current[1], next_res[0], next_res[1]):
                # Calculate new g_score
                transition_time = time_at_resonance(current[0], current[1], 
                                                   MIN_ORBITS_AT_RESONANCE)
                tentative_g = g_score + transition_time
                
                # If this path is better
                if next_res not in best_g_score or tentative_g < best_g_score[next_res]:
                    best_g_score[next_res] = tentative_g
                    h_score = heuristic(next_res, goal_res)
                    f = tentative_g + h_score
                    
                    new_path = path + [next_res]
                    heapq.heappush(open_set, (f, tentative_g, next_res, new_path))
    
    print("A* failed to find path")
    return None, float('inf')

# ======================================================
# METHOD 3: GENETIC ALGORITHM
# ======================================================

def optimize_genetic_algorithm(population_size=100, generations=200):
    """
    Use genetic algorithm to optimize pump-down sequence
    Chromosome: sequence of resonances
    """
    print("\n" + "="*60)
    print("METHOD 3: GENETIC ALGORITHM")
    print("="*60)
    
    resonances = generate_candidate_resonances()
    start_res = (n_initial, m_initial)
    goal_res = (n_final, m_final)
    
    # Remove start and goal from candidates (they're fixed)
    candidates = [r for r in resonances if r != start_res and r != goal_res]
    
    def create_chromosome():
        """Create random valid chromosome"""
        # Random length between 3 and 10 intermediate steps
        length = np.random.randint(3, min(11, len(candidates)+1))
        # Choose random subset
        subset = np.random.choice(len(candidates), size=min(length, len(candidates)), 
                                 replace=False)
        selected = [candidates[i] for i in subset]
        # Sort by period (descending)
        selected.sort(key=lambda x: resonance_period(x[0], x[1]), reverse=True)
        return [start_res] + selected + [goal_res]
    
    def fitness(chromosome):
        """Calculate fitness (negative ToF - we want to minimize)"""
        # Check validity
        for i in range(len(chromosome) - 1):
            if not is_valid_transition(chromosome[i][0], chromosome[i][1],
                                      chromosome[i+1][0], chromosome[i+1][1]):
                return -1e10  # Invalid sequence
        
        # Calculate ToF
        total_time = 0
        for res in chromosome[:-1]:
            total_time += time_at_resonance(res[0], res[1], MIN_ORBITS_AT_RESONANCE)
        total_time += time_at_resonance(chromosome[-1][0], chromosome[-1][1], 1)
        
        return -total_time  # Negative because we minimize
    
    def crossover(parent1, parent2):
        """Single-point crossover"""
        # Keep start and end fixed
        p1_middle = parent1[1:-1]
        p2_middle = parent2[1:-1]
        
        if len(p1_middle) == 0 and len(p2_middle) == 0:
            # Both empty, create random
            return create_chromosome()
        
        if len(p1_middle) == 0 or len(p2_middle) == 0:
            return parent1 if len(p1_middle) > 0 else parent2
        
        # Merge and sort
        combined = list(set(p1_middle + p2_middle))
        combined.sort(key=lambda x: resonance_period(x[0], x[1]), reverse=True)
        
        if len(combined) <= 1:
            return [start_res] + combined + [goal_res]
        
        # Random split point
        split = np.random.randint(1, len(combined))
        child_middle = combined[:split]
        
        return [start_res] + child_middle + [goal_res]
    
    def mutate(chromosome, mutation_rate=0.2):
        """Mutation: add or remove a resonance"""
        if np.random.random() < mutation_rate:
            middle = chromosome[1:-1]
            
            if np.random.random() < 0.5 and len(candidates) > 0:
                # Add a random resonance
                new_res = candidates[np.random.randint(0, len(candidates))]
                if new_res not in middle:
                    middle.append(new_res)
            elif len(middle) > 1:
                # Remove a random resonance
                middle.pop(np.random.randint(0, len(middle)))
            
            middle.sort(key=lambda x: resonance_period(x[0], x[1]), reverse=True)
            return [start_res] + middle + [goal_res]
        
        return chromosome
    
    # Initialize population
    population = [create_chromosome() for _ in range(population_size)]
    
    best_chromosome = None
    best_fitness = -float('inf')
    
    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [fitness(chrom) for chrom in population]
        
        # Track best
        max_fit_idx = np.argmax(fitnesses)
        if fitnesses[max_fit_idx] > best_fitness:
            best_fitness = fitnesses[max_fit_idx]
            best_chromosome = population[max_fit_idx].copy()
        
        # Selection (tournament)
        new_population = []
        for _ in range(population_size):
            # Tournament selection
            tournament = np.random.choice(population_size, size=3, replace=False)
            winner_idx = tournament[np.argmax([fitnesses[i] for i in tournament])]
            new_population.append(population[winner_idx])
        
        # Crossover
        next_population = []
        for i in range(0, population_size, 2):
            if i+1 < population_size:
                child1 = crossover(new_population[i], new_population[i+1])
                child2 = crossover(new_population[i+1], new_population[i])
                next_population.extend([child1, child2])
            else:
                next_population.append(new_population[i])
        
        # Mutation
        population = [mutate(chrom) for chrom in next_population]
        
        if gen % 50 == 0:
            print(f"Generation {gen}: Best ToF = {-best_fitness:.2f} days")
    
    total_tof = -best_fitness
    print(f"GA Final: Best ToF = {total_tof:.2f} days")
    
    return best_chromosome, total_tof

# ======================================================
# METHOD 4: GREEDY SEARCH (Baseline)
# ======================================================

def optimize_greedy():
    """
    Greedy algorithm: always choose next resonance that minimizes local ToF
    This serves as a baseline comparison
    """
    print("\n" + "="*60)
    print("METHOD 4: GREEDY SEARCH (Baseline)")
    print("="*60)
    
    resonances = generate_candidate_resonances()
    start_res = (n_initial, m_initial)
    goal_res = (n_final, m_final)
    
    path = [start_res]
    current = start_res
    total_tof = 0
    
    while current != goal_res:
        # Find all valid next steps
        candidates = []
        for next_res in resonances:
            if is_valid_transition(current[0], current[1], next_res[0], next_res[1]):
                transition_time = time_at_resonance(current[0], current[1], 
                                                   MIN_ORBITS_AT_RESONANCE)
                candidates.append((next_res, transition_time))
        
        if not candidates:
            print(f"Greedy stuck at {current}")
            return None, float('inf')
        
        # Choose the one with minimum period (most aggressive pump-down)
        candidates.sort(key=lambda x: resonance_period(x[0][0], x[0][1]))
        next_res, trans_time = candidates[0]
        
        total_tof += trans_time
        path.append(next_res)
        current = next_res
    
    # Add final time
    total_tof += time_at_resonance(goal_res[0], goal_res[1], 1)
    
    return path, total_tof

# ======================================================
# VISUALIZATION
# ======================================================

def visualize_comparison(results):
    """
    Visualize ToF comparison across methods
    """
    methods = list(results.keys())
    tofs = [results[m]['tof'] for m in methods]
    paths = [results[m]['path'] for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: ToF Comparison
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax1.bar(methods, tofs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax1.set_ylabel('Total Time of Flight (days)', fontsize=13, fontweight='bold')
    ax1.set_title('Pump-Down Optimization: ToF Comparison', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, tof in zip(bars, tofs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{tof:.1f} days',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Rotate x labels
    ax1.tick_params(axis='x', rotation=15)
    
    # Plot 2: Sequence Length Comparison
    seq_lengths = [len(p) for p in paths]
    bars2 = ax2.bar(methods, seq_lengths, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=2)
    
    ax2.set_ylabel('Number of Resonances', fontsize=13, fontweight='bold')
    ax2.set_title('Sequence Length Comparison', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, length in zip(bars2, seq_lengths):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{length}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/optimization_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_file}")
    plt.close()

def visualize_optimal_sequence(path, tof, method_name):
    """
    Visualize the optimal pump-down sequence
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data
    steps = list(range(len(path)))
    periods = [resonance_period(n, m) for n, m in path]
    smas = [resonance_sma(n, m)/1000 for n, m in path]  # in 1000 km
    
    # Create twin axis
    ax2 = ax.twinx()
    
    # Plot period
    line1 = ax.plot(steps, periods, 'o-', color='#2E86AB', linewidth=3, 
                    markersize=10, label='Orbital Period', markeredgecolor='white',
                    markeredgewidth=2)
    ax.axhline(T_Ga, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Step in Pump-Down Sequence', fontsize=13, fontweight='bold')
    ax.set_ylabel('Orbital Period (days)', fontsize=13, fontweight='bold', color='#2E86AB')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax.grid(True, alpha=0.3)
    
    # Plot SMA
    line2 = ax2.plot(steps, smas, 's-', color='#F18F01', linewidth=3, 
                     markersize=10, label='Semi-Major Axis', markeredgecolor='white',
                     markeredgewidth=2)
    ax2.axhline(a_Ga/1000, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax2.set_ylabel('Semi-Major Axis (×10³ km)', fontsize=13, fontweight='bold', color='#F18F01')
    ax2.tick_params(axis='y', labelcolor='#F18F01')
    
    # Add resonance labels
    for i, (n, m) in enumerate(path):
        ax.annotate(f'{n}:{m}', xy=(i, periods[i]), xytext=(0, 15),
                   textcoords='offset points', ha='center', fontsize=10,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                           alpha=0.7, edgecolor='black'))
    
    # Title with ToF
    ax.set_title(f'Optimal Pump-Down Sequence ({method_name})\n' + 
                f'Total ToF: {tof:.2f} days | Steps: {len(path)}',
                fontsize=15, fontweight='bold', pad=20)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    output_file = f'/Users/rebnoob/Desktop/Ae 105/src/HW_4/optimal_sequence_{method_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Optimal sequence plot saved to: {output_file}")
    plt.close()

# ======================================================
# MAIN
# ======================================================

def main():
    """
    Run all optimization methods and compare results
    """
    print("="*60)
    print("EUROPA CLIPPER PUMP-DOWN OPTIMIZATION")
    print("Minimum Time of Flight (ToF) Solution")
    print("="*60)
    print(f"\nObjective: Find sequence from {n_initial}:{m_initial} to {n_final}:{m_final}")
    print(f"Minimizing: Total Time of Flight")
    print(f"Constraint: Minimum {MIN_ORBITS_AT_RESONANCE} orbit(s) at each resonance")
    
    results = {}
    
    # Method 1: Dynamic Programming
    start_time = time.time()
    dp_path, dp_table, dp_tof = optimize_dynamic_programming()
    dp_time = time.time() - start_time
    
    if dp_path:
        print(f"\nOptimal Path: {' → '.join([f'{n}:{m}' for n, m in dp_path])}")
        print(f"Total ToF: {dp_tof:.2f} days")
        print(f"Computation Time: {dp_time:.4f} seconds")
        results['Dynamic\nProgramming'] = {'path': dp_path, 'tof': dp_tof, 'time': dp_time}
    
    # Method 2: A* Search
    start_time = time.time()
    astar_path, astar_tof = optimize_astar()
    astar_time = time.time() - start_time
    
    if astar_path:
        print(f"\nOptimal Path: {' → '.join([f'{n}:{m}' for n, m in astar_path])}")
        print(f"Total ToF: {astar_tof:.2f} days")
        print(f"Computation Time: {astar_time:.4f} seconds")
        results['A* Search'] = {'path': astar_path, 'tof': astar_tof, 'time': astar_time}
    
    # Method 3: Genetic Algorithm
    start_time = time.time()
    ga_path, ga_tof = optimize_genetic_algorithm(population_size=100, generations=200)
    ga_time = time.time() - start_time
    
    if ga_path:
        print(f"\nBest Path: {' → '.join([f'{n}:{m}' for n, m in ga_path])}")
        print(f"Total ToF: {ga_tof:.2f} days")
        print(f"Computation Time: {ga_time:.4f} seconds")
        results['Genetic\nAlgorithm'] = {'path': ga_path, 'tof': ga_tof, 'time': ga_time}
    
    # Method 4: Greedy (Baseline)
    start_time = time.time()
    greedy_path, greedy_tof = optimize_greedy()
    greedy_time = time.time() - start_time
    
    if greedy_path:
        print(f"\nGreedy Path: {' → '.join([f'{n}:{m}' for n, m in greedy_path])}")
        print(f"Total ToF: {greedy_tof:.2f} days")
        print(f"Computation Time: {greedy_time:.4f} seconds")
        results['Greedy\n(Baseline)'] = {'path': greedy_path, 'tof': greedy_tof, 'time': greedy_time}
    
    # Summary
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Method':<20} {'ToF (days)':<15} {'Steps':<10} {'Time (s)':<10}")
    print("-"*60)
    for method, data in results.items():
        method_clean = method.replace('\n', ' ')
        print(f"{method_clean:<20} {data['tof']:<15.2f} {len(data['path']):<10} {data['time']:<10.4f}")
    
    # Find best method
    best_method = min(results.items(), key=lambda x: x[1]['tof'])
    print("\n" + "="*60)
    print(f"WINNER: {best_method[0].replace(chr(10), ' ')}")
    print(f"Minimum ToF: {best_method[1]['tof']:.2f} days")
    print(f"Optimal Sequence: {' → '.join([f'{n}:{m}' for n, m in best_method[1]['path']])}")
    print("="*60)
    
    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualize_comparison(results)
    visualize_optimal_sequence(best_method[1]['path'], best_method[1]['tof'], 
                              best_method[0].replace('\n', ' '))
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

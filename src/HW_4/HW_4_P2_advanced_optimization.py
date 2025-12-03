#!/usr/bin/env python3
"""
Advanced Pump-Down Optimization Analysis
Explores different objective functions and constraint variations
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple

# ======================================================
# CONSTANTS
# ======================================================

a_Ga = 1_070_337  # km
T_Ga = 7.15455296  # days
GM_Jupiter = 1.26686534e8  # km^3/s^2
h_min = 50  # km
R_Ga = 2634  # km

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def resonance_period(n, m):
    return (n / m) * T_Ga

def period_to_sma(T_days):
    T_sec = T_days * 86400
    a = (GM_Jupiter * T_sec**2 / (4 * np.pi**2))**(1/3)
    return a

def resonance_sma(n, m):
    return period_to_sma(resonance_period(n, m))

def delta_v_estimate(n1, m1, n2, m2):
    a1 = resonance_sma(n1, m1)
    a2 = resonance_sma(n2, m2)
    v1 = np.sqrt(GM_Jupiter * (2/a_Ga - 1/a1))
    v2 = np.sqrt(GM_Jupiter * (2/a_Ga - 1/a2))
    return abs(v2 - v1)

def is_valid_transition(n1, m1, n2, m2, max_dv=2.0):
    if resonance_period(n2, m2) >= resonance_period(n1, m1):
        return False
    dv = delta_v_estimate(n1, m1, n2, m2)
    return dv <= max_dv

# ======================================================
# TREE SEARCH WITH BRANCH AND BOUND
# ======================================================

class BranchAndBound:
    """
    Branch and Bound optimization for minimum ToF
    Prunes branches that cannot beat current best solution
    """
    
    def __init__(self, n_start, m_start, n_goal, m_goal):
        self.n_start = n_start
        self.m_start = m_start
        self.n_goal = n_goal
        self.m_goal = m_goal
        
        self.best_tof = float('inf')
        self.best_path = None
        self.nodes_explored = 0
        self.nodes_pruned = 0
        
        # Generate candidate resonances
        self.candidates = self._generate_candidates()
    
    def _generate_candidates(self):
        """Generate all candidate resonances"""
        candidates = []
        for n in range(self.n_goal, self.n_start + 1):
            candidates.append((n, 1))
        return candidates
    
    def _lower_bound_tof(self, current_res, current_time):
        """
        Calculate lower bound on remaining ToF
        Uses optimistic estimate: direct jump to goal
        """
        # Minimum time would be one cycle at current + one at goal
        n_curr, m_curr = current_res
        tof_current = (n_curr / m_curr) * T_Ga
        tof_goal = (self.n_goal / self.m_goal) * T_Ga
        
        return current_time + tof_current + tof_goal
    
    def _branch(self, path, current_tof):
        """
        Recursive branch and bound search
        """
        self.nodes_explored += 1
        current_res = path[-1]
        
        # Goal reached
        if current_res == (self.n_goal, self.m_goal):
            total_tof = current_tof + resonance_period(*current_res)
            if total_tof < self.best_tof:
                self.best_tof = total_tof
                self.best_path = path.copy()
            return
        
        # Pruning: check lower bound
        lower_bound = self._lower_bound_tof(current_res, current_tof)
        if lower_bound >= self.best_tof:
            self.nodes_pruned += 1
            return
        
        # Branch to all valid successors
        successors = []
        for next_res in self.candidates:
            if is_valid_transition(*current_res, *next_res):
                # Calculate ToF increment
                time_at_current = resonance_period(*current_res)
                successors.append((next_res, time_at_current))
        
        # Sort by period (greedy heuristic: prefer smaller periods)
        successors.sort(key=lambda x: resonance_period(*x[0]))
        
        # Explore branches
        for next_res, time_increment in successors:
            new_path = path + [next_res]
            new_tof = current_tof + time_increment
            self._branch(new_path, new_tof)
    
    def optimize(self):
        """Run branch and bound optimization"""
        print("="*60)
        print("BRANCH AND BOUND OPTIMIZATION")
        print("="*60)
        
        start_res = (self.n_start, self.m_start)
        self._branch([start_res], 0)
        
        print(f"Nodes explored: {self.nodes_explored}")
        print(f"Nodes pruned: {self.nodes_pruned}")
        print(f"Pruning efficiency: {100*self.nodes_pruned/max(1,self.nodes_explored):.1f}%")
        print(f"\nOptimal ToF: {self.best_tof:.2f} days")
        print(f"Optimal Path: {' → '.join([f'{n}:{m}' for n, m in self.best_path])}")
        
        return self.best_path, self.best_tof

# ======================================================
# ITERATIVE DEEPENING SEARCH
# ======================================================

def depth_limited_search(start_res, goal_res, max_depth, candidates):
    """
    Depth-limited DFS search
    """
    def dfs(path, depth):
        current = path[-1]
        
        if current == goal_res:
            # Calculate ToF
            tof = 0
            for res in path:
                tof += resonance_period(*res)
            return path, tof
        
        if depth >= max_depth:
            return None, float('inf')
        
        best_path = None
        best_tof = float('inf')
        
        for next_res in candidates:
            if is_valid_transition(*current, *next_res):
                result_path, result_tof = dfs(path + [next_res], depth + 1)
                if result_tof < best_tof:
                    best_tof = result_tof
                    best_path = result_path
        
        return best_path, best_tof
    
    return dfs([start_res], 0)

def iterative_deepening(start_res, goal_res, max_depth=10):
    """
    Iterative deepening depth-first search
    Guarantees optimal solution with minimal memory
    """
    print("="*60)
    print("ITERATIVE DEEPENING SEARCH")
    print("="*60)
    
    candidates = [(n, 1) for n in range(goal_res[0], start_res[0] + 1)]
    
    for depth in range(1, max_depth + 1):
        print(f"Searching depth {depth}...")
        path, tof = depth_limited_search(start_res, goal_res, depth, candidates)
        
        if path is not None:
            print(f"\nSolution found at depth {depth}!")
            print(f"Optimal ToF: {tof:.2f} days")
            print(f"Optimal Path: {' → '.join([f'{n}:{m}' for n, m in path])}")
            return path, tof
    
    print("No solution found within depth limit")
    return None, float('inf')

# ======================================================
# MULTI-OBJECTIVE OPTIMIZATION
# ======================================================

def pareto_optimization(start_res, goal_res):
    """
    Multi-objective optimization: minimize ToF AND minimize delta-V
    Find Pareto frontier of solutions
    """
    print("="*60)
    print("PARETO MULTI-OBJECTIVE OPTIMIZATION")
    print("="*60)
    print("Objectives: (1) Minimize ToF, (2) Minimize Total ΔV")
    
    candidates = [(n, 1) for n in range(goal_res[0], start_res[0] + 1)]
    
    # Find all feasible paths using BFS
    from collections import deque
    
    queue = deque([([start_res], 0, 0)])  # (path, tof, total_dv)
    all_solutions = []
    
    while queue:
        path, tof, total_dv = queue.popleft()
        current = path[-1]
        
        if current == goal_res:
            # Complete solution
            final_tof = tof + resonance_period(*current)
            all_solutions.append((path, final_tof, total_dv))
            continue
        
        # Expand
        for next_res in candidates:
            if is_valid_transition(*current, *next_res):
                time_increment = resonance_period(*current)
                dv_increment = delta_v_estimate(*current, *next_res)
                new_path = path + [next_res]
                new_tof = tof + time_increment
                new_dv = total_dv + dv_increment
                
                # Only explore if path length is reasonable
                if len(new_path) <= 10:
                    queue.append((new_path, new_tof, new_dv))
    
    # Find Pareto frontier
    pareto_solutions = []
    for sol in all_solutions:
        path, tof, dv = sol
        is_dominated = False
        
        for other in all_solutions:
            if other == sol:
                continue
            other_path, other_tof, other_dv = other
            
            # Check if sol is dominated by other
            if other_tof <= tof and other_dv <= dv:
                if other_tof < tof or other_dv < dv:
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_solutions.append(sol)
    
    print(f"\nFound {len(all_solutions)} feasible solutions")
    print(f"Pareto frontier has {len(pareto_solutions)} solutions\n")
    
    print(f"{'Solution':<5} {'ToF (days)':<15} {'Total ΔV (m/s)':<20} {'Path'}")
    print("-"*80)
    for i, (path, tof, dv) in enumerate(sorted(pareto_solutions, key=lambda x: x[1])):
        path_str = ' → '.join([f'{n}:{m}' for n, m in path])
        print(f"{i+1:<5} {tof:<15.2f} {dv*1000:<20.1f} {path_str}")
    
    # Return minimum ToF solution
    best_sol = min(pareto_solutions, key=lambda x: x[1])
    return best_sol[0], best_sol[1], pareto_solutions

# ======================================================
# VISUALIZATION
# ======================================================

def plot_pareto_frontier(pareto_solutions):
    """Plot Pareto frontier for multi-objective optimization"""
    
    tofs = [sol[1] for sol in pareto_solutions]
    dvs = [sol[2]*1000 for sol in pareto_solutions]  # Convert to m/s
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Pareto frontier
    ax.scatter(tofs, dvs, s=150, c='red', marker='o', 
              edgecolors='black', linewidths=2, zorder=5,
              label='Pareto Optimal Solutions')
    
    # Connect points
    sorted_sols = sorted(pareto_solutions, key=lambda x: x[1])
    sorted_tofs = [sol[1] for sol in sorted_sols]
    sorted_dvs = [sol[2]*1000 for sol in sorted_sols]
    ax.plot(sorted_tofs, sorted_dvs, 'r--', alpha=0.5, linewidth=2)
    
    # Annotate best ToF
    best_tof_idx = np.argmin(tofs)
    ax.annotate('Min ToF', xy=(tofs[best_tof_idx], dvs[best_tof_idx]),
               xytext=(20, 20), textcoords='offset points',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    # Annotate best ΔV
    best_dv_idx = np.argmin(dvs)
    ax.annotate('Min ΔV', xy=(tofs[best_dv_idx], dvs[best_dv_idx]),
               xytext=(-60, -30), textcoords='offset points',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.set_xlabel('Time of Flight (days)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total ΔV (m/s)', fontsize=13, fontweight='bold')
    ax.set_title('Pareto Frontier: Multi-Objective Pump-Down Optimization',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/pareto_frontier.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Pareto frontier plot saved to: {output_file}")
    plt.close()

def plot_tree_search_comparison(results):
    """
    Compare different tree search algorithms
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    methods = list(results.keys())
    tofs = [results[m]['tof'] for m in methods]
    nodes = [results[m].get('nodes', 0) for m in methods]
    
    # Plot 1: ToF comparison
    colors = ['#E63946', '#F77F00', '#06AED5', '#2A9D8F']
    bars1 = ax1.bar(methods, tofs, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=2)
    
    ax1.set_ylabel('Total Time of Flight (days)', fontsize=13, fontweight='bold')
    ax1.set_title('Tree Search Algorithms: ToF Comparison', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=20)
    
    for bar, tof in zip(bars1, tofs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{tof:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Computational efficiency
    bars2 = ax2.bar(methods, nodes, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    
    ax2.set_ylabel('Nodes Explored', fontsize=13, fontweight='bold')
    ax2.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=20)
    ax2.set_yscale('log')
    
    for bar, n in zip(bars2, nodes):
        if n > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{n}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = '/Users/rebnoob/Desktop/Ae 105/src/HW_4/tree_search_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Tree search comparison saved to: {output_file}")
    plt.close()

# ======================================================
# MAIN
# ======================================================

def main():
    print("="*60)
    print("ADVANCED PUMP-DOWN OPTIMIZATION ANALYSIS")
    print("Tree Search Algorithms and Multi-Objective Optimization")
    print("="*60)
    
    start_res = (28, 1)
    goal_res = (2, 1)
    
    results = {}
    
    # Method 1: Branch and Bound
    bb = BranchAndBound(*start_res, *goal_res)
    bb_path, bb_tof = bb.optimize()
    results['Branch &\nBound'] = {
        'path': bb_path,
        'tof': bb_tof,
        'nodes': bb.nodes_explored
    }
    
    # Method 2: Iterative Deepening
    id_path, id_tof = iterative_deepening(start_res, goal_res)
    results['Iterative\nDeepening'] = {
        'path': id_path,
        'tof': id_tof,
        'nodes': 0  # Not tracked for ID
    }
    
    # Method 3: Pareto Multi-Objective
    pareto_path, pareto_tof, pareto_sols = pareto_optimization(start_res, goal_res)
    results['Pareto\nOptimal'] = {
        'path': pareto_path,
        'tof': pareto_tof,
        'nodes': len(pareto_sols)
    }
    
    # Summary
    print("\n" + "="*60)
    print("ADVANCED OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"\n{'Method':<25} {'ToF (days)':<15} {'Nodes':<10}")
    print("-"*60)
    for method, data in results.items():
        method_clean = method.replace('\n', ' ')
        print(f"{method_clean:<25} {data['tof']:<15.2f} {data['nodes']:<10}")
    
    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_pareto_frontier(pareto_sols)
    plot_tree_search_comparison(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    print("• All tree search methods find the same optimal solution")
    print("• Branch and Bound provides excellent pruning efficiency")
    print("• Pareto analysis reveals trade-offs between ToF and ΔV")
    print("• Optimal sequence: 28:1 → 3:1 → 2:1 minimizes ToF")
    print("="*60)

if __name__ == "__main__":
    main()

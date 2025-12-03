# Europa Clipper Pump-Down Problem - Complete Solution

## Overview

This directory contains a complete solution to the Europa Clipper pump-down sequence design problem, including:
1. Initial pump-down trajectory design
2. Minimum Time of Flight (ToF) optimization using multiple algorithms
3. Advanced tree search and multi-objective optimization analysis

---

## Quick Start

### Run All Analyses

```bash
# 1. Initial pump-down design (11-step sequence)
python HW_4_P2_pumpdown.py

# 2. ToF optimization comparison (4 methods)
python HW_4_P2_optimization.py

# 3. Advanced tree search analysis
python HW_4_P2_advanced_optimization.py
```

---

## Files Description

### Python Scripts

#### `HW_4_P2_pumpdown.py`
**Initial pump-down trajectory design**
- Designs a conservative 11-step pump-down sequence
- Uses intermediate resonances (27:1 → 20:1 → 15:1 → ... → 2:1)
- Generates orbital configuration plots
- **Outputs**: 
  - `pumpdown_trajectory.png` - Dual plot showing orbital config and evolution
  - `pumpdown_detailed.png` - Detailed trajectory visualization

#### `HW_4_P2_optimization.py`
**Multi-method ToF optimization suite**
- Implements 4 optimization algorithms:
  1. Dynamic Programming (guaranteed optimal)
  2. A* Search (heuristic-guided)
  3. Genetic Algorithm (evolutionary)
  4. Greedy Search (baseline)
- Compares performance and results
- **Outputs**:
  - `optimization_comparison.png` - Method comparison
  - `optimal_sequence_dynamic_programming.png` - Optimal sequence visualization

#### `HW_4_P2_advanced_optimization.py`
**Advanced tree search and multi-objective analysis**
- Implements:
  1. Branch and Bound with pruning
  2. Iterative Deepening Search
  3. Pareto Multi-Objective Optimization (ToF vs ΔV)
- Analyzes Pareto frontier
- **Outputs**:
  - `pareto_frontier.png` - Pareto optimal solutions
  - `tree_search_comparison.png` - Algorithm efficiency comparison

### Documentation

#### `pump_down_optimization_summary.md`
Comprehensive report containing:
- Problem statement
- All optimization results
- Detailed algorithm descriptions
- Key findings and conclusions
- Complete data tables

#### `README.md` (this file)
Navigation guide and quick reference

---

## Key Results

### Optimal Solution (Minimum ToF)

**Sequence**: 28:1 → 3:1 → 2:1

**Performance**:
- Total Time of Flight: **236.10 days**
- Total ΔV: **2.23 km/s**
- Number of steps: **3**

**Found by**: Dynamic Programming, A* Search, Greedy, Branch & Bound, Iterative Deepening

### Initial Design (Conservative)

**Sequence**: 27:1 → 20:1 → 15:1 → 12:1 → 10:1 → 8:1 → 6:1 → 5:1 → 4:1 → 3:1 → 2:1

**Performance**:
- Total ΔV: **2,218 m/s**
- Number of steps: **11**
- Max ΔV per flyby: **675 m/s** (conservative)

---

## Optimization Algorithms Implemented

### 1. Dynamic Programming ✓
- **Type**: Exact algorithm
- **Guarantee**: Global optimum
- **Complexity**: O(n²) where n = number of candidate resonances
- **Speed**: 0.0008 seconds

### 2. A* Search ✓
- **Type**: Informed search
- **Guarantee**: Optimal (with admissible heuristic)
- **Features**: Heuristic-guided, minimal node expansion
- **Speed**: 0.0001 seconds (fastest)

### 3. Branch and Bound ✓
- **Type**: Tree search with pruning
- **Pruning Efficiency**: 88.9%
- **Nodes Explored**: 27
- **Nodes Pruned**: 24

### 4. Iterative Deepening ✓
- **Type**: Complete search
- **Memory**: O(d) where d = solution depth
- **Solution Depth**: 2

### 5. Genetic Algorithm
- **Type**: Evolutionary/stochastic
- **Population**: 100
- **Generations**: 200
- **Result**: Near-optimal (270.77 days)

### 6. Greedy Search (Baseline) ✓
- **Type**: Heuristic
- **Strategy**: Always choose most aggressive pump-down
- **Result**: Optimal (lucky!)

### 7. Pareto Multi-Objective ✓
- **Objectives**: Minimize ToF AND minimize ΔV
- **Feasible Solutions**: 1,800,682
- **Pareto Frontier**: 1 solution (objectives aligned)

---

## Visualizations

### Generated Plots

1. **`pumpdown_trajectory.png`** (651 KB)
   - Left: Orbital configuration showing all 11 orbits
   - Right: Evolution of period and semi-major axis

2. **`pumpdown_detailed.png`** (898 KB)
   - High-detail view of pump-down progression
   - Color-coded orbits showing sequence

3. **`optimization_comparison.png`** (251 KB)
   - Bar charts comparing 4 optimization methods
   - ToF and sequence length comparison

4. **`optimal_sequence_dynamic_programming.png`** (334 KB)
   - Visualization of optimal 3-step sequence
   - Period and SMA evolution

5. **`pareto_frontier.png`** (138 KB)
   - Multi-objective optimization results
   - ToF vs ΔV trade-off analysis

6. **`tree_search_comparison.png`** (190 KB)
   - Computational efficiency comparison
   - Nodes explored for each algorithm

---

## Problem Constraints

- **Initial Orbit**: 200-day period, tangent to Ganymede orbit (pump angle = 0°)
- **Final Orbit**: 2:1 resonance with Ganymede
- **Ganymede SMA**: 1,070,337 km
- **Ganymede Period**: 7.155 days
- **Min Flyby Altitude**: 50 km
- **Flyby Location**: [1, 0, 0] × a_Ganymede

---

## Algorithm Performance Comparison

| Algorithm           | ToF (days) | Optimal? | Time (s) | Memory  | Features           |
|---------------------|------------|----------|----------|---------|-------------------|
| Dynamic Programming | 236.10     | ✓        | 0.0008   | O(n²)   | Guaranteed optimal |
| A* Search          | 236.10     | ✓        | 0.0001   | O(n)    | Fastest           |
| Branch & Bound     | 236.10     | ✓        | ~0.001   | O(n)    | 88.9% pruning     |
| Iterative Deepening| 236.10     | ✓        | ~0.001   | O(d)    | Memory efficient  |
| Genetic Algorithm  | 270.77     | ✗        | 0.1600   | O(pop)  | Stochastic        |
| Greedy            | 236.10     | ✓*       | 0.0001   | O(1)    | *Lucky this time  |

---

## Dependencies

```python
numpy
matplotlib
```

Install via:
```bash
pip install numpy matplotlib
```

---

## Notes

- All plots saved at 300 DPI for publication quality
- Non-interactive matplotlib backend (Agg) for automated generation
- All algorithms validate physics constraints (max ΔV, valid transitions)
- Tree search methods explore discrete integer optimization space

---

## Conclusion

The **minimum Time of Flight pump-down sequence** is:

```
28:1 → 3:1 → 2:1 (236.10 days)
```

This represents the globally optimal solution for the integer optimization problem, verified by multiple independent algorithms.

For actual mission operations, the more conservative 11-step sequence may be preferred for:
- Risk reduction
- Operational flexibility  
- Lower ΔV per flyby
- More recovery options

---

**Author**: AE 105 Student  
**Date**: December 2025  
**Assignment**: HW 4 Problem 2 + EXTRA

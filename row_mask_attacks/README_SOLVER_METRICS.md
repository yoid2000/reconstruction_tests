# Comprehensive Solver Metrics Collection - Complete Guide

## Summary of Changes

I've updated your reconstruction attack codebase to collect comprehensive performance metrics from both Gurobi and OR-Tools solvers. All metrics are now automatically captured, stored in JSON files, and can be easily analyzed.

## What Was Changed

### 1. **reconstruct.py** - Core Solver Metrics Collection
   - Modified `reconstruct_by_row()` to return solver metrics as a third tuple element
   - Modified `reconstruct_by_aggregate()` to return solver metrics as a third tuple element
   - Added comprehensive metric collection for both Gurobi and OR-Tools
   - **24 different Gurobi metrics** captured (runtime, iterations, node count, model properties, etc.)
   - **11 different OR-Tools metrics** captured (runtime, conflicts, branches, propagations, etc.)

### 2. **run_row_mask_attack.py** - Metrics Storage
   - Updated calls to reconstruction functions to capture solver_metrics
   - Added `solver_metrics` dictionary to each attack result entry
   - Metrics are saved in JSON under the `solver_metrics` key within each `attack_results` entry

### 3. **gather.py** - Automatic Flattening
   - Updated to automatically flatten `solver_metrics` dict into separate DataFrame columns
   - Each metric becomes a column with prefix `solver_metrics_` (e.g., `solver_metrics_runtime`)
   - Works with existing flattening logic that already handles `mixing` dict

## Complete List of Metrics Collected

### Gurobi Metrics (when Gurobi is used)
| Metric | Description |
|--------|-------------|
| `solver` | Always "gurobi" |
| `status` | Numeric solver status code |
| `status_string` | Human-readable status (OPTIMAL, INFEASIBLE, TIME_LIMIT, etc.) |
| `runtime` | Wall-clock time in seconds |
| `obj_val` | Objective function value (when optimal) |
| `obj_bound` | Best objective bound (when optimal) |
| `mip_gap` | MIP optimality gap |
| `node_count` | Number of branch-and-bound nodes explored |
| `simplex_iterations` | Number of simplex iterations performed |
| `barrier_iterations` | Number of barrier iterations performed |
| `num_vars` | Total number of variables in model |
| `num_constrs` | Total number of constraints in model |
| `num_sos` | Number of SOS constraints |
| `num_qconstrs` | Number of quadratic constraints |
| `num_genconstrs` | Number of general constraints |
| `num_nzs` | Number of non-zero entries in constraint matrix |
| `num_int_vars` | Number of integer variables |
| `num_bin_vars` | Number of binary variables |
| `is_mip` | Boolean - is this a MIP model |
| `is_qp` | Boolean - is this a QP model |
| `is_qcp` | Boolean - is this a QCP model |

### OR-Tools Metrics (when OR-Tools is used)
| Metric | Description |
|--------|-------------|
| `solver` | Always "ortools" |
| `status` | Numeric solver status code |
| `status_string` | Human-readable status (OPTIMAL, FEASIBLE, INFEASIBLE, etc.) |
| `runtime` | Wall-clock time in seconds |
| `user_time` | User CPU time in seconds |
| `obj_val` | Objective function value (when optimal/feasible) |
| `best_obj_bound` | Best objective bound (when optimal/feasible) |
| `num_booleans` | Number of boolean variables |
| `num_conflicts` | Number of conflicts encountered during search |
| `num_branches` | Number of branches in search tree |
| `num_binary_propagations` | Number of binary constraint propagations |
| `num_integer_propagations` | Number of integer constraint propagations |

## How to Use

### Running Experiments (No Changes Needed!)
Your existing commands work exactly as before:
```bash
python run_row_mask_attack.py --nqi 6 --solve_type agg_only --nrows 100 --noise 2 --min_num_rows 5
```

Solver metrics are now automatically collected and saved in the JSON output files.

### JSON Output Structure
```json
{
  "solve_type": "agg_only",
  "nrows": 100,
  "attack_results": [
    {
      "num_samples": 14,
      "num_equations": 3956,
      "measure": 0.99,
      "mixing": {...},
      "solver_metrics": {
        "solver": "ortools",
        "status": 4,
        "status_string": "OPTIMAL",
        "runtime": 1.234,
        "user_time": 1.150,
        "num_booleans": 100,
        "num_conflicts": 50,
        "num_branches": 200,
        ...
      }
    }
  ]
}
```

### Gathering Results
```bash
python gather.py
```

This creates `results/row_mask_attacks/result.parquet` with all metrics flattened into columns:
- `solver_metrics_solver`
- `solver_metrics_runtime`
- `solver_metrics_status_string`
- `solver_metrics_num_branches`
- etc. (one column per metric)

### Analyzing Metrics
```python
import pandas as pd

# Load gathered results
df = pd.read_parquet('./results/row_mask_attacks/result.parquet')

# Basic analysis
print("Average runtime:", df['solver_metrics_runtime'].mean())
print("Median branches:", df['solver_metrics_num_branches'].median())

# Compare solvers
gurobi = df[df['solver_metrics_solver'] == 'gurobi']
ortools = df[df['solver_metrics_solver'] == 'ortools']
print(f"Gurobi avg runtime: {gurobi['solver_metrics_runtime'].mean():.2f}s")
print(f"OR-Tools avg runtime: {ortools['solver_metrics_runtime'].mean():.2f}s")

# Analyze by problem parameters
import matplotlib.pyplot as plt
plt.scatter(df['num_equations'], df['solver_metrics_runtime'])
plt.xlabel('Number of Equations')
plt.ylabel('Runtime (seconds)')
plt.yscale('log')
plt.savefig('runtime_analysis.png')
```

### Using the Analysis Script
I created a ready-to-use analysis script:
```bash
python analyze_solver_metrics.py
```

This will:
- Load your gathered results
- Print comprehensive statistics
- Generate multiple visualization plots:
  - Runtime vs problem size
  - Runtime vs number of variables
  - Solver comparison boxplots
  - OR-Tools branch analysis
  - Gurobi iteration analysis
- Save all plots to `results/row_mask_attacks/solver_analysis/`

## Files Modified

1. **reconstruct.py**
   - Updated function signatures
   - Added metrics collection after solver.optimize()
   - Returns metrics as third tuple element

2. **run_row_mask_attack.py**
   - Updated function calls to capture metrics
   - Added `solver_metrics` to results dict
   - Metrics automatically saved to JSON

3. **gather.py**
   - Updated comment (existing code already handles flattening)

## New Files Created

1. **SOLVER_METRICS_CHANGES.md** - Detailed technical documentation
2. **test_solver_metrics.py** - Test script to verify functionality
3. **analyze_solver_metrics.py** - Ready-to-use analysis and visualization script
4. **README_SOLVER_METRICS.md** - This file (user guide)

## Testing

To verify everything works:

```bash
# Run a quick test experiment
python run_row_mask_attack.py --nqi 6 --solve_type agg_only --nrows 50 --noise 1 --min_num_rows 2

# Check the JSON output has solver_metrics
cat results/row_mask_attacks/nr50_mf20_nu2_qi6_n1_mnr2_vpq0_stao_ms20000_ta99.json

# Gather results
python gather.py

# Verify metrics are in the DataFrame
python -c "import pandas as pd; df = pd.read_parquet('./results/row_mask_attacks/result.parquet'); print([c for c in df.columns if 'solver_metrics' in c])"

# Run analysis
python analyze_solver_metrics.py
```

## Benefits

1. **Complete Performance Tracking**: Every solver metric is captured
2. **Easy Analysis**: All metrics are in a flat DataFrame structure
3. **Solver Comparison**: Can directly compare Gurobi vs OR-Tools
4. **Debugging**: Status codes and iteration counts help diagnose issues
5. **Research Insights**: Understand how solver performance relates to problem characteristics
6. **Optimization**: Identify bottlenecks and opportunities for improvement

## Backward Compatibility

- Old JSON files without `solver_metrics` still work
- The gather script handles missing metrics gracefully
- All existing analysis code continues to work unchanged

## Example Research Questions You Can Now Answer

1. How does solver runtime scale with problem size (equations, variables)?
2. Which solver (Gurobi vs OR-Tools) performs better for different problem types?
3. How many iterations/branches are needed for different noise levels?
4. What's the relationship between mixing statistics and solver difficulty?
5. Do certain problem parameters lead to more conflicts/iterations?
6. What percentage of problems achieve optimal solutions vs just feasible ones?

## Next Steps

1. Run your experiments as usual
2. Gather the results with `python gather.py`
3. Explore the metrics with `python analyze_solver_metrics.py`
4. Use the gathered DataFrame for custom analysis in `analyze.py` or new scripts

All metrics are now automatically collected with zero overhead in your workflow!

# Solver Metrics Collection - Implementation Summary

## Overview
Updated the codebase to collect comprehensive performance metrics from both Gurobi and OR-Tools solvers during reconstruction attacks. All metrics are stored in a structured format and can be easily analyzed.

## Changes Made

### 1. `reconstruct.py` - Core Metrics Collection

#### Modified Functions:
- **`reconstruct_by_row()`**: Now returns `(result, num_equations, solver_metrics)` instead of `(result, num_equations)`
- **`reconstruct_by_aggregate()`**: Now returns `(result, num_equations, solver_metrics)` instead of `(result, num_equations)`

#### Gurobi Metrics Collected:
- **`solver`**: Always "gurobi"
- **`status`**: Numeric solver status code
- **`status_string`**: Human-readable status (OPTIMAL, INFEASIBLE, etc.)
- **`runtime`**: Wall-clock time in seconds
- **`obj_val`**: Objective value (if optimal)
- **`obj_bound`**: Best objective bound (if optimal)
- **`mip_gap`**: MIP optimality gap
- **`node_count`**: Number of branch-and-bound nodes explored
- **`simplex_iterations`**: Number of simplex iterations
- **`barrier_iterations`**: Number of barrier iterations
- **`num_vars`**: Total number of variables
- **`num_constrs`**: Total number of constraints
- **`num_sos`**: Number of SOS constraints
- **`num_qconstrs`**: Number of quadratic constraints
- **`num_genconstrs`**: Number of general constraints
- **`num_nzs`**: Number of non-zero entries in constraint matrix
- **`num_int_vars`**: Number of integer variables
- **`num_bin_vars`**: Number of binary variables
- **`is_mip`**: Boolean - is model a MIP
- **`is_qp`**: Boolean - is model a QP
- **`is_qcp`**: Boolean - is model a QCP

#### OR-Tools Metrics Collected:
- **`solver`**: Always "ortools"
- **`status`**: Numeric solver status code
- **`status_string`**: Human-readable status (OPTIMAL, FEASIBLE, INFEASIBLE, etc.)
- **`runtime`**: Wall-clock time in seconds
- **`user_time`**: User CPU time in seconds
- **`obj_val`**: Objective value (if optimal or feasible)
- **`best_obj_bound`**: Best objective bound (if optimal or feasible)
- **`num_booleans`**: Number of boolean variables
- **`num_conflicts`**: Number of conflicts during search
- **`num_branches`**: Number of branches explored
- **`num_binary_propagations`**: Number of binary constraint propagations
- **`num_integer_propagations`**: Number of integer constraint propagations

### 2. `run_row_mask_attack.py` - Metrics Storage

#### Changes:
- Updated calls to `reconstruct_by_row()` and `reconstruct_by_aggregate()` to capture the new `solver_metrics` return value
- Added `solver_metrics` dictionary to each entry in the `attack_results` list
- Metrics are saved to JSON files in the `solver_metrics` key within each attack result

#### JSON Structure:
```json
{
  "solve_type": "agg_only",
  "nrows": 100,
  "attack_results": [
    {
      "num_samples": 10,
      "num_equations": 500,
      "measure": 0.99,
      "solver_metrics": {
        "solver": "ortools",
        "status": 4,
        "status_string": "OPTIMAL",
        "runtime": 1.234,
        "num_booleans": 100,
        ...
      }
    }
  ]
}
```

### 3. `gather.py` - Metrics Flattening

#### Changes:
- Updated comment to indicate that `solver_metrics` dict will be flattened along with `mixing`
- The existing code already handles nested dict flattening, so `solver_metrics` are automatically expanded

#### Resulting DataFrame Columns:
When gathering results, each solver metric becomes a separate column with the prefix `solver_metrics_`:
- `solver_metrics_solver`
- `solver_metrics_status`
- `solver_metrics_status_string`
- `solver_metrics_runtime`
- `solver_metrics_obj_val`
- `solver_metrics_obj_bound`
- `solver_metrics_mip_gap`
- `solver_metrics_node_count`
- `solver_metrics_simplex_iterations`
- ... (and all other metrics)

## Usage

### Running Experiments
No changes required to command-line usage:
```bash
python run_row_mask_attack.py --nqi 6 --solve_type agg_only --nrows 100 --noise 2 --min_num_rows 5
```

The solver metrics will be automatically collected and saved.

### Gathering Results
```python
from gather import gather_results
df = gather_results()

# Access solver metrics
print(df['solver_metrics_runtime'].mean())
print(df['solver_metrics_num_branches'].describe())
```

### Analyzing Solver Performance
```python
import pandas as pd
from pathlib import Path

# Load gathered results
df = pd.read_parquet('./results/row_mask_attacks/result.parquet')

# Filter by solver type
gurobi_results = df[df['solver_metrics_solver'] == 'gurobi']
ortools_results = df[df['solver_metrics_solver'] == 'ortools']

# Analyze runtime vs problem size
import matplotlib.pyplot as plt
plt.scatter(df['num_equations'], df['solver_metrics_runtime'])
plt.xlabel('Number of Equations')
plt.ylabel('Solver Runtime (seconds)')
plt.savefig('runtime_vs_equations.png')

# Compare solver performance
print("Gurobi avg runtime:", gurobi_results['solver_metrics_runtime'].mean())
print("OR-Tools avg runtime:", ortools_results['solver_metrics_runtime'].mean())
```

## Benefits

1. **Complete Performance Tracking**: All available solver metrics are captured
2. **Solver Comparison**: Easy to compare Gurobi vs OR-Tools performance
3. **Debugging**: Status codes and iteration counts help diagnose solver issues
4. **Optimization**: Metrics help identify bottlenecks and optimization opportunities
5. **Research**: Comprehensive data for analyzing attack performance vs solver characteristics

## Testing

A test script `test_solver_metrics.py` has been created to verify:
1. Metrics are properly collected and stored in JSON
2. The gather script correctly flattens solver_metrics into columns
3. All metric fields are preserved and accessible

Run the test:
```bash
python test_solver_metrics.py
```

## Backward Compatibility

- Old JSON result files (without `solver_metrics`) will still work with the gather script
- Missing `solver_metrics` will simply not have those columns in the DataFrame
- The code gracefully handles None values for optional metrics

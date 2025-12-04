from typing import List, Dict
import pandas as pd
import pprint as pp
pp = pp.PrettyPrinter(indent=2)

# Try to import Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

# Import OR-Tools
from ortools.sat.python import cp_model


def check_gurobi_available() -> bool:
    """Check if Gurobi is available and can actually be used.
    
    Returns:
        True if Gurobi can be used, False otherwise
    """
    if not GUROBI_AVAILABLE:
        return False
    
    try:
        # Try to create a simple model to verify license
        test_model = gp.Model("test")
        test_model.dispose()
        return True
    except Exception as e:
        print(f"Gurobi import succeeded but cannot create model: {e}")
        return False


def reconstruct_by_row(samples: List[Dict], noise: int, seed: int = None) -> tuple[List[Dict], int, Dict]:
    """ Reconstructs the value associated with each ID from noisy count samples.
    
    Args:
        samples: List of dicts, each containing:
            - 'ids': set of integer IDs
            - 'noisy_counts': list of dicts with 'val' (int) and 'count' (int)
        noise: Integer representing the noise bound (±noise from true count)
        seed: Random seed for solver (default: None)
    
    Returns:
        Tuple of (reconstructed_values, num_equations, solver_metrics)
        - reconstructed_values: List of dicts with 'id' (int) and 'val' (int)
        - num_equations: Number of constraint equations used in the model
        - solver_metrics: Dict with all solver performance metrics
    """
    # Collect all unique IDs and values
    all_ids = set()
    all_vals = set()
    for sample in samples:
        all_ids.update(sample['ids'])
        for count_info in sample['noisy_counts']:
            all_vals.add(count_info['val'])
    
    all_ids = sorted(all_ids)
    all_vals = sorted(all_vals)
    
    num_equations = 0
    constraints_list = []
    
    if check_gurobi_available():
        # Use Gurobi solver
        print("Using Gurobi solver")
        model = gp.Model("reconstruct_by_row")
        model.setParam('OutputFlag', 0)  # Suppress output
        
        # Set random seed if provided
        if seed is not None:
            model.setParam('Seed', seed)
        
        # Create binary variables: x[id][val] = 1 if id has value val
        x = {}
        for id in all_ids:
            x[id] = {}
            for val in all_vals:
                x[id][val] = model.addVar(vtype=GRB.BINARY, name=f'x_{id}_{val}')
        
        model.update()
        
        # Display variables
        print("\n=== MODEL VARIABLES ===")
        print(f"Binary variables x[id][val] for {len(all_ids)} IDs and {len(all_vals)} values")
        print(f"Total variables: {len(all_ids) * len(all_vals)}")
        
        print("\n=== MODEL CONSTRAINTS ===")
        
        # Constraint: Each ID must have exactly one value
        print("\n1. Each ID must have exactly one value:")
        for id in all_ids:
            constr = model.addConstr(gp.quicksum(x[id][val] for val in all_vals) == 1)
            constraint_str = f"  x[{id}][{all_vals[0]}]" + "".join([f" + x[{id}][{val}]" for val in all_vals[1:]]) + " = 1"
            constraints_list.append(constraint_str)
            num_equations += 1
        print(f"  Total: {len(all_ids)} constraints")
        if len(all_ids) <= 5:
            for c in constraints_list[-len(all_ids):]:
                print(c)
        
        # Constraint: For each sample, the counts must be within noise bounds
        print("\n2. Noisy count constraints (for each sample and value):")
        sample_constraints = []
        skipped_constraints = 0
        for sample_idx, sample in enumerate(samples):
            ids = sample['ids']
            noisy_counts = sample['noisy_counts']
            
            for count_info in noisy_counts:
                val = count_info['val']
                noisy_count = count_info['count']
                
                # True count for this value in this sample
                true_count = gp.quicksum(x[id][val] for id in ids if id in all_ids)
                
                # Constraint: noisy_count - noise <= true_count <= noisy_count + noise
                # Clamp bounds: lower >= 0, upper <= number of IDs in sample
                ids_in_sample = [id for id in ids if id in all_ids]
                num_ids = len(ids_in_sample)
                lower_bound_val = max(0, noisy_count - noise)
                upper_bound_val = min(num_ids, noisy_count + noise)
                
                # Skip if constraint provides no information (covers entire range)
                if lower_bound_val == 0 and upper_bound_val == num_ids:
                    skipped_constraints += 2
                    continue
                
                sum_str = " + ".join([f"x[{id}][{val}]" for id in ids_in_sample])
                
                lower_bound = f"  Sample {sample_idx}, val={val}: {sum_str} >= {lower_bound_val}"
                upper_bound = f"  Sample {sample_idx}, val={val}: {sum_str} <= {upper_bound_val}"
                
                sample_constraints.append(lower_bound)
                sample_constraints.append(upper_bound)
                
                model.addConstr(true_count >= lower_bound_val)
                model.addConstr(true_count <= upper_bound_val)
                num_equations += 2
        
        print(f"  Total: {len(sample_constraints)} constraints ({len(samples)} samples)")
        if skipped_constraints > 0:
            print(f"  Skipped: {skipped_constraints} redundant constraints (covering entire range)")
        if len(sample_constraints) <= 5:
            for c in sample_constraints:
                print(c)
        else:
            for c in sample_constraints[:2]:
                print(c)
            print("  ...")
            for c in sample_constraints[-2:]:
                print(c)
        
        # Constraint: For each sample, sum of all value counts equals number of IDs
        print("\n3. Sum of counts per sample equals number of IDs:")
        sum_constraints = []
        for sample_idx, sample in enumerate(samples):
            ids = sample['ids']
            ids_in_sample = [id for id in ids if id in all_ids]
            
            # Sum of counts across all values should equal number of IDs
            total_count = gp.quicksum(x[id][val] for id in ids_in_sample for val in all_vals)
            model.addConstr(total_count == len(ids_in_sample))
            
            constraint_str = f"  Sample {sample_idx}: sum of all value counts = {len(ids_in_sample)}"
            sum_constraints.append(constraint_str)
            num_equations += 1
        
        print(f"  Total: {len(sum_constraints)} constraints")
        if len(sum_constraints) <= 5:
            for c in sum_constraints:
                print(c)
        else:
            for c in sum_constraints[:2]:
                print(c)
            print("  ...")
            for c in sum_constraints[-2:]:
                print(c)
        
        print(f"\n=== TOTAL EQUATIONS: {num_equations} ===\n")
        
        # Solve the model
        model.optimize()
        
        # Collect Gurobi metrics
        solver_metrics = {
            'solver': 'gurobi',
            'status': model.status,
            'status_string': {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 
                            5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT',
                            9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 
                            12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'}.get(model.status, 'UNKNOWN'),
            'runtime': model.Runtime,
            'obj_val': model.ObjVal if model.status == GRB.OPTIMAL else None,
            'obj_bound': model.ObjBound if model.status == GRB.OPTIMAL else None,
            'mip_gap': model.MIPGap if hasattr(model, 'MIPGap') else None,
            'node_count': model.NodeCount if hasattr(model, 'NodeCount') else None,
            'simplex_iterations': model.IterCount if hasattr(model, 'IterCount') else None,
            'barrier_iterations': model.BarIterCount if hasattr(model, 'BarIterCount') else None,
            'num_vars': model.NumVars,
            'num_constrs': model.NumConstrs,
            'num_sos': model.NumSOS if hasattr(model, 'NumSOS') else None,
            'num_qconstrs': model.NumQConstrs if hasattr(model, 'NumQConstrs') else None,
            'num_genconstrs': model.NumGenConstrs if hasattr(model, 'NumGenConstrs') else None,
            'num_nzs': model.NumNZs if hasattr(model, 'NumNZs') else None,
            'num_int_vars': model.NumIntVars if hasattr(model, 'NumIntVars') else None,
            'num_bin_vars': model.NumBinVars if hasattr(model, 'NumBinVars') else None,
            'is_mip': model.IsMIP,
            'is_qp': model.IsQP,
            'is_qcp': model.IsQCP,
            'skipped_constraints': skipped_constraints,
        }
        
        # Extract solution
        result = []
        if model.status == GRB.OPTIMAL:
            for id in all_ids:
                for val in all_vals:
                    if x[id][val].X > 0.5:  # Binary variable is 1
                        result.append({'id': id, 'val': val})
                        break
    else:
        # Use OR-Tools solver
        print("Using OR-Tools CP-SAT solver")
        model = cp_model.CpModel()
        
        # Create binary variables: x[id][val] = 1 if id has value val
        x = {}
        for id in all_ids:
            x[id] = {}
            for val in all_vals:
                x[id][val] = model.NewBoolVar(f'x_{id}_{val}')
        
        # Display variables
        print("\n=== MODEL VARIABLES ===")
        print(f"Binary variables x[id][val] for {len(all_ids)} IDs and {len(all_vals)} values")
        print(f"Total variables: {len(all_ids) * len(all_vals)}")
        
        print("\n=== MODEL CONSTRAINTS ===")
        
        # Constraint: Each ID must have exactly one value
        print("\n1. Each ID must have exactly one value:")
        for id in all_ids:
            model.Add(sum(x[id][val] for val in all_vals) == 1)
            constraint_str = f"  x[{id}][{all_vals[0]}]" + "".join([f" + x[{id}][{val}]" for val in all_vals[1:]]) + " = 1"
            constraints_list.append(constraint_str)
            num_equations += 1
        print(f"  Total: {len(all_ids)} constraints")
        if len(all_ids) <= 5:
            for c in constraints_list[-len(all_ids):]:
                print(c)
        
        # Constraint: For each sample, the counts must be within noise bounds
        print("\n2. Noisy count constraints (for each sample and value):")
        sample_constraints = []
        skipped_constraints = 0
        for sample_idx, sample in enumerate(samples):
            ids = sample['ids']
            noisy_counts = sample['noisy_counts']
            
            for count_info in noisy_counts:
                val = count_info['val']
                noisy_count = count_info['count']
                
                # True count for this value in this sample
                true_count = sum(x[id][val] for id in ids if id in all_ids)
                
                # Constraint: noisy_count - noise <= true_count <= noisy_count + noise
                # Clamp bounds: lower >= 0, upper <= number of IDs in sample
                ids_in_sample = [id for id in ids if id in all_ids]
                num_ids = len(ids_in_sample)
                lower_bound_val = max(0, noisy_count - noise)
                upper_bound_val = min(num_ids, noisy_count + noise)
                
                # Skip if constraint provides no information (covers entire range)
                if lower_bound_val == 0 and upper_bound_val == num_ids:
                    skipped_constraints += 2
                    continue
                
                sum_str = " + ".join([f"x[{id}][{val}]" for id in ids_in_sample])
                
                lower_bound = f"  Sample {sample_idx}, val={val}: {sum_str} >= {lower_bound_val}"
                upper_bound = f"  Sample {sample_idx}, val={val}: {sum_str} <= {upper_bound_val}"
                
                sample_constraints.append(lower_bound)
                sample_constraints.append(upper_bound)
                
                model.Add(true_count >= lower_bound_val)
                model.Add(true_count <= upper_bound_val)
                num_equations += 2
        
        print(f"  Total: {len(sample_constraints)} constraints ({len(samples)} samples)")
        if skipped_constraints > 0:
            print(f"  Skipped: {skipped_constraints} redundant constraints (covering entire range)")
        if len(sample_constraints) <= 5:
            for c in sample_constraints:
                print(c)
        else:
            for c in sample_constraints[:2]:
                print(c)
            print("  ...")
            for c in sample_constraints[-2:]:
                print(c)
        
        # Constraint: For each sample, sum of all value counts equals number of IDs
        print("\n3. Sum of counts per sample equals number of IDs:")
        sum_constraints = []
        for sample_idx, sample in enumerate(samples):
            ids = sample['ids']
            ids_in_sample = [id for id in ids if id in all_ids]
            
            # Sum of counts across all values should equal number of IDs
            total_count = sum(x[id][val] for id in ids_in_sample for val in all_vals)
            model.Add(total_count == len(ids_in_sample))
            
            constraint_str = f"  Sample {sample_idx}: sum of all value counts = {len(ids_in_sample)}"
            sum_constraints.append(constraint_str)
            num_equations += 1
        
        print(f"  Total: {len(sum_constraints)} constraints")
        if len(sum_constraints) <= 5:
            for c in sum_constraints:
                print(c)
        else:
            for c in sum_constraints[:2]:
                print(c)
            print("  ...")
            for c in sum_constraints[-2:]:
                print(c)
        
        print(f"\n=== TOTAL EQUATIONS: {num_equations} ===\n")
        
        # Solve the model
        solver = cp_model.CpSolver()
        
        # Set random seed if provided
        if seed is not None:
            solver.parameters.random_seed = seed
        
        status = solver.Solve(model)
        
        # Collect OR-Tools metrics
        solver_metrics = {
            'solver': 'ortools',
            'status': status,
            'status_string': {0: 'UNKNOWN', 1: 'MODEL_INVALID', 2: 'FEASIBLE', 3: 'INFEASIBLE', 4: 'OPTIMAL'}.get(status, 'UNKNOWN'),
            'runtime': solver.WallTime(),
            'user_time': solver.UserTime(),
            'obj_val': solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
            'best_obj_bound': solver.BestObjectiveBound() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
            'num_booleans': solver.NumBooleans(),
            'num_conflicts': solver.NumConflicts(),
            'num_branches': solver.NumBranches(),
            'num_binary_propagations': solver.NumBinaryPropagations(),
            'num_integer_propagations': solver.NumIntegerPropagations(),
            'skipped_constraints': skipped_constraints,
        }
        
        # Extract solution
        result = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for id in all_ids:
                for val in all_vals:
                    if solver.Value(x[id][val]) == 1:
                        result.append({'id': id, 'val': val})
                        break
    
    return result, num_equations, solver_metrics

def measure_by_aggregate(df: pd.DataFrame, reconstructed: List[Dict]) -> Dict[str, float]:
    """ Measures the accuracy of aggregate reconstruction.
    
    Args:
        df: DataFrame with QI columns and 'val' column (ground truth)
        reconstructed: Output from reconstruct_by_aggregate(), list of dicts with QI columns and 'val'
    
    Returns:
        Dict with two fractions:
        - 'qi_and_val_match': Fraction where QI columns AND value match a row in df
        - 'qi_match': Fraction where QI columns match a row in df (ignoring value)
    """
    if len(reconstructed) == 0:
        return {'qi_and_val_match': 0.0, 'qi_match': 0.0}
    
    # Get QI column names (all columns except 'id' and 'val')
    qi_cols = [col for col in df.columns if col not in ['id', 'val']]
    
    # Create sets for efficient lookup
    # For QI+val match: tuples of (qi_col1_val, qi_col2_val, ..., target_val)
    df_qi_and_val = set()
    for _, row in df.iterrows():
        qi_vals_tuple = tuple(row[col] for col in qi_cols)
        target_val = row['val']
        df_qi_and_val.add(qi_vals_tuple + (target_val,))
    
    # For QI-only match: tuples of (qi_col1_val, qi_col2_val, ...)
    df_qi_only = set()
    for _, row in df.iterrows():
        qi_vals_tuple = tuple(row[col] for col in qi_cols)
        df_qi_only.add(qi_vals_tuple)
    
    # Count matches in reconstructed
    qi_and_val_matches = 0
    qi_matches = 0
    
    for recon_row in reconstructed:
        # Extract QI values and target value from reconstructed row
        qi_vals_tuple = tuple(recon_row[col] for col in qi_cols)
        target_val = recon_row['val']
        
        # Check QI+val match
        if qi_vals_tuple + (target_val,) in df_qi_and_val:
            qi_and_val_matches += 1
        
        # Check QI-only match
        if qi_vals_tuple in df_qi_only:
            qi_matches += 1
    
    total_reconstructed = len(reconstructed)
    
    return {
        'qi_and_val_match': qi_and_val_matches / total_reconstructed,
        'qi_match': qi_matches / total_reconstructed
    }

def measure_by_row(df: pd.DataFrame, reconstructed: List[Dict]) -> float:
    """ Measures the accuracy of reconstruction.
    
    Args:
        df: DataFrame with columns 'id' and 'val' (ground truth)
        reconstructed: Output from reconstruct_by_row(), list of dicts with 'id' and 'val'
    
    Returns:
        Fraction of correct assignments (float between 0 and 1)
    """
    # Convert reconstructed to dict for easy lookup
    recon_dict = {item['id']: item['val'] for item in reconstructed}
    
    # Count correct assignments
    correct = 0
    total = 0
    
    for _, row in df.iterrows():
        id = row['id']
        true_val = row['val']
        
        if id in recon_dict:
            total += 1
            if recon_dict[id] == true_val:
                correct += 1
    
    return correct / total if total > 0 else 0.0


def reconstruct_by_aggregate(samples: List[Dict], noise: int, total_rows: int, all_qi_cols: List[str], seed: int = None) -> tuple[List[Dict], int, Dict]:
    """Reconstructs rows with QI column values and target values from aggregate samples.
    
    Args:
        samples: List of dicts, each containing:
            - 'qi_cols': list of QI column names (subset of all_qi_cols)
            - 'qi_vals': list of values for those QI columns (same length as qi_cols)
            - 'noisy_counts': list of dicts with 'val' (int) and 'count' (int)
        noise: Integer representing the noise bound (±noise from true count)
        total_rows: Exact number of rows to reconstruct
        all_qi_cols: List of all QI column names
        seed: Random seed for solver (default: None)
    
    Returns:
        Tuple of (reconstructed_rows, num_equations, solver_metrics)
        - reconstructed_rows: List of dicts with QI columns and 'val' (target value)
        - num_equations: Number of constraint equations used in the model
        - solver_metrics: Dict with all solver performance metrics
    """
    # Step 1: Validate inputs and extract domains
    all_qi_cols_set = set(all_qi_cols)
    qi_domains = {col: set() for col in all_qi_cols}
    all_target_vals = set()
    
    for sample_idx, sample in enumerate(samples):
        # Validate sample structure
        if 'qi_cols' not in sample or 'qi_vals' not in sample or 'noisy_counts' not in sample:
            raise ValueError(f"Sample {sample_idx} missing required keys (qi_cols, qi_vals, or noisy_counts)")
        
        qi_cols = sample['qi_cols']
        qi_vals = sample['qi_vals']
        
        # Validate qi_cols and qi_vals lengths match
        if len(qi_cols) != len(qi_vals):
            raise ValueError(f"Sample {sample_idx}: qi_cols length ({len(qi_cols)}) != qi_vals length ({len(qi_vals)})")
        
        # Validate all qi_cols are in all_qi_cols
        for col in qi_cols:
            if col not in all_qi_cols_set:
                raise ValueError(f"Sample {sample_idx}: qi_col '{col}' not in all_qi_cols")
        
        # Collect domain values
        for col, val in zip(qi_cols, qi_vals):
            qi_domains[col].add(val)
        
        # Collect target values
        for count_info in sample['noisy_counts']:
            all_target_vals.add(count_info['val'])
    
    # Convert domains to sorted lists
    qi_domains_sorted = {col: sorted(vals) for col, vals in qi_domains.items()}
    qi_domains = qi_domains_sorted
    all_target_vals = sorted(all_target_vals)
    
    num_equations = 0
    
    if check_gurobi_available():
        # Use Gurobi solver
        print("Using Gurobi solver")
        model = gp.Model("reconstruct_by_aggregate")
        model.setParam('OutputFlag', 0)  # Suppress output
        
        # Set random seed if provided
        if seed is not None:
            model.setParam('Seed', seed)
        
        # Step 2: Create binary variables
        x = {}  # x[row_idx][qi_col][qi_val]
        for row_idx in range(total_rows):
            x[row_idx] = {}
            for qi_col in all_qi_cols:
                x[row_idx][qi_col] = {}
                for qi_val in qi_domains[qi_col]:
                    x[row_idx][qi_col][qi_val] = model.addVar(
                        vtype=GRB.BINARY, 
                        name=f'x_{row_idx}_{qi_col}_{qi_val}'
                    )
        
        y = {}  # y[row_idx][target_val]
        for row_idx in range(total_rows):
            y[row_idx] = {}
            for target_val in all_target_vals:
                y[row_idx][target_val] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f'y_{row_idx}_{target_val}'
                )
        
        model.update()
        
        # Display variables
        print("\n=== MODEL VARIABLES ===")
        total_x_vars = sum(len(qi_domains[col]) for col in all_qi_cols) * total_rows
        total_y_vars = len(all_target_vals) * total_rows
        print(f"Binary variables x[row][qi_col][qi_val]: {total_x_vars}")
        print(f"Binary variables y[row][target_val]: {total_y_vars}")
        print(f"Total variables: {total_x_vars + total_y_vars}")
        
        print("\n=== MODEL CONSTRAINTS ===")
        
        # Step 3: Add basic assignment constraints
        print("\n1. Each row must have exactly one value per QI column:")
        for row_idx in range(total_rows):
            for qi_col in all_qi_cols:
                model.addConstr(
                    gp.quicksum(x[row_idx][qi_col][qi_val] for qi_val in qi_domains[qi_col]) == 1
                )
                num_equations += 1
        print(f"  Total: {total_rows * len(all_qi_cols)} constraints")
        
        print("\n2. Each row must have exactly one target value:")
        for row_idx in range(total_rows):
            model.addConstr(
                gp.quicksum(y[row_idx][target_val] for target_val in all_target_vals) == 1
            )
            num_equations += 1
        print(f"  Total: {total_rows} constraints")
        
        # Step 4: Add partial-match counting constraints
        print("\n3. Partial-match counting constraints (with auxiliary variables):")
        match_constraints_count = 0
        skipped_constraints = 0
        
        for sample_idx, sample in enumerate(samples):
            qi_cols = sample['qi_cols']
            qi_vals = sample['qi_vals']
            noisy_counts = sample['noisy_counts']
            
            for count_info in noisy_counts:
                target_val = count_info['val']
                noisy_count = count_info['count']
                
                # Clamp bounds: lower >= 0, upper <= total_rows
                lower_bound_val = max(0, noisy_count - noise)
                upper_bound_val = min(total_rows, noisy_count + noise)
                
                # Skip if constraint provides no information (covers entire range)
                if lower_bound_val == 0 and upper_bound_val == total_rows:
                    skipped_constraints += 2
                    continue
                
                # Create auxiliary match variables for each row
                match_vars = []
                for row_idx in range(total_rows):
                    match_var = model.addVar(
                        vtype=GRB.BINARY,
                        name=f'match_{sample_idx}_{row_idx}_{target_val}'
                    )
                    
                    # Build list of variables that must all be 1 for a match
                    and_vars = []
                    for col, val in zip(qi_cols, qi_vals):
                        and_vars.append(x[row_idx][col][val])
                    and_vars.append(y[row_idx][target_val])
                    
                    # Add AND constraint
                    model.addGenConstrAnd(match_var, and_vars)
                    num_equations += 1  # Count the AND constraint
                    
                    match_vars.append(match_var)
                
                # Add counting constraints
                total_matches = gp.quicksum(match_vars)
                model.addConstr(total_matches >= lower_bound_val)
                model.addConstr(total_matches <= upper_bound_val)
                num_equations += 2
                match_constraints_count += 2
        
        print(f"  Auxiliary match variables created for each sample-target-row combination")
        print(f"  Counting constraints (upper and lower bounds): {match_constraints_count}")
        if skipped_constraints > 0:
            print(f"  Skipped: {skipped_constraints} redundant constraints (covering entire range)")
        
        print(f"\n=== TOTAL EQUATIONS: {num_equations} ===\n")
        
        # Step 5: Solve and extract solution
        model.optimize()
        
        # Collect Gurobi metrics
        solver_metrics = {
            'solver': 'gurobi',
            'status': model.status,
            'status_string': {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 
                            5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT',
                            9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 
                            12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'}.get(model.status, 'UNKNOWN'),
            'runtime': model.Runtime,
            'obj_val': model.ObjVal if model.status == GRB.OPTIMAL else None,
            'obj_bound': model.ObjBound if model.status == GRB.OPTIMAL else None,
            'mip_gap': model.MIPGap if hasattr(model, 'MIPGap') else None,
            'node_count': model.NodeCount if hasattr(model, 'NodeCount') else None,
            'simplex_iterations': model.IterCount if hasattr(model, 'IterCount') else None,
            'barrier_iterations': model.BarIterCount if hasattr(model, 'BarIterCount') else None,
            'num_vars': model.NumVars,
            'num_constrs': model.NumConstrs,
            'num_sos': model.NumSOS if hasattr(model, 'NumSOS') else None,
            'num_qconstrs': model.NumQConstrs if hasattr(model, 'NumQConstrs') else None,
            'num_genconstrs': model.NumGenConstrs if hasattr(model, 'NumGenConstrs') else None,
            'num_nzs': model.NumNZs if hasattr(model, 'NumNZs') else None,
            'num_int_vars': model.NumIntVars if hasattr(model, 'NumIntVars') else None,
            'num_bin_vars': model.NumBinVars if hasattr(model, 'NumBinVars') else None,
            'is_mip': model.IsMIP,
            'is_qp': model.IsQP,
            'is_qcp': model.IsQCP,
            'skipped_constraints': skipped_constraints,
        }
        
        result = []
        if model.status == GRB.OPTIMAL:
            for row_idx in range(total_rows):
                row_dict = {}
                
                # Extract QI column values
                for qi_col in all_qi_cols:
                    for qi_val in qi_domains[qi_col]:
                        if x[row_idx][qi_col][qi_val].X > 0.5:
                            row_dict[qi_col] = qi_val
                            break
                
                # Extract target value
                for target_val in all_target_vals:
                    if y[row_idx][target_val].X > 0.5:
                        row_dict['val'] = target_val
                        break
                
                result.append(row_dict)
        
    else:
        # Use OR-Tools solver
        print("Using OR-Tools CP-SAT solver")
        model = cp_model.CpModel()
        
        # Step 2: Create binary variables
        x = {}  # x[row_idx][qi_col][qi_val]
        for row_idx in range(total_rows):
            x[row_idx] = {}
            for qi_col in all_qi_cols:
                x[row_idx][qi_col] = {}
                for qi_val in qi_domains[qi_col]:
                    x[row_idx][qi_col][qi_val] = model.NewBoolVar(
                        f'x_{row_idx}_{qi_col}_{qi_val}'
                    )
        
        y = {}  # y[row_idx][target_val]
        for row_idx in range(total_rows):
            y[row_idx] = {}
            for target_val in all_target_vals:
                y[row_idx][target_val] = model.NewBoolVar(
                    f'y_{row_idx}_{target_val}'
                )
        
        # Display variables
        print("\n=== MODEL VARIABLES ===")
        total_x_vars = sum(len(qi_domains[col]) for col in all_qi_cols) * total_rows
        total_y_vars = len(all_target_vals) * total_rows
        print(f"Binary variables x[row][qi_col][qi_val]: {total_x_vars}")
        print(f"Binary variables y[row][target_val]: {total_y_vars}")
        print(f"Total variables: {total_x_vars + total_y_vars}")
        
        print("\n=== MODEL CONSTRAINTS ===")
        
        # Step 3: Add basic assignment constraints
        print("\n1. Each row must have exactly one value per QI column:")
        for row_idx in range(total_rows):
            for qi_col in all_qi_cols:
                model.Add(
                    sum(x[row_idx][qi_col][qi_val] for qi_val in qi_domains[qi_col]) == 1
                )
                num_equations += 1
        print(f"  Total: {total_rows * len(all_qi_cols)} constraints")
        
        print("\n2. Each row must have exactly one target value:")
        for row_idx in range(total_rows):
            model.Add(
                sum(y[row_idx][target_val] for target_val in all_target_vals) == 1
            )
            num_equations += 1
        print(f"  Total: {total_rows} constraints")
        
        # Step 4: Add partial-match counting constraints
        print("\n3. Partial-match counting constraints (with auxiliary variables):")
        match_constraints_count = 0
        skipped_constraints = 0
        
        for sample_idx, sample in enumerate(samples):
            qi_cols = sample['qi_cols']
            qi_vals = sample['qi_vals']
            noisy_counts = sample['noisy_counts']
            
            for count_info in noisy_counts:
                target_val = count_info['val']
                noisy_count = count_info['count']
                
                # Clamp bounds: lower >= 0, upper <= total_rows
                lower_bound_val = max(0, noisy_count - noise)
                upper_bound_val = min(total_rows, noisy_count + noise)
                
                # Skip if constraint provides no information (covers entire range)
                if lower_bound_val == 0 and upper_bound_val == total_rows:
                    skipped_constraints += 2
                    continue
                
                # Create auxiliary match variables for each row
                match_vars = []
                for row_idx in range(total_rows):
                    match_var = model.NewBoolVar(
                        f'match_{sample_idx}_{row_idx}_{target_val}'
                    )
                    
                    # Build list of variables that must all be 1 for a match
                    and_vars = []
                    for col, val in zip(qi_cols, qi_vals):
                        and_vars.append(x[row_idx][col][val])
                    and_vars.append(y[row_idx][target_val])
                    
                    # Add AND constraint using AddBoolAnd
                    model.AddBoolAnd(and_vars).OnlyEnforceIf(match_var)
                    model.AddBoolOr([v.Not() for v in and_vars]).OnlyEnforceIf(match_var.Not())
                    num_equations += 2  # Count both implications
                    
                    match_vars.append(match_var)
                
                # Add counting constraints
                total_matches = sum(match_vars)
                model.Add(total_matches >= lower_bound_val)
                model.Add(total_matches <= upper_bound_val)
                num_equations += 2
                match_constraints_count += 2
        
        print(f"  Auxiliary match variables created for each sample-target-row combination")
        print(f"  Counting constraints (upper and lower bounds): {match_constraints_count}")
        if skipped_constraints > 0:
            print(f"  Skipped: {skipped_constraints} redundant constraints (covering entire range)")
        
        print(f"\n=== TOTAL EQUATIONS: {num_equations} ===\n")
        
        # Step 5: Solve and extract solution
        solver = cp_model.CpSolver()
        
        # Set random seed if provided
        if seed is not None:
            solver.parameters.random_seed = seed
        
        status = solver.Solve(model)
        
        # Collect OR-Tools metrics
        solver_metrics = {
            'solver': 'ortools',
            'status': status,
            'status_string': {0: 'UNKNOWN', 1: 'MODEL_INVALID', 2: 'FEASIBLE', 3: 'INFEASIBLE', 4: 'OPTIMAL'}.get(status, 'UNKNOWN'),
            'runtime': solver.WallTime(),
            'user_time': solver.UserTime(),
            'obj_val': solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
            'best_obj_bound': solver.BestObjectiveBound() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
            'num_booleans': solver.NumBooleans(),
            'num_conflicts': solver.NumConflicts(),
            'num_branches': solver.NumBranches(),
            'num_binary_propagations': solver.NumBinaryPropagations(),
            'num_integer_propagations': solver.NumIntegerPropagations(),
            'skipped_constraints': skipped_constraints,
        }
        
        result = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for row_idx in range(total_rows):
                row_dict = {}
                
                # Extract QI column values
                for qi_col in all_qi_cols:
                    for qi_val in qi_domains[qi_col]:
                        if solver.Value(x[row_idx][qi_col][qi_val]) == 1:
                            row_dict[qi_col] = qi_val
                            break
                
                # Extract target value
                for target_val in all_target_vals:
                    if solver.Value(y[row_idx][target_val]) == 1:
                        row_dict['val'] = target_val
                        break
                
                result.append(row_dict)
    
    return result, num_equations, solver_metrics

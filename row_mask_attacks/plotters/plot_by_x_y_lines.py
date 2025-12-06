import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_by_x_y_lines(df: pd.DataFrame, x_col: str, y_col: str, lines_col: str, thresh: float = 0.95, thresh_direction: str = 'lowest'):
    """Plot lowest/highest y value where measure >= threshold for each (x, lines) pair.
    
    For each pair of values (x, l) in x_col and lines_col, find the lowest or highest value y 
    of y_col where measure >= thresh. Connect points with the same lines_col value.
    When no data exceeds threshold for an x value, display "None" above that x-tick.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis (find lowest/highest value where measure >= thresh)
        lines_col: Column name for line grouping
        thresh: Threshold for measure column (default: 0.95)
        thresh_direction: 'lowest' or 'highest' - which y value to select (default: 'lowest')
        output_dir: Directory to save plot (default: results/row_mask_attacks/plots)
    """

    maps = {          'nrows': "Number rows",
                      'nunique': "Distinct target values",
                      'noise': "Noise",
                      'nqi': "Number QI columns",
                      'min_num_rows': "Suppress threshold",
                      'vals_per_qi': "Distinct QI values",
                   }
    dashed_columns = {'nrows': 150,
                      'nunique': 2,
                      'noise': 4,
                      'nqi': 6,
                      'min_num_rows': 5,
                      'vals_per_qi': 0,
                   }
    reportable_columns = ['nrows', 'nunique', 'noise', 'nqi', 'min_num_rows', 'vals_per_qi',]
    output_dir = Path('./results/row_mask_attacks/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if thresh_direction not in ['lowest', 'highest']:
        raise ValueError(f"thresh_direction must be 'lowest' or 'highest', got '{thresh_direction}'")
    
    # Extract reportable column values (excluding x, y, lines columns)
    used_cols = {x_col, y_col, lines_col}
    display_cols = [col for col in reportable_columns if col not in used_cols and col in df.columns]
    
    reportable_values = {}
    for col in display_cols:
        unique_vals = df[col].unique()
        if len(unique_vals) == 1:
            display_name = maps.get(col, col)
            val = unique_vals[0]
            # Display "auto" instead of 0 for vals_per_qi
            if col == 'vals_per_qi' and val == 0:
                val = 'auto'
            reportable_values[display_name] = val
        else:
            print(f"Warning: Column '{col}' has {len(unique_vals)} unique values, expected 1")
            display_name = maps.get(col, col)
            reportable_values[display_name] = f"Multiple ({len(unique_vals)})"
    
    # Get unique values for x and lines
    x_values = sorted(df[x_col].unique())
    line_values = sorted(df[lines_col].unique())
    
    # For each (x, line) pair, find the lowest/highest y value where measure >= thresh
    plot_data = []
    x_values_with_none = set()  # Track x values where ALL lines have no valid data
    
    for x_val in x_values:
        has_any_valid_data = False
        
        for line_val in line_values:
            # Filter to this (x, line) combination
            subset = df[(df[x_col] == x_val) & (df[lines_col] == line_val)]
            
            if len(subset) == 0:
                continue
            
            # Find rows where measure >= thresh
            valid_rows = subset[subset['measure'] >= thresh]
            
            if len(valid_rows) > 0:
                # Get the lowest or highest y value from rows that meet threshold
                if thresh_direction == 'lowest':
                    y_val = valid_rows[y_col].min()
                else:  # highest
                    y_val = valid_rows[y_col].max()
                
                plot_data.append({
                    'x': x_val,
                    'y': y_val,
                    'line': line_val
                })
                has_any_valid_data = True
        
        # If no line had valid data for this x value, mark it
        if not has_any_valid_data:
            x_values_with_none.add(x_val)
    
    if len(plot_data) == 0 and len(x_values_with_none) == 0:
        print(f"No data points found")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
    # Create a line for each line_val
    for idx, line_val in enumerate(line_values):
        line_data = [p for p in plot_data if p['line'] == line_val]
        
        if len(line_data) > 0:
            # Sort by x value
            line_data = sorted(line_data, key=lambda p: p['x'])
            
            x_vals = [p['x'] for p in line_data]
            y_vals = [p['y'] for p in line_data]
            
            # Determine if this line should be dashed
            linestyle = 'solid'
            linewidth = 2
            if lines_col in dashed_columns and line_val == dashed_columns[lines_col]:
                linestyle = 'dashed'
                linewidth = 4
            
            # Get display name for lines_col
            lines_display = maps.get(lines_col, lines_col)
            
            # Format line_val for display (show "auto" for vals_per_qi=0)
            display_line_val = line_val
            if lines_col == 'vals_per_qi' and line_val == 0:
                display_line_val = 'auto'
            
            # Plot line with different marker for each line
            ax.plot(x_vals, y_vals, marker=markers[idx % len(markers)], 
                    linewidth=linewidth, markersize=8, linestyle=linestyle,
                    label=f'{lines_display}={display_line_val}')
    
    # Determine positioning for "None" labels
    if len(plot_data) > 0:
        y_values = [p['y'] for p in plot_data]
        y_min_data = min(y_values)
        y_max_data = max(y_values)
        
        # Position "None" at the minimum y value
        none_y_pos = y_min_data
        
        # Check if we should use log scale
        if all(y > 0 for y in y_values) and y_max_data / y_min_data > 10:
            ax.set_yscale('log')
    else:
        # No valid data points, only "None" labels
        none_y_pos = 0.5
        ax.set_ylim(0, 1)
    
    # Add "None" labels for x values with no valid data
    for x_val in x_values_with_none:
        ax.text(x_val, none_y_pos, 'None', 
                ha='center', va='bottom', fontsize=10, 
                style='italic', color='red', fontweight='bold')
    
    # Add text box with reportable values
    if reportable_values:
        text_lines = [f'{col}: {val}' for col, val in reportable_values.items()]
        text_str = '\n'.join(text_lines)
        
        # Find best position for text box to avoid overlapping with data
        # Try corners in order: upper-right, upper-left, lower-right, lower-left
        positions = [
            (0.98, 0.98, 'top', 'right'),      # upper-right
            (0.02, 0.98, 'top', 'left'),       # upper-left
            (0.98, 0.02, 'bottom', 'right'),   # lower-right
            (0.02, 0.02, 'bottom', 'left'),    # lower-left
        ]
        
        # Calculate density of data points in each quadrant
        if len(plot_data) > 0:
            # Normalize plot data to 0-1 range
            x_plot_vals = [p['x'] for p in plot_data]
            y_plot_vals = [p['y'] for p in plot_data]
            x_min, x_max = min(x_plot_vals), max(x_plot_vals)
            y_min, y_max = min(y_plot_vals), max(y_plot_vals)
            
            # Avoid division by zero
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1
            
            # Count points in each quadrant (upper-right, upper-left, lower-right, lower-left)
            quadrant_counts = [0, 0, 0, 0]
            for p in plot_data:
                x_norm = (p['x'] - x_min) / x_range
                y_norm = (p['y'] - y_min) / y_range
                
                if x_norm >= 0.5 and y_norm >= 0.5:
                    quadrant_counts[0] += 1  # upper-right
                elif x_norm < 0.5 and y_norm >= 0.5:
                    quadrant_counts[1] += 1  # upper-left
                elif x_norm >= 0.5 and y_norm < 0.5:
                    quadrant_counts[2] += 1  # lower-right
                else:
                    quadrant_counts[3] += 1  # lower-left
            
            # Find position with fewest data points
            best_idx = quadrant_counts.index(min(quadrant_counts))
            x_pos, y_pos, va, ha = positions[best_idx]
        else:
            # Default to upper-right if no data
            x_pos, y_pos, va, ha = positions[0]
        
        # Position text box
        ax.text(x_pos, y_pos, text_str,
                transform=ax.transAxes,
                verticalalignment=va,
                horizontalalignment=ha,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9,
                family='monospace')
    
    # Set x-axis to show all x values and add margin on the left
    ax.set_xticks(x_values)
    if len(x_values) > 1:
        x_range = x_values[-1] - x_values[0]
        left_margin = x_range * 0.05  # 5% margin on the left
        ax.set_xlim(left=x_values[0] - left_margin)
    
    # Get display names for axes and title
    x_display = maps.get(x_col, x_col)
    y_display = maps.get(y_col, y_col)
    lines_display = maps.get(lines_col, lines_col)
    
    ax.set_xlabel(x_display)
    ax.set_ylabel(f'{y_display} ({thresh_direction} where measure >= {thresh})')
    ax.set_title(f'{thresh_direction.capitalize()} {y_display} (measure ≥ {thresh}) by {x_display} and {lines_display}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'plot_by_x_{x_col}_y_{y_col}_l_{lines_col}_thr_{thresh}.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved: {filepath}")

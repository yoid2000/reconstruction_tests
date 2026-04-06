import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Optional, List
import re


def _sanitize_filename_part(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')


def _debug_multi_value_column(
    df: pd.DataFrame,
    col: str,
    x_col: str,
    y_col: str,
    lines_col: str,
    tag: str,
    output_dir: Path,
    max_groups: int = 20,
):
    """Print and persist diagnostics when a reportable column has >1 unique value."""
    if col not in df.columns:
        return

    group_cols = [c for c in [x_col, lines_col] if c in df.columns]
    if len(group_cols) != 2:
        return

    nunique_by_group = (
        df.groupby(group_cols, dropna=False)[col]
        .nunique(dropna=False)
        .reset_index(name='nunique_values')
    )
    problem_groups = nunique_by_group[nunique_by_group['nunique_values'] > 1].copy()

    if len(problem_groups) == 0:
        value_counts = df[col].value_counts(dropna=False).to_dict()
        print(
            f"Debug: '{col}' has multiple global values for plot x={x_col}, y={y_col}, lines={lines_col}, "
            f"but each ({x_col}, {lines_col}) group is consistent. Global counts: {value_counts}"
        )
        return

    print(
        f"Debug: '{col}' varies within {len(problem_groups)} ({x_col}, {lines_col}) groups "
        f"for plot x={x_col}, y={y_col}, lines={lines_col}, tag='{tag}'."
    )

    for _, grp in problem_groups.head(max_groups).iterrows():
        x_val = grp[x_col]
        line_val = grp[lines_col]
        subset = df[(df[x_col] == x_val) & (df[lines_col] == line_val)]
        value_counts = subset[col].value_counts(dropna=False).to_dict()
        print(
            f"  {x_col}={x_val}, {lines_col}={line_val}: "
            f"n_rows={len(subset)}, {col}_counts={value_counts}"
        )

    if len(problem_groups) > max_groups:
        print(f"  ... {len(problem_groups) - max_groups} additional groups omitted from stdout")

    debug_cols = [
        c for c in [
            x_col,
            y_col,
            lines_col,
            col,
            'noise',
            'nqi',
            'nrows',
            'nunique',
            'supp_thresh',
            'actual_vals_per_qi',
            'vals_per_qi',
            'solve_type',
            'filename',
            'seed',
        ]
        if c in df.columns
    ]

    mask = pd.Series(False, index=df.index)
    for _, grp in problem_groups.iterrows():
        mask |= (df[x_col] == grp[x_col]) & (df[lines_col] == grp[lines_col])

    debug_df = df.loc[mask, debug_cols].copy()
    debug_dir = output_dir / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"multi_{_sanitize_filename_part(col)}"
        f"_x_{_sanitize_filename_part(x_col)}"
        f"_y_{_sanitize_filename_part(y_col)}"
        f"_l_{_sanitize_filename_part(lines_col)}"
        f"_tag_{_sanitize_filename_part(tag or 'none')}.csv"
    )
    debug_path = debug_dir / filename
    debug_df.to_csv(debug_path, index=False)
    print(f"Debug: wrote {len(debug_df)} rows to {debug_path}")


def plot_by_x_y_lines(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    lines_col: str,
    thresh: float = 0.95,
    thresh_direction: str = 'lowest',
    show_defaults: bool = False,
    tag: str = '',
    extra_y_cols: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    measure_col: str = 'measure',
    metric_label: str = 'Accuracy',
):
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
        output_dir: Directory to save plot (default: results/plots)
    """

    if extra_y_cols is None:
        extra_y_cols = []

    if output_dir is None:
        output_dir = Path('./results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    thresh_str = ''
    dir_str = ''
    if y_col == 'noise' or x_col == 'noise':
        thresh_str = f"{thresh:.3f}"
        dir_str = thresh_direction
    filename = f'x_{x_col}_y_{y_col}_l_{lines_col}_{thresh_str}_{dir_str}_{tag}.png'
    filepath = output_dir / filename
    if filepath.exists():
        print(f"plot_by_x_y_lines: plot already exists: {filepath}; skipping plot")
        return

    maps = {          'nrows': "Number rows",
                      'nunique': "Target values",
                      'noise': "Noise",
                      'nqi': "Number QI columns",
                      'supp_thresh': "Suppress thresh",
                      'vals_per_qi': "QI values",
                      'actual_vals_per_qi': "QI values",
                      'known_qi_fraction': "Known QI fraction",
                      'measure': metric_label,
                      'num_samples': "Number of samples",
                      'mixing_avg': "Mixing average",
                      'separation_average': "Separation average",
                      'max_qi': "Max QI columns",
                      'num_suppressed': "Number suppressed",
                      'med_solver_metrics_runtime': "Median solver runtime (s)",
                   }
    dashed_columns = {'nrows': 150,
                      'nunique': 2,
                      'noise': 2,
                      'nqi': 6,
                      'supp_thresh': 2,
                      'actual_vals_per_qi': 2,
                      'max_qi': 1000,
                      'known_qi_fraction': 1.0,
                      }
    reportable_columns = ['max_qi', 'nrows', 'nunique', 'noise', 'nqi', 'supp_thresh', 'actual_vals_per_qi','known_qi_fraction']
    
    if thresh_direction not in ['lowest', 'highest']:
        raise ValueError(f"thresh_direction must be 'lowest' or 'highest', got '{thresh_direction}'")
    
    # Extract reportable column values (excluding x, y, lines columns)
    used_cols = {x_col, y_col, lines_col}
    display_cols = [col for col in reportable_columns if col not in used_cols and col in df.columns]

    if lines_col != 'noise' and y_col != 'noise' and x_col != 'noise':
        default_noise = dashed_columns.get('noise')
        df = df[df['noise'] == default_noise]
    if 'known_qi_fraction' in df.columns and 'known_qi_fraction' not in used_cols:
        default_kqf = dashed_columns.get('known_qi_fraction', 1.0)
        before_count = len(df)
        df = df[np.isclose(df['known_qi_fraction'].astype(float), float(default_kqf))]
        after_count = len(df)
        if after_count < before_count:
            print(
                "plot_by_x_y_lines: filtered known_qi_fraction "
                f"to {default_kqf} for x={x_col}, y={y_col}, lines={lines_col}: "
                f"{before_count} -> {after_count} rows"
            )
    
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
            display_name = maps.get(col, col)
            reportable_values[display_name] = f"Multiple ({len(unique_vals)})"
            if col != 'actual_vals_per_qi':
                print(f"Warning: Column '{col}' has {len(unique_vals)} unique values, expected 1")
                _debug_multi_value_column(
                    df=df,
                    col=col,
                    x_col=x_col,
                    y_col=y_col,
                    lines_col=lines_col,
                    tag=tag,
                    output_dir=output_dir,
                )
    
    # Get unique values for x and lines
    x_values = sorted(df[x_col].unique())
    line_values = sorted(df[lines_col].unique())
    
    # For each (x, line) pair, find the lowest/highest y value where measure >= thresh
    plot_data = []
    extra_plot_datas = [[] for _ in extra_y_cols]
    print('extra_plot_datas:', extra_plot_datas)
    x_values_with_none = set()  # Track x values where ALL lines have no valid data
    
    for x_val in x_values:
        has_any_valid_data = False
        
        for line_val in line_values:
            # Filter to this (x, line) combination
            df_subset = df[(df[x_col] == x_val) & (df[lines_col] == line_val)]
            
            if len(df_subset) == 0:
                continue
            
            ylabel_note = ''
            if lines_col == 'noise' or (y_col != 'noise' and x_col != 'noise'):
                if len(df_subset) != 1:
                    # throw exception
                    for col in df_subset.columns:
                        # check if all values in this column are the same
                        if len(df_subset[col].unique()) > 1:
                            print(f"Column '{col}' has multiple values for x={x_val}, {lines_col}={line_val}: {df_subset[col].unique()}")
                    raise ValueError(f"Expected exactly one row for {x_col}={x_val}, {lines_col}={line_val}, got {len(df_subset)} rows")
                row_used = df_subset.iloc[0]
                y_val = row_used[y_col]
                target_accuracy = row_used.get(measure_col, float('nan')) >= row_used.get('target_accuracy', float('inf'))
                
                plot_data.append({
                    'x': x_val,
                    'y': y_val,
                    'target_accuracy': target_accuracy,
                    'line': line_val
                })
                has_any_valid_data = True
            else:
                ylabel_note = f' ({metric_label} >= {thresh})'
                # Original threshold-based logic for other y columns
                # Find rows where measure >= thresh
                df_valid_rows = df_subset[df_subset[measure_col] >= thresh]
                
                if len(df_valid_rows) > 0:
                    # Get the lowest or highest y value from rows that meet threshold
                    if thresh_direction == 'lowest':
                        row_used = df_valid_rows.loc[df_valid_rows[y_col].idxmin()]
                    elif thresh_direction == 'highest':
                        row_used = df_valid_rows.loc[df_valid_rows[y_col].idxmax()]
                    else:
                        row_used = df_valid_rows.iloc[0]
                    y_val = row_used[y_col]
                    target_accuracy = row_used.get(measure_col, float('nan')) >= row_used.get('target_accuracy', float('inf'))
                    
                    plot_data.append({
                        'x': x_val,
                        'y': y_val,
                        'target_accuracy': False,
                        'line': line_val
                    })
                    for extra_plot_data, extra_y_col in zip(extra_plot_datas, extra_y_cols):
                        extra_y_val = row_used[extra_y_col]
                        extra_plot_data.append({
                            'x': x_val,
                            'y': extra_y_val,
                            'target_accuracy': target_accuracy,
                            'line': line_val
                        })
                        print(f"Extra plot data added: x={x_val}, y={extra_y_val}, line={line_val}")
                    has_any_valid_data = True
        
        # If no line had valid data for this x value, mark it
        if not has_any_valid_data:
            x_values_with_none.add(x_val)
    
    if len(plot_data) == 0 and len(x_values_with_none) == 0:
        print(f"No data points found")
        return
    
    do_x_y_plot(df, plot_data, line_values, x_values, x_col, y_col, lines_col, ylabel_note, thresh_str, dir_str, output_dir, show_defaults, reportable_values, tag, dashed_columns, maps, x_values_with_none)
    for extra_y_col, extra_plot_data in zip(extra_y_cols, extra_plot_datas):
        do_x_y_plot(df, extra_plot_data, line_values, x_values, x_col, extra_y_col, lines_col, ylabel_note, thresh_str, dir_str, output_dir, show_defaults, reportable_values, f"{tag}_extra", dashed_columns, maps, x_values_with_none)

def do_x_y_plot(df, plot_data, line_values, x_values, x_col, y_col, lines_col, ylabel_note, thresh_str, dir_str, output_dir, show_defaults, reportable_values, tag, dashed_columns, maps, x_values_with_none):
    # Create plot
    fig, ax = plt.subplots(figsize=(5, 4))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
    # Color palette to avoid cycling repeats when there are many lines
    #cmap = plt.get_cmap('tab20')        # has dark/light pairs
    cmap = plt.get_cmap('tab10')
    reserved_dashed_color = '#444444'
    reserved_dashed_marker = 'X'
    color_idx = 0  # for non-dashed lines
    marker_idx = 0  # for non-dashed lines

    # Create a line for each line_val
    for idx, line_val in enumerate(line_values):
        line_data = [p for p in plot_data if p['line'] == line_val]
        
        if len(line_data) > 0:
            # Sort by x value
            line_data = sorted(line_data, key=lambda p: p['x'])
            
            x_vals = [p['x'] for p in line_data]
            y_vals = [p['y'] for p in line_data]
            linestyle = 'solid'
            linewidth = 2
            is_dashed_line = lines_col in dashed_columns and line_val == dashed_columns[lines_col]
            if is_dashed_line:
                linestyle = 'dashed'
                linewidth = 3
                color = reserved_dashed_color
                marker = reserved_dashed_marker
            else:
                color = cmap(color_idx % cmap.N)
                marker = markers[marker_idx % len(markers)]
                color_idx += 1
                marker_idx += 1
            
            # Get display name for lines_col
            lines_display = maps.get(lines_col, lines_col)
            
            # Format line_val for display (show "auto" for vals_per_qi=0)
            display_line_val = line_val
            if lines_col == 'vals_per_qi' and line_val == 0:
                display_line_val = 'auto'
            
            # Plot main line using y_vals with different marker for each line
            line_obj = ax.plot(
                x_vals,
                y_vals,
                marker=marker,
                linewidth=linewidth,
                markersize=8,
                linestyle=linestyle,
                label=f'{lines_display}={display_line_val}',
                color=color,
                alpha=1.0,
            )
            
            # Highlight points that reached target_accuracy with a white dot overlay
            ta_x = [p['x'] for p in line_data if p.get('target_accuracy')]
            ta_y = [p['y'] for p in line_data if p.get('target_accuracy')]
            if ta_x:
                ax.plot(
                    ta_x,
                    ta_y,
                    marker=marker,
                    linestyle='none',
                    markersize=8.0,
                    markerfacecolor='white',
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                )
            
    # Determine positioning for "None" labels
    if len(plot_data) > 0:
        y_values = [p['y'] for p in plot_data]
        y_min_data = min(y_values)
        y_max_data = max(y_values)
        
        # Position "None" at the minimum y value
        none_y_pos = y_min_data
        
        # Check if we should use log scale
        if all(y > 0 for y in y_values) and y_max_data / y_min_data > 500:
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
    if show_defaults and reportable_values:
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
        text_artist = ax.text(x_pos, y_pos, text_str,
                              transform=ax.transAxes,
                              verticalalignment=va,
                              horizontalalignment=ha,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                              fontsize=8,
                              family='monospace')

        # Draw an invisible rectangle around the text unless it covers >=2 points
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_disp = text_artist.get_window_extent(renderer=renderer)
        bbox_data = bbox_disp.transformed(ax.transData.inverted())
        overlap_count = sum(
            1 for p in plot_data
            if bbox_data.x0 <= p['x'] <= bbox_data.x1 and bbox_data.y0 <= p['y'] <= bbox_data.y1
        )
        if overlap_count >= 2:
            text_artist.remove()
        else:
            bbox_axes = bbox_disp.transformed(ax.transAxes.inverted())
            reserve_patch = Rectangle((bbox_axes.x0, bbox_axes.y0),
                                      bbox_axes.width, bbox_axes.height,
                                      transform=ax.transAxes,
                                      facecolor='none', edgecolor='none')
            reserve_patch.set_in_layout(False)
            ax.add_patch(reserve_patch)
    
    # Set x-axis to show all x values and add margin on the left
    ax.set_xticks(x_values)
    
    # Special handling for nqi x-axis labels
    if x_col == 'nqi' and lines_col != 'nrows' and lines_col != 'actual_vals_per_qi':
        # Create labels showing "nqi (actual_vals_per_qi)"
        tick_labels = []
        for x_val in x_values:
            # Get actual_vals_per_qi for this nqi value
            df_subset = df[df[x_col] == x_val]
            if 'actual_vals_per_qi' in df.columns and len(df_subset) > 0:
                actual_vals = df_subset['actual_vals_per_qi'].unique()
                if len(actual_vals) == 1:
                    actual_val = int(actual_vals[0])  # Convert to integer
                    tick_labels.append(f'{x_val} ({actual_val})')
                else:
                    # Multiple actual values, just show nqi
                    tick_labels.append(str(x_val))
            else:
                # No actual_vals_per_qi column, just show nqi
                tick_labels.append(str(x_val))
        ax.set_xticklabels(tick_labels)
      
    if len(x_values) > 1:
        x_range = x_values[-1] - x_values[0]
        left_margin = x_range * 0.05  # 5% margin on the left
        ax.set_xlim(left=x_values[0] - left_margin)
    
    # Get display names for axes and title
    x_display = maps.get(x_col, x_col)
    y_display = maps.get(y_col, y_col)
    lines_display = maps.get(lines_col, lines_col)
    
    # Modify x_display if showing actual_vals_per_qi in parentheses
    if x_col == 'nqi' and lines_col != 'nrows' and lines_col != 'actual_vals_per_qi':
        x_display = f"{x_display} ({maps.get('actual_vals_per_qi', 'actual_vals_per_qi')})"
    
    ax.set_xlabel(x_display, fontsize=14)
    ax.set_ylabel(f'{y_display} {ylabel_note}', fontsize=14)
    #ax.set_title(f'{thresh_direction.capitalize()} {y_display} (measure ≥ {thresh}) by {x_display} and {lines_display}')
    # Smaller, tighter legend to reduce footprint
    ax.legend(fontsize=9, framealpha=0.85, labelspacing=0.35, handlelength=1.8, borderpad=0.4)

    # If using log scale on y, ensure the highest tick mark has a label
    if ax.get_yscale() == 'log':
        fig.canvas.draw()
        yticks = ax.get_yticks()
        formatter = ax.yaxis.get_major_formatter()
        labels = [formatter(tick, idx) for idx, tick in enumerate(yticks)]
        if labels and (labels[-1] is None or str(labels[-1]).strip() == ''):
            labels[-1] = f"{yticks[-1]:g}"
            ax.set_yticks(yticks)
            ax.set_yticklabels(labels)

    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    for plottype in ['png', 'pdf']:
        filename = f'x_{x_col}_y_{y_col}_l_{lines_col}_{thresh_str}_{dir_str}_{tag}.{plottype}'
        filepath = output_dir / filename
        plt.savefig(filepath)
        print(f"Saved: {filepath}")
    plt.close()

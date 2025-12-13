import pandas as pd
from pathlib import Path
from typing import Optional


def make_noise_min_num_rows_table(df: pd.DataFrame, note: str, thresh: float = 0.9):
    """Generate tables showing max measure by noise, min_num_rows, and nrows.
    
    Creates both a text table and a LaTeX table showing maximum measure
    for each combination of noise, min_num_rows, and nrows.
    
    Args:
        df: DataFrame with columns 'noise', 'min_num_rows', 'nrows', and 'measure'
        note: String to include in output filenames, labels, and captions
        thresh: Threshold for bolding values in LaTeX (default: 0.9)
    """
    
    # Create output directory
    output_dir = Path('./results/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique sorted values
    noise_values = sorted(df['noise'].unique())
    min_num_rows_values = sorted(df['min_num_rows'].unique())
    nrows_values = sorted(df['nrows'].unique())
    
    # Build the data structure: dict[noise][min_num_rows][nrows] = max
    table_data = {}
    for noise in noise_values:
        table_data[noise] = {}
        for mnr in min_num_rows_values:
            table_data[noise][mnr] = {}
            for nrows in nrows_values:
                # Filter data
                subset = df[(df['noise'] == noise) & 
                           (df['min_num_rows'] == mnr) & 
                           (df['nrows'] == nrows)]
                
                if len(subset) > 0 and 'measure' in subset.columns:
                    max_measure = subset['measure'].max()
                    table_data[noise][mnr][nrows] = max_measure
                else:
                    table_data[noise][mnr][nrows] = None
    
    # Generate text table
    text_output = _generate_text_table(table_data, noise_values, min_num_rows_values, nrows_values)
    text_file = output_dir / f'noise_min_num_rows_{note}.txt'
    with open(text_file, 'w') as f:
        f.write(text_output)
    print(f"Saved: {text_file}")
    
    # Generate LaTeX table
    latex_output = _generate_latex_table(table_data, noise_values, min_num_rows_values, nrows_values, note, thresh)
    latex_file = output_dir / f'noise_min_num_rows_{note}.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_output)
    print(f"Saved: {latex_file}")


def _generate_text_table(table_data: dict, noise_values: list, min_num_rows_values: list, nrows_values: list) -> str:
    """Generate a fixed-width text table."""
    
    lines = []
    
    # Header row
    header_parts = ['Noise', 'MinRows']
    for nrows in nrows_values:
        header_parts.append(f'n={nrows}')
    
    # Calculate column widths - more compact
    col_widths = [6, 8]  # noise and min_num_rows
    data_col_width = 6  # for max value only (e.g., "0.00")
    col_widths.extend([data_col_width] * len(nrows_values))
    
    # Format header
    header_line = ''
    for i, part in enumerate(header_parts):
        header_line += part.ljust(col_widths[i]) + ' '
    lines.append(header_line.rstrip())
    
    # Separator line
    sep_line = ''
    for width in col_widths:
        sep_line += '-' * width + ' '
    lines.append(sep_line.rstrip())
    
    # Data rows
    for noise in noise_values:
        for mnr in min_num_rows_values:
            row_parts = [str(int(noise)), str(int(mnr))]
            
            for nrows in nrows_values:
                value = table_data[noise][mnr][nrows]
                if value is None:
                    cell = '---'
                else:
                    cell = f'{value:.2f}'
                row_parts.append(cell)
            
            # Format row
            row_line = ''
            for i, part in enumerate(row_parts):
                row_line += part.ljust(col_widths[i]) + ' '
            lines.append(row_line.rstrip())
    
    return '\n'.join(lines)


def _generate_latex_table(table_data: dict, noise_values: list, min_num_rows_values: list, 
                          nrows_values: list, note: str, thresh: float) -> str:
    """Generate a LaTeX table using booktabs."""
    
    lines = []
    
    # Table preamble
    num_cols = 2 + len(nrows_values)
    col_spec = 'rr' + 'r' * len(nrows_values)
    
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    
    # Header row
    header_parts = ['Noise', 'Min Rows']
    for nrows in nrows_values:
        header_parts.append(f'$n={nrows}$')
    lines.append(' & '.join(header_parts) + ' \\\\')
    lines.append('\\midrule')
    
    # Data rows
    prev_noise = None
    for noise in noise_values:
        # Add separator between different noise values
        if prev_noise is not None and noise != prev_noise:
            lines.append('\\midrule')
        prev_noise = noise
        
        for mnr in min_num_rows_values:
            row_parts = [str(int(noise)), str(int(mnr))]
            
            for nrows in nrows_values:
                value = table_data[noise][mnr][nrows]
                if value is None:
                    cell = '---'
                else:
                    formatted_val = f'{value:.2f}'
                    # Bold if >= threshold
                    if value >= thresh:
                        cell = f'\\textbf{{{formatted_val}}}'
                    else:
                        cell = formatted_val
                row_parts.append(cell)
            
            lines.append(' & '.join(row_parts) + ' \\\\')
    
    # Table closing
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append(f'\\caption{{Maximum measure values by noise, minimum rows threshold, and number of rows ({note}). Values $\\geq {thresh}$ are shown in bold.}}')
    lines.append(f'\\label{{tab:noise_min_num_rows_{note}}}')
    lines.append('\\end{table}')
    
    return '\n'.join(lines)

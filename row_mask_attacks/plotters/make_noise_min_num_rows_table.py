import pandas as pd
from pathlib import Path
from typing import Optional


def make_noise_min_num_rows_table(df: pd.DataFrame, nqi, note: str, thresh: float = 0.9):
    """Generate tables showing accuracy by noise, supp_thresh, and nrows.
    
    Creates both a text table and a LaTeX table showing maximum measure
    for each combination of noise, supp_thresh, and nrows.
    
    Args:
        df: DataFrame with columns 'noise', 'supp_thresh', 'nrows', and 'measure'
        nqi: Number of QI columns
        note: String to include in output filenames, labels, and captions
        thresh: Threshold for bolding values in LaTeX (default: 0.9)
    """
    
    # Create output directory
    output_dir = Path('./results/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique sorted values
    noise_values = sorted(df['noise'].unique())
    supp_thresh_values = sorted(df['supp_thresh'].unique())
    nrows_values = sorted(df['nrows'].unique())
    
    # Build the data structure: dict[noise][supp_thresh][nrows] = max
    table_data = {}
    for noise in noise_values:
        table_data[noise] = {}
        for stv in supp_thresh_values:
            table_data[noise][stv] = {}
            for nrows in nrows_values:
                # Filter data
                subset = df[(df['noise'] == noise) & 
                           (df['supp_thresh'] == stv) & 
                           (df['nrows'] == nrows)]

                if len(subset) > 1:
                    raise ValueError(
                        "Expected exactly one row for noise/supp_thresh/nrows "
                        f"but found {len(subset)} rows for noise={noise}, "
                        f"supp_thresh={stv}, nrows={nrows}"
                    )

                if len(subset) == 1 and 'measure' in subset.columns:
                    measure = subset['measure'].iloc[0]
                    table_data[noise][stv][nrows] = measure
                else:
                    table_data[noise][stv][nrows] = None
    
    # Generate text table
    text_output = _generate_text_table(table_data, noise_values, supp_thresh_values, nrows_values)
    text_file = output_dir / f'noise_supp_thresh_{note}.txt'
    with open(text_file, 'w') as f:
        f.write(text_output)
    print(f"Saved: {text_file}")
    
    # Generate LaTeX table
    latex_output = _generate_latex_table(table_data, noise_values, supp_thresh_values, nrows_values, note, thresh, nqi)
    latex_file = output_dir / f'noise_supp_thresh_{note}.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_output)
    print(f"Saved: {latex_file}")


def _generate_text_table(table_data: dict, noise_values: list, supp_thresh_values: list, nrows_values: list) -> str:
    """Generate a fixed-width text table."""
    
    lines = []
    
    # Header row
    header_parts = ['$e$', '$\tau$']
    for nrows in nrows_values:
        header_parts.append(f'{nrows}')
    
    # Calculate column widths - more compact
    col_widths = [6, 8]  # noise and supp_thresh
    data_col_width = 6  # for max value only (e.g., "0.00")
    col_widths.extend([data_col_width] * len(nrows_values))

    # Spanning header for nrows
    span_width = sum(col_widths[2:]) + max(len(col_widths[2:]) - 1, 0)
    span_text = "Number of rows"
    span_line = (
        " ".ljust(col_widths[0]) + " " +
        " ".ljust(col_widths[1]) + " " +
        span_text.center(span_width)
    )
    lines.append(span_line.rstrip())

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
        for stv in supp_thresh_values:
            row_parts = [str(int(noise)), str(int(stv))]
            
            for nrows in nrows_values:
                value = table_data[noise][stv][nrows]
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


def _generate_latex_table(table_data: dict, noise_values: list, supp_thresh_values: list, 
                          nrows_values: list, note: str, thresh: float, nqi: int) -> str:
    """Generate a LaTeX table using booktabs."""
    
    lines = []
    
    # Table preamble
    num_cols = 2 + len(nrows_values)
    col_spec = 'rr' + 'r' * len(nrows_values)
    
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    
    # Header row
    lines.append(f'\\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{{len(nrows_values)}}}{{c}}{{Number of rows}} \\\\')
    header_parts = ['$e$', '$\\tau$']
    for nrows in nrows_values:
        header_parts.append(f'${nrows}$')
    lines.append(' & '.join(header_parts) + ' \\\\')
    lines.append('\\midrule')
    
    # Data rows
    prev_noise = None
    for noise in noise_values:
        # Add separator between different noise values
        if prev_noise is not None and noise != prev_noise:
            lines.append('\\midrule')
        prev_noise = noise
        
        for stv in supp_thresh_values:
            row_parts = [str(int(noise)), str(int(stv))]
            
            for nrows in nrows_values:
                value = table_data[noise][stv][nrows]
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
    lines.append(f'\\caption{{Average accuracy by noise $e$, suppression threshold $\\tau$, and number of rows $R$, where the number of QI columns is $C={nqi}$. Accuracy values $\\geq {thresh}$ are shown in bold.}}')
    lines.append(f'\\label{{tab:noise_supp_thresh_{note}}}')
    lines.append('\\end{table}')
    
    return '\n'.join(lines)

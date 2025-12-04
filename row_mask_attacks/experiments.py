# Define parameter ranges
experiments = [
    {   # Pure Dinur-style (individual selection of rows, binary target)
        'dont_run': True,
        'experiment_group': 'pure_dinur_basics',
        'solve_type': 'pure_row',
        'seed': [0],
        'nrows': [100, 200],
        'mask_size': [20, 30, 50],
        'nunique': [2],
        'noise': [0, 2, 4, 8, 16],
        'nqi': [0],
        'min_num_rows': [5],
        'vals_per_qi': [0],
    },
    {   # Aggregated Dinur-style (aggregate selection of row IDs, binary target)
        'dont_run': True,
        'experiment_group': 'agg_dinur_basics',
        'solve_type': 'agg_row',
        'seed': [0],
        'nrows': [100, 200],
        'mask_size': [0],       # not used
        'nunique': [2],
        'noise': [0, 2, 4, 8, 16],
        'nqi': [3, 5, 7, 9, 11],
        'min_num_rows': [5],
        'vals_per_qi': [0],      # auto-select
    },
    {   # Aggregated Dinur-style, explore vals_per_qi, nunique
        'dont_run': True,
        'experiment_group': 'agg_dinur_explore_vals_per_qi_nunique',
        'solve_type': 'agg_row',
        'seed': [0],
        'nrows': [200],
        'mask_size': [0],       # not used
        'nunique': [2,4,8],
        'noise': [0, 2, 4, 8, 16],
        'nqi': [11],
        'min_num_rows': [5],
        'vals_per_qi': [2,3,4,5,6,7,8,9],      # auto-select
    },
    {   # Pure Dinur-style, test effect of nunique
        'dont_run': True,
        'experiment_group': 'pure_dinur_nunique',
        'solve_type': 'pure_row',
        'seed': [0],
        'nrows': [200],
        'mask_size': [20],
        'nunique': [2, 4, 8],
        'noise': [4],
        'nqi': [0],
        'min_num_rows': [5],
        'vals_per_qi': [0],      # auto-select
    },
    {   # Pure Dinur-style, test effect of min_num_rows
        'dont_run': True,
        'experiment_group': 'pure_dinur_min_num_rows',
        'solve_type': 'pure_row',
        'seed': [0],
        'nrows': [200],
        'mask_size': [20],
        'nunique': [2],
        'noise': [4],
        'nqi': [0],
        'min_num_rows': [5, 10, 15],
        'vals_per_qi': [0],      # auto-select
    },
    {   # Aggregated Dinur-style, test effect of nunique
        'dont_run': True,
        'experiment_group': 'agg_dinur_nunique',
        'solve_type': 'agg_row',
        'seed': [0],
        'nrows': [200],
        'mask_size': [0],       # not used
        'nunique': [2, 4, 8],
        'noise': [4],
        'nqi': [11],
        'min_num_rows': [5],
        'vals_per_qi': [0],      # auto-select
    },
    {   # Aggregated Dinur-style, test effect of min_num_rows
        'dont_run': True,
        'experiment_group': 'agg_dinur_min_num_rows',
        'solve_type': 'agg_row',
        'seed': [0],
        'nrows': [200],
        'mask_size': [0],       # not used
        'nunique': [2],
        'noise': [4],
        'nqi': [11],
        'min_num_rows': [5, 10, 15],
        'vals_per_qi': [0],      # auto-select
    },
    {   # Aggregated Dinur-style, test effect of vals_per_qi
        'dont_run': True,
        'experiment_group': 'agg_dinur_vals_per_qi',
        'solve_type': 'agg_row',
        'seed': [0],
        'nrows': [200],
        'mask_size': [0],       # not used
        'nunique': [2],
        'noise': [4],
        'nqi': [11],
        'min_num_rows': [5],
        'vals_per_qi': [0, 5, 10],
    },
    {   # Aggregated Dinur-style, explore vals_per_qi, nrows
        'dont_run': True,
        'experiment_group': 'agg_dinur_explore_vals_per_qi_nrows',
        'solve_type': 'agg_row',
        'seed': [0],
        'nrows': [100,200],
        'mask_size': [0],       # not used
        'nunique': [2],
        'noise': [0, 2, 4, 8, 16],
        'nqi': [11],
        'min_num_rows': [5],
        'vals_per_qi': [2,3,4,5,6,7,8,9],      # auto-select
    },
    {   # Aggregated Dinur-style, explore vals_per_qi, min_num_rows
        'dont_run': True,
        'experiment_group': 'agg_dinur_explore_vals_per_qi_min_num_rows',
        'solve_type': 'agg_row',
        'seed': [0],
        'nrows': [200],
        'mask_size': [0],       # not used
        'nunique': [2],
        'noise': [0, 2, 4, 8, 16],
        'nqi': [11],
        'min_num_rows': [5, 10, 15],
        'vals_per_qi': [2,3,4,5,6,7,8,9],      # auto-select
    },
    {   # Pure Dinur-style, test effect of nrows
        'dont_run': True,
        'experiment_group': 'pure_dinur_nrows',
        'solve_type': 'pure_row',
        'seed': [0],
        'nrows': [100, 200, 300],
        'mask_size': [20],
        'nunique': [2],
        'noise': [4],
        'nqi': [0],
        'min_num_rows': [5],
        'vals_per_qi': [0],      # auto-select
    },
    {   # Pure Dinur-style, test effect of nrows, better granularity
        'dont_run': True,
        'experiment_group': 'pure_dinur_nrows_granular',
        'solve_type': 'pure_row',
        'seed': [0],
        'nrows': [25, 50, 75, 100, 125, 150, 175, 200],
        'mask_size': [10],
        'nunique': [2],
        'noise': [4],
        'nqi': [0],
        'min_num_rows': [5],
        'vals_per_qi': [0],      # auto-select
    },
    {   # Agg Dinur-style, test effect of nrows
        'dont_run': True,
        'experiment_group': 'agg_dinur_nrows',
        'solve_type': 'agg_row',
        'seed': [0],
        'nrows': [25, 50, 75, 100, 125, 150, 175, 200],
        'mask_size': [0],
        'nunique': [2],
        'noise': [4],
        'nqi': [11],
        'min_num_rows': [5],
        'vals_per_qi': [2],
    },
    {   # Agg Dinur-style, x=noise, y=nqi, lines=nrows
        'dont_run': False,
        'experiment_group': 'agg_dinur_x_noise_y_nqi_lines_nrows',
        'solve_type': 'agg_row',
        'seed': [0,1,2,3,4,5,6,7,8,9],
        'nrows': [50,75,100,125],
        'mask_size': [0],
        'nunique': [2],
        'noise': [2,4,6,8,10,12,14,16],
        'nqi': [3,5,7,9,11],
        'min_num_rows': [5],
        'vals_per_qi': [0],   # auto-select
    },
]

def read_experiments():
    return experiments

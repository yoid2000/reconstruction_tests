new_experiments = [
    {   # Agg known best
        'dont_run': False,
        'used_in_paper': True,
        'experiment_group': 'agg_known_best',
        'solve_type': 'agg_known',
        'seed': [1],
        'nrows': [50],
        'mask_size': [0],
        'nunique': [2],
        'noise': [1],
        'nqi': [6],
        'min_num_rows': [2],
        'vals_per_qi': [0],   # auto-select
        'known_qi_fraction': [0.0, 0.25, 0.5, 0.75],
    },
]

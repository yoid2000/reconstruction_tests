import copy

DEFAULT_USE_OBJECTIVE = True
DEFAULT_TIME_LIMIT_SECONDS = (3 * 24 * 60 * 60)  # 3 days in seconds
DEFAULT_SLACK_LIMIT_MULTIPLE = 2
DEFAULT_SLACK_LIMIT_MIN = 10
DEFAULT_MAX_NUM_CONTINGENCY_TABLES = 100000

# Define parameter ranges
experiments = [
    {   # Agg Dinur-style, no noise suppression
        'dont_run': True,
        'used_in_paper': True,
        'experiment_group': 'agg_dinur_no_noise_supp',
        'solve_type': 'agg_row',
        'nrows': [150],
        'nqi': [11],
        'nunique': [2],
        'vals_per_qi': [2],
        'mask_size': [0],
        'noise': [0],
        'max_num_contingency_tables': [20, 40, 70, 100],
        'min_num_rows': [1],
        'corr_strength': [0.0],
        'known_qi_fraction': [1.0],
        'max_qi': [1000],
        'max_samples': [20000],
        'use_objective': [DEFAULT_USE_OBJECTIVE],
        'time_limit_seconds': [DEFAULT_TIME_LIMIT_SECONDS],
        'slack_limit_multiple': [DEFAULT_SLACK_LIMIT_MULTIPLE],
        'slack_limit_min': [DEFAULT_SLACK_LIMIT_MIN],
        'path_to_dataset': [''],
        'target_column': [''],
    },
    {   # Agg Dinur-style, a little noise suppression
        'dont_run': True,
        'used_in_paper': True,
        'experiment_group': 'agg_dinur_a_little_noise_supp',
        'solve_type': 'agg_row',
        'nrows': [150],
        'nqi': [11],
        'nunique': [2],
        'vals_per_qi': [2],
        'mask_size': [0],
        'noise': [0, 1, 2, 4, 5, 7, 10],
        'max_num_contingency_tables': [50, 100, 150, 200],
        'min_num_rows': [1, 2, 3, 4, 5, 7, 10],
        'corr_strength': [0.0],
        'known_qi_fraction': [1.0],
        'max_qi': [1000],
        'max_samples': [20000],
        'use_objective': [DEFAULT_USE_OBJECTIVE],
        'time_limit_seconds': [DEFAULT_TIME_LIMIT_SECONDS],
        'slack_limit_multiple': [DEFAULT_SLACK_LIMIT_MULTIPLE],
        'slack_limit_min': [DEFAULT_SLACK_LIMIT_MIN],
        'path_to_dataset': [''],
        'target_column': [''],
    },
    {   # Agg Dinur-style, noise, no suppression
        'dont_run': True,
        'used_in_paper': True,
        'experiment_group': 'agg_dinur_noise_no_supp',
        'solve_type': 'agg_row',
        'nrows': [150],
        'nqi': [11],
        'nunique': [2],
        'vals_per_qi': [2],
        'mask_size': [0],
        'noise': [0, 1, 2, 4, 5, 7, 10, 15, 20],
        'max_num_contingency_tables': [50, 100, 150, 200],
        'min_num_rows': [1],
        'corr_strength': [0.0],
        'known_qi_fraction': [1.0],
        'max_qi': [1000],
        'max_samples': [20000],
        'use_objective': [DEFAULT_USE_OBJECTIVE],
        'time_limit_seconds': [DEFAULT_TIME_LIMIT_SECONDS],
        'slack_limit_multiple': [DEFAULT_SLACK_LIMIT_MULTIPLE],
        'slack_limit_min': [DEFAULT_SLACK_LIMIT_MIN],
        'path_to_dataset': [''],
        'target_column': [''],
    },
]

def read_experiments(used_only_in_paper: bool = True) -> list[dict]:
    adjusted_experiments = copy.deepcopy(experiments)
    if used_only_in_paper:
        adjusted_experiments = [exp for exp in adjusted_experiments if exp.get('used_in_paper', False)]
    for exp in adjusted_experiments:
        if 'min_num_rows' in exp:
            values = exp['min_num_rows']
            if not isinstance(values, list):
                values = [values]
            exp['supp_thresh'] = [value - 1 for value in values]
    for exp in adjusted_experiments:
        if 'corr_strength' not in exp:
            exp['corr_strength'] = [0.0]
        elif not isinstance(exp['corr_strength'], list):
            exp['corr_strength'] = [exp['corr_strength']]
        if 'path_to_dataset' not in exp:
            exp['path_to_dataset'] = ['']
        elif not isinstance(exp['path_to_dataset'], list):
            exp['path_to_dataset'] = [exp['path_to_dataset']]
        exp['path_to_dataset'] = ['' if value is None else value for value in exp['path_to_dataset']]
        if 'target_column' not in exp:
            exp['target_column'] = ['']
        elif not isinstance(exp['target_column'], list):
            exp['target_column'] = [exp['target_column']]
        exp['target_column'] = ['' if value is None else value for value in exp['target_column']]
        if 'max_num_contingency_tables' not in exp:
            exp['max_num_contingency_tables'] = [DEFAULT_MAX_NUM_CONTINGENCY_TABLES]
        elif not isinstance(exp['max_num_contingency_tables'], list):
            exp['max_num_contingency_tables'] = [exp['max_num_contingency_tables']]
    return adjusted_experiments


## run_row_mask_attack.py

`run_row_mask_attack.py` runs a reconstruction attack whereby the attacker has the ability to select individual rows and obtain a noisy count. 

When run with no command line arguments, it lists the attacks (jobs) and their associated parameters, and generates the file run.slurm, which can be used to run all the attacks in a SLURM cluster.

Note that the timeout for the slurm jobs is determined by `max_time_minutes`. The purpose for this is to allow us to set the timeout to a low value (say 30 minutes) so that quick jobs can complete, even though long jobs will time out. The work cycle is to then do `gather.py` to create result.parquet, increase `max_time_minutes`, and then run `run_row_mask_attack_py` again to create slurm jobs that only include the longer jobs.

When run with -h or --help, it lists the command line arguments.

When run with a single integer, it runs the attack associated with the index defined by the integer.

Otherwise, it runs an attack with the following parameters:

```
  --nrows NROWS         Number of rows in the dataframe being attacked
  --mask_fraction MASK_FRACTION
                        The fraction of rows that are in the attackers list of rows (0.0-1.0)
  --nunique NUNIQUE     Number of unique values in the column being predicted (reconstructed)
  --noise NOISE         An integer indicating the amount of noise (which is uniformly chosen from the 
                        range -NOISE to +NOISE)
```

The attack result is stored under `results/` in a json file

## gather.py

`gather.py` reads in the json files produced by `run_row_mask_attack.py`, reads the data, and places the result in `results/result.parquet`.

If run with --force, then `result.parquet` is created from scratch from the json files. Else, `result.parquet` is appended with the data from the json files not already in `result.parquet`. 

## analyze.py

Reads `result.parquet`and generates a variety of plots and tables.

If the `--more_seeds` command flag is set, then it determines which experiments require more seeds to reach the target 95% confidence interval, and writes `more_seeds_experiments.py`.  This file is then used by `run_row_mask_attack.py` to run the additional experiments.

## experiments.py

Contains the configurations of experimental parameters. Used both by `run_row_mask_attack.py` to run experiments (only those with `dont_run = False`) and `analyze.py` to analyze experiments.

## Workflow

The general workflow for running experiments is then as follows:

1. Add a set of experimental parameters to experiments.py. We recommend configuring 5 distinct seeds.
2. Set `max_time_minutes` in `run_row_mask_attack.py` to the lowest value that is likely to cause most experiments to finish before SLURM ends them.
3. Run `run_row_mask_attack.py` to generate the appropriate `run.slurm` file.
4. Do `sbatch run.slurm` on the SLURM cluster.
5. After finished, run `gather.py` to update `result.parquet`.
6. If not all experiments completed, increase `max_time_minutes` and run from step 3 again. Note the reason we don't set `max_time_minutes` to a high value at first is because SLURM won't schedule as many parallel jobs.
7. Repeat 3-6 until all experiments are run, or until the maximum `max_time_minutes` allowed by the SLURM configuration is run.
8. Run `analyze.py --more_seeds`. If this causes additional experiments to be configured in `more_seeds_experiments.py`, then do the 3-6 loop again until these are completed.



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

## analyze.py


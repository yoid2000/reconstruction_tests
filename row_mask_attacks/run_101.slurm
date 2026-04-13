#!/bin/bash
#SBATCH --job-name=101_recon_test
#SBATCH --output=/INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/slurm_out/job_101_%A_%a.out
#SBATCH --time=10080
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-12
arrayNum="${SLURM_ARRAY_TASK_ID}"
source /INS/syndiffix/work/paul/github/reconstruction_tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/run_row_mask_attack.py --slurm_run 101 --job_num $arrayNum

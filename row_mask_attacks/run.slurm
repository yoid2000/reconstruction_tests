#!/bin/bash
#SBATCH --job-name=recon_test
#SBATCH --output=/INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/slurm_out/out.%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-74
arrayNum="${SLURM_ARRAY_TASK_ID}"
source /INS/syndiffix/work/paul/github/reconstruction_tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/run_row_mask_attack.py $arrayNum

#!/bin/bash
#SBATCH --job-name=fake_test
#SBATCH --output=/INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/fake_out/job_%A_%a.out
#SBATCH --time=1:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-461
arrayNum="${SLURM_ARRAY_TASK_ID}"
source /INS/syndiffix/work/paul/github/reconstruction_tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/fake_row_mask_attack.py $arrayNum

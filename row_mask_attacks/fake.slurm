#!/bin/bash
#SBATCH --job-name=fake_test
#SBATCH --output=/dev/null
#SBATCH --time=30:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-76135
arrayNum="${SLURM_ARRAY_TASK_ID}"
source /INS/syndiffix/work/paul/github/reconstruction_tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/fake_row_mask_attack.py $arrayNum

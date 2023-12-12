#!/bin/bash
#SBATCH --account=def-kdhkdh
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --array=1-50
#SBATCH --job-name=workflow
#SBATCH --output=output/%x-%A-%a.out
#SBATCH --mail-user=kevinkw.zhang@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.8.10
module load scipy-stack
source ../lib/python/3.8.10/ENV/bin/activate

python workflow.py $SLURM_ARRAY_TASK_ID $1 $2
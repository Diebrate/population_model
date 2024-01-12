#!/bin/bash
#SBATCH --account=def-kdhkdh
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=eval
#SBATCH --output=output/%x-%j.out
#SBATCH --mail-user=kevinkw.zhang@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.8.10
module load scipy-stack
source ../lib/python/3.8.10/ENV/bin/activate

python workflow_eval.py
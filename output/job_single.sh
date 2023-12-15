#!/bin/bash
#SBATCH --account=def-kdhkdh
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=single
#SBATCH --output=output/%x-%j.out

module load python/3.8.10
module load scipy-stack
source ../lib/python/3.8.10/ENV/bin/activate

python workflow_single.py $1 $2 $3 $4 $5 $6
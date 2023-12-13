#!/bin/bash
#SBATCH --account=def-kdhkdh
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=valid
#SBATCH --output=output/%x-%j.out
#SBATCH --mail-user=kevinkw.zhang@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.8.10
module load scipy-stack
source ../lib/python/3.8.10/ENV/bin/activate

python workflow_valid.py
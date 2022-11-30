#!/bin/bash

#SBATCH --job-name=postproc
#SBATCH --account=e3sm
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:10:00

# Setup My Environment
source /gpfs/fs1/soft/chrysalis/manual/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detcli
python3 post_proc.py
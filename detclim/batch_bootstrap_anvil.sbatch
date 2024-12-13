#!/bin/bash

#SBATCH --job-name=bootstrap
#SBATCH --account=condo
#SBATCH --partition=acme-small
#SBATCH --nodes=1
#SBATCH --time=01:00:00

# Setup My Environment
source /soft/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detcli

python3 bootstrap_ks_test.py \
    --base ${CASEA} \
    --test ${CASEB} \
    --runlen ${RUNLEN} \
    --iter 500 \
    --nodes 1 \
    --rolling 12
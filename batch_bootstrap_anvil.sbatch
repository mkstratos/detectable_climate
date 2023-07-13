#!/bin/bash

#SBATCH --job-name=bootstrap
#SBATCH --account=condo
#SBATCH --partition=acme-small
#SBATCH --nodes=1
#SBATCH --time=01:00:00

# Setup My Environment
source /soft/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detcli

base=pl_1e-10
test=gworo-1yr-10pct

/usr/bin/time -f "TIME: %es\n MEM: %M KB\n CPU: %P" \
    python3 bootstrap_ks_test.py \
    --base $base \
    --test $test \
    --iter 500 \
    --nodes 1 \
    --rolling 12
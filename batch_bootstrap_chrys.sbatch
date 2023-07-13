#!/bin/bash

#SBATCH --job-name=bootstrap
#SBATCH --account=condo
#SBATCH --partition=acme-small
#SBATCH --nodes=1
#SBATCH --time=01:00:00

# Setup My Environment
source /soft/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detcli
/usr/bin/time -f "TIME: %es\n MEM: %M KB\n CPU: %P" \
    python3 boostrap_ks_test.py --base ctl --test 2p5pct --iter 500

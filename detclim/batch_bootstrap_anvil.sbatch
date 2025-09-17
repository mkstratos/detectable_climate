#!/bin/bash

#SBATCH --job-name=bootstrap
#SBATCH --account=condo
#SBATCH --partition=acme-small
#SBATCH --nodes=1
#SBATCH --time=01:00:00

# Setup My Environment
source /soft/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detcli

RUNLEN=1year
CASEA=ctl
CASEB=clubb_c1-5p0pct

for esize in 10 20 30 40 50 60
do
    python3 bootstrap_ks_test.py \
        --base ${CASEA} \
        --test ${CASEB} \
        --runlen ${RUNLEN} \
        --iter 1000 \
        --nodes 1 \
        --rolling 12 \
        --esize ${esize}
done

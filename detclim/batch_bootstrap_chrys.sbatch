#!/bin/bash

#SBATCH --job-name=bootstrap
#SBATCH --account=e3sm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=02:30:00

# Setup My Environment
source /soft/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detcli
RUNLEN=1year
CASEA=ctl
for CASEB in ctl clubb_c1-1p0pct clubb_c1-3p0pct clubb_c1-5p0pct clubb_c1-10p0pct
do
    python3 bootstrap_ks_test.py --base ${CASEA} --test ${CASEB} --runlen ${RUNLEN} --iter 1000 --nodes 1 --rolling 12
done

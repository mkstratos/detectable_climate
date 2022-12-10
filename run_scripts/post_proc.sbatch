#!/bin/bash

#SBATCH --job-name=postproc
#SBATCH --account=e3sm
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:55:00

# Setup My Environment
source /gpfs/fs1/soft/chrysalis/manual/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detcli
CASE=20221205.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00201_n0030_b_c
BASE=20221205.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00201_n0030
# CASE=20221130.F2010.ne4_oQU240.dtcl_control_n0030_b_d
# BASE=20221130.F2010.ne4_oQU240.dtcl_control_n0030
python3 post_proc.py --case ${CASE} --base ${BASE}

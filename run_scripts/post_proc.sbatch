#!/bin/bash

#SBATCH --job-name=postproc
#SBATCH --account=e3sm
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=01:00:00

# Setup My Environment
source /gpfs/fs1/soft/chrysalis/manual/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detcli
CASE=20230123.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00205_n0120
# BASE=20221201.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0022_n0030

python3 post_proc.py --case ${CASE}  # --base ${BASE}

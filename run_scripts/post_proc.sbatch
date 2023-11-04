#!/bin/bash

#SBATCH --job-name=postproc
#SBATCH --account=e3sm
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --time=02:00:00

# Setup My Environment
source /gpfs/fs1/soft/chrysalis/manual/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detcli
# CASE=20230124.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00205_n0120
# CASE=20230126.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00202_n0120
# CASE=20230124.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00200_n0120
# BASE=20221201.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0022_n0030
# CASE=20230321.F2010.ne4_oQU240.dtcl_pertlim_1e-10_n0120
# CASE=20230322.F2010.ne4_oQU240.dtcl_pertlim_1e-14_n0120
# CASE=20230327.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00200_n0120
# CASE=20230403.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00200_n0120
# CASE=20230422.F2010.ne4_oQU240.dtcl_clubb_c1_2p4240_n0120
# CASE=20230515.F2010.ne4_oQU240.dtcl_effgw_oro_0p4125_n0120
# CASE=20230517.F2010.ne4_oQU240.dtcl_effgw_oro_0p37875_n0120

# CASE=20230615.F2010.ne4_oQU240.dtcl_effwg_oro_0p376875_n0120
# CASE=20230613.F2010.ne4_oQU240.dtcl_effgw_oro_0p4125_n0120
echo POSTPROCESSING ${CASE} on ${MACH}
# /usr/bin/time -f "TIME: %es\n MEM: %M KB\n CPU: %P" \
python3 post_proc.py --case ${CASE} --mach ${MACH} && \
python3 post_proc_pertlim_test.py --case ${CASE} --mach ${MACH}
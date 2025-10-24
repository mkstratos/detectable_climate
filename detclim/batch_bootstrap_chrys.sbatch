#!/bin/bash

#SBATCH --job-name=bootstrap
#SBATCH --account=e3sm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=03:00:00

# Setup My Environment
source /soft/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detclim_new
RUNLEN=1year
CASEA=ctl
NITER=1000

# Call this script with (e.g. ) GRP=clubb sbatch batch_bootstrap_chrys.sbatch
# to perform all bootstraps for the CLUBB parameter sweep
if [ ${GRP} = "clubb" ]; then
    CASEA=ctl
    CASES=(ctl clubb_c1-1p0pct clubb_c1-3p0pct clubb_c1-5p0pct clubb_c1-10p0pct opt-O1 fastmath)
else if [ ${GRP} = "effgw" ]; then
    CASEA=ctl
    CASES=(effgw_oro-1p0pct effgw_oro-5p0pct effgw_oro-10p0pct effgw_oro-20p0pct effgw_oro-30p0pct effgw_oro-40p0pct effgw_oro-50p0pct)
else
    CASEA=ctl-miller
    CASES=(ctl-miller zmconv_c0_ocn-0p5pct zmconv_c0_ocn-1p0pct zmconv_c0_ocn-3p0pct zmconv_c0_ocn-5p0pct)
fi
fi

for tsize in 15 20 25 30 35 40 45 50 55 60
do
    for CASEB in ${CASES[@]}
    do
        # Do the control test for finding threshold
        python3 bootstrap_tests.py \
            --base ${CASEB} \
            --test ${CASEB} \
            --runlen ${RUNLEN} \
            --iter ${NITER} \
            --nodes 1 \
            --rolling 12 \
            --esize ${tsize}

        # Do the control test for finding false discovery rate
        python3 bootstrap_tests.py \
            --base ${CASEB} \
            --test ${CASEB} \
            --runlen ${RUNLEN} \
            --iter ${NITER} \
            --nodes 1 \
            --rolling 12 \
            --esize ${tsize} \
            --ctl
        if [ ${CASEA} != ${CASEB} ]; then
            # IF A and B are the same, we've already done this, so skip it otherwise...
            # do the test for comparison to control
            python3 bootstrap_tests.py \
                --base ${CASEA} \
                --test ${CASEB} \
                --runlen ${RUNLEN} \
                --iter ${NITER} \
                --nodes 1 \
                --rolling 12 \
                --esize ${tsize}
        fi
    done
done

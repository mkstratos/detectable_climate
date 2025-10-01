#!/bin/bash

#SBATCH --job-name=bootstrap
#SBATCH --account=e3sm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=04:30:00

# Setup My Environment
source /soft/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate detclim_new
RUNLEN=1year
CASEA=ctl
NITER=1000

# Control self-comparisons
for CASEA in ctl ctl-miller clubb_c1-1p0pct clubb_c1-3p0pct clubb_c1-5p0pct clubb_c1-10p0pct \
    effgw_oro-1p0pct effgw_oro-5p0pct effgw_oro-10p0pct effgw_oro-20p0pct effgw_oro-30p0pct effgw_oro-40p0pct effgw_oro-50p0pct \
    opt-O1 fastmath zmconv_c0_ocn-0p5pct zmconv_c0_ocn-1p0pct zmconv_c0_ocn-3p0pct zmconv_c0_ocn-5p0pct
do
    python3 bootstrap_tests_new.py \
        --base ${CASEA} \
        --test ${CASEA} \
        --runlen ${RUNLEN} \
        --iter ${NITER} \
        --nodes 1 \
        --rolling 12 \
        --ctl
done

# Control self-comparisons (non-threshold finding)
for CASEA in ctl ctl-miller clubb_c1-1p0pct clubb_c1-3p0pct clubb_c1-5p0pct clubb_c1-10p0pct \
    effgw_oro-1p0pct effgw_oro-5p0pct effgw_oro-10p0pct effgw_oro-20p0pct effgw_oro-30p0pct effgw_oro-40p0pct effgw_oro-50p0pct \
    opt-O1 fastmath zmconv_c0_ocn-0p5pct zmconv_c0_ocn-1p0pct zmconv_c0_ocn-3p0pct zmconv_c0_ocn-5p0pct
do
    python3 bootstrap_tests_new.py \
        --base ${CASEA} \
        --test ${CASEA} \
        --runlen ${RUNLEN} \
        --iter ${NITER} \
        --nodes 1 \
        --rolling 12
done

# Compare to control
CASEA=ctl
for CASEB in ctl-miller clubb_c1-1p0pct clubb_c1-3p0pct clubb_c1-5p0pct clubb_c1-10p0pct \
    effgw_oro-1p0pct effgw_oro-5p0pct effgw_oro-10p0pct effgw_oro-20p0pct effgw_oro-30p0pct effgw_oro-40p0pct effgw_oro-50p0pct \
    opt-O1 fastmath
do
    python3 bootstrap_tests_new.py \
        --base ${CASEA} \
        --test ${CASEB} \
        --runlen ${RUNLEN} \
        --iter ${NITER} \
        --nodes 1 \
        --rolling 12
done

CASEA=ctl-miller
for CASEB in zmconv_c0_ocn-0p5pct zmconv_c0_ocn-1p0pct zmconv_c0_ocn-3p0pct zmconv_c0_ocn-5p0pct
do
    python3 bootstrap_tests_new.py \
        --base ${CASEA} \
        --test ${CASEB} \
        --runlen ${RUNLEN} \
        --iter ${NITER} \
        --nodes 1 \
        --rolling 12
done

# CTL compare 19 x 2 + 14 + 4

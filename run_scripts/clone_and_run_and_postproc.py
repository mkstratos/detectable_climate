#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clone a run, submit the case, postprocess, (todo: do the KS Test bootstrapping).
"""
import subprocess as sp
from pathlib import Path
import os
import datetime as dt
import json
import argparse

PARAM_DEFAULTS = {
    "effgw_oro": 0.375,
    "clubb_c1": 2.4,
    "zmconv_c0_lnd": 0.0020,
    "zmconv_c0_ocn": 0.0020,
}


def cl_args():
    """Parse command line args to set parameters.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--base", default="ctl", help="Short name of the base case.")
    parser.add_argument("--param_name", default="effgw_oro")
    parser.add_argument("--pct_change", default=1.0, type=float)
    return parser.parse_args()



def main(cl_args):
    """Setup runs."""
    mach = "chrys"
    model_branch = "maint-2.0"
    scratch_dir = Path("/lcrc/group/e3sm/ac.mkelleher/scratch")
    script_dir = Path(
        os.environ["HOME"], "baseline", model_branch, "E3SM", "cime", "scripts"
    )
    base_case = "20230321.F2010.ne4_oQU240.dtcl_pertlim_1e-10_n0120"

    run_len = "1year"
    param_name = cl_args.param_name
    param_default = PARAM_DEFAULTS[param_name]
    pct_change = cl_args.pct_change

    param_val = param_default * (1 + pct_change / 100)
    param_str = f"{param_val:.06f}".replace(".", "p")
    short_name = f"{param_name}-{pct_change}pct"

    # Here's the example call to create a clone
    # ~/baseline/maint-2.0/E3SM/cime/scripts/create_clone \
    #   --case 20230615.F2010.ne4_oQU240.dtcl_effwg_oro_0p376875_n0120 \
    #   --clone 20230321.F2010.ne4_oQU240.dtcl_pertlim_1e-10_n0120 \
    #   --keepexe \
    #   --cime-output-root $SCRATCH/chrys
    cset, res = base_case.split(".")[1:3]

    new_case = (
        f"{dt.datetime.now().strftime('%Y%m%d')}.{cset}.{res}."
        f"dtcl_{param_name}_{param_str}_n0120"
    )

    # Add this new case to the json database
    cases = json.load(open("../case_db.json", "r"))
    cases[run_len][short_name] = new_case
    print(f"ADDING {run_len}, {short_name} = \n\t{new_case} to case_db.json")
    with open("../case_db.json", "w") as _fo:
        _fo.write(json.dumps(cases))

    os.system(
        f"{Path(script_dir, 'create_clone')} "
        f"--case {new_case} "
        f"--clone {base_case} "
        f"--cime-output-root {Path(scratch_dir, mach)} "
        "--keepexe"
    )

    nl_files = Path(new_case).glob("user_nl_eam_*")
    for _nl_file in nl_files:
        with open(_nl_file, "a", encoding="utf-8") as _nlh:
            _nlh.write(f"{param_name} = {param_val:.08f}\n")

    print(f"SUBMITTING CASE FROM: {Path(new_case).resolve()}")
    submit_result = sp.check_output(
        ["./case.submit"],
        stderr=sp.STDOUT,
        shell=True,
        cwd=Path(new_case).resolve(),
    )
    print(submit_result.decode())

    job_id = int(
        [_line for _line in submit_result.decode().split("\n") if "job id" in _line][
            0
        ].split()[-1]
    )

    res = sp.check_output(
        f"CASE={new_case} sbatch --dependency=afterok:{job_id} post_proc.sbatch",
        shell=True,
        stderr=sp.STDOUT,
    )
    post_proc_job = int(res.decode().split()[-1])

    _ = sp.check_output(
        (
            f"CASEA=ctl CASEB={short_name} RUNLEN={run_len} "
            f"sbatch --dependency=afterok:{post_proc_job} batch_bootstrap_{mach}.sbatch"
        ),
        shell=True,
        cwd=Path("../").resolve(),
        stderr=sp.STDOUT,
    )


if __name__ == "__main__":
    main()

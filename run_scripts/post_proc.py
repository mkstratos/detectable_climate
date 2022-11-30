#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import subprocess as sp
from functools import partial
import multiprocessing as mp


NCPU = mp.cpu_count()


def print_cmd(cmd):
    """Print out a list as space separated string."""
    print(" ".join([str(_cmi) for _cmi in cmd]))


def nco_aavg(in_file, overwrite=False, debug_only=True):
    """Perform area average on a netCDF file using NCO."""
    _logfilename = "aavg.log"

    aavg_outfile = Path(
        *in_file.parts[:-1],
        f"{in_file.stem}_aavg{in_file.suffix}",
    )

    wgt_avg = [
        "ncwa",
        "--no_tmp_fl",
        "-O",
        "-a",
        "ncol,lev",
        "-w",
        "area",
        in_file,
        "-o",
        aavg_outfile,
    ]
    if (not overwrite and aavg_outfile.exists()) or "aavg" in in_file.name:
        print(f"{aavg_outfile} exists")

    elif not debug_only:
        sp.check_call(
            wgt_avg,
            stdout=open(_logfilename, "a"),
            stderr=open(_logfilename, "a"),
        )

    else:
        print_cmd(wgt_avg)



def main(overwrite=False, debug_only=True):
    """Post process an ensemble run."""
    scratch = Path("/lcrc/group/e3sm/ac.mkelleher/scratch/chrys/")
    case = "20221128.F2010.ne4_oQU240.dtcl_control"
    case_dir = Path(scratch, case, "run")
    ninst = 8

    case_files = sorted(case_dir.glob(f"{case}.eam_*.h0*.nc"))

    pool_size = min(NCPU, len(case_files))
    print(f"DO AREA AVG TO {len(case_files)} FILES WITH {pool_size} PROCESSES")
    print(f"    {case_files[0].name}")

    with mp.Pool(pool_size) as pool:
        _results = pool.map_async(
            partial(
                nco_aavg,
                overwrite=overwrite,
                debug_only=debug_only,
            ),
            case_files,
        )
        results = _results.get()
    print(f"{'#' * 20}COMPLETED{'#' * 20}")

if __name__ == "__main__":
    main(overwrite=True, debug_only=False)
    # main()
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

    print(f"\tSTART AVG: {in_file.name.split('.')[-2]}")
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
    print(f"\t  END AVG: {in_file.name.split('.')[-2]}")


def combine_files(ninst, file_dir):

    for i in range(1, ninst + 1):
        _files = sorted(Path(file_dir).glob(f"*eam_{i:04d}*aavg.nc"))
        # print(_files[0])
        out_file = _files[0].name.split(".")
        out_file = ".".join([*_files[0].name.split(".")[:-2], "nc"])
        out_dir = Path(file_dir, "combined")
        if not out_dir.exists():
            print(f"CREATING COMBO DIR: {out_dir}")
            out_dir.mkdir(parents=True)

        sp.call(["ncrcat", *_files, Path(out_dir, out_file)])


def main(overwrite=False, debug_only=True):
    """Post process an ensemble run."""
    serial = False
    scratch = Path("/lcrc/group/e3sm/ac.mkelleher/scratch/chrys/")
    # case = "20221128.F2010.ne4_oQU240.dtcl_control"
    case = "20221130.F2010.ne4_oQU240.dtcl_control_n0030"

    # case = "20221201.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0022_n0030"
    case_dir = Path(scratch, case, "run")
    ninst = 30

    case_files = sorted(case_dir.glob(f"{case}.eam_*.h0*.nc"))
    case_files = [_file for _file in case_files if "aavg" not in _file.name]

    if serial:
        for _file in case_files:
            nco_aavg(_file, overwrite=overwrite, debug_only=debug_only)
    else:
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
        # combine_files(ninst, case_dir)


if __name__ == "__main__":
    main(overwrite=True, debug_only=False)
    # main()

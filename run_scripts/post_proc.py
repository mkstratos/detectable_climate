#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import subprocess as sp
from functools import partial
import multiprocessing as mp
import argparse
import shutil

NCPU = mp.cpu_count()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--case", default=None, help="Name of the test case.")
    parser.add_argument(
        "--overwrite",
        "-O",
        default=False,
        action="store_true",
        help="Overwrite files. Default: False",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Do not run, only print commands to stdout. Default: False",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Combine --case with --base",
    )
    return parser.parse_args()


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


def combine_ensembles(scratch, base_case_name, new_case_name, debug=False):
    """Move cloned case files into another directory to combine outputs."""
    new_files = sorted(Path(scratch, new_case_name, "run").glob(f"*aavg.nc"))
    old_files = sorted(Path(scratch, base_case_name, "run").glob(f"*aavg.nc"))

    try:
        ninst_0 = int(
            [_ix for _ix in old_files[-1].name.split(".") if "eam" in _ix][0].split(
                "_"
            )[-1]
        )
    except ValueError:
        print("LAST OLD INSTANCE NOT FOUND")
        raise
    for new_file in new_files:
        inst = int(
            [_ix for _ix in new_file.stem.split(".") if "eam" in _ix][0].split("_")[-1]
        )
        file_date = new_file.stem.split(".")[-1]
        _newname = f"{base_case_name}.eam_{inst + ninst_0:04d}.h0.{file_date}.nc"

        if debug:
            print(f"{new_file.exists()}")
            print(f"copy {new_file} to\n     {Path(scratch, base_case_name, 'run', _newname)}")
        else:
            shutil.copy2(new_file, Path(scratch, base_case_name, "run", _newname))


def main(cl_args):
    """Post process an ensemble run."""
    debug_only = cl_args.debug
    overwrite = cl_args.overwrite
    case = cl_args.case
    assert case is not None, "SPECIFY CASE"

    serial = True

    scratch = Path("/lcrc/group/e3sm/ac.mkelleher/scratch/chrys/")
    # case = "20221128.F2010.ne4_oQU240.dtcl_control"
    # case = "20221130.F2010.ne4_oQU240.dtcl_control_n0030"
    # case = "20221205.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00201_n0030"
    # case = "20221206.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0030_n0030"

    # case = "20221201.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0022_n0030"
    case_dir = Path(scratch, case, "run")
    ninst = 30

    case_files = sorted(case_dir.glob(f"{case}.eam_*.h0*.nc"))
    case_files = [_file for _file in case_files if "aavg" not in _file.name]

    if serial:
        for _file in case_files:
            nco_aavg(_file, overwrite=overwrite, debug_only=debug_only)
        print(f"DONE: {_file}")
        _check = sp.check_output(["ncdump", "-v", "T", _file])
        print("\t" + " ".join(_check.decode().split("\n")[-4:-3]).strip())
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

    if cl_args.base is not None:
        combine_ensembles(scratch, cl_args.base, case, debug=debug_only)


if __name__ == "__main__":
    main(cl_args=parse_args())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import subprocess as sp
from functools import partial
import multiprocessing as mp
import argparse
import shutil
import os

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
    parser.add_argument(
        "--mach",
        default="chrys",
        help="Machine on which the run took place"
    )

    return parser.parse_args()


def print_cmd(cmd):
    """Print out a list as space separated string."""
    print(" ".join([str(_cmi) for _cmi in cmd]))


def combine_files(ninst, file_dir, file_s=None, file_e=None):

    for i in range(1, ninst + 1):
        _files = sorted(Path(file_dir).glob(f"*eam_{i:04d}*aavg.nc"))
        _files = _files[file_s:file_e]
        # print(_files[0])
        out_file = _files[0].name.split(".")
        out_file = ".".join([*_files[0].name.split(".")[:-2], "aavg", "nc"])
        out_dir = Path(file_dir, "combined")
        if not out_dir.exists():
            print(f"CREATING COMBO DIR: {out_dir}")
            out_dir.mkdir(parents=True)

        sp.call(["ncrcat", *_files, Path(out_dir, out_file)])


def move_files(case_dir):
    """Remove individual aavg files, move combined files to run dir."""
    combo_dir = Path(case_dir, "combined")
    if not combo_dir.exists():
        raise FileNotFoundError("Combined directory not found")
    indv_files = Path(case_dir).glob("*eam*aavg.nc")
    for _file in indv_files:
        os.remove(_file)
    os.system(f"mv {combo_dir.resolve()}/*.nc {case_dir.resolve()}")


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
            print(
                f"copy {new_file} to\n     {Path(scratch, base_case_name, 'run', _newname)}"
            )
        else:
            shutil.copy2(new_file, Path(scratch, base_case_name, "run", _newname))


def remove_files(case_name, case_dir):
    """Remove un-needed mpassi, mpaso, elm, cpl, restart files."""
    globs_to_remove = [
        f"{case_name}*.mpassi*.nc",
        f"{case_name}*.mpasso*.nc",
        f"{case_name}*.elm*.nc",
        f"{case_name}*.mosart*.nc",
        f"{case_name}*.r*.nc",
    ]
    for _glob in globs_to_remove:
        print(f"REMOVING {len(sorted(case_dir.glob(_glob)))} files matching {_glob}")
        os.system(f"rm -f {case_dir}/{_glob}")


def main(cl_args):
    """Post process an ensemble run."""
    debug_only = cl_args.debug
    overwrite = cl_args.overwrite
    case = cl_args.case
    assert case is not None, "SPECIFY CASE"

    serial = False

    scratch = Path("/lcrc/group/e3sm/ac.mkelleher/scratch/", cl_args.mach)
    # case = "20221128.F2010.ne4_oQU240.dtcl_control"
    # case = "20221130.F2010.ne4_oQU240.dtcl_control_n0030"
    # case = "20221205.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00201_n0030"
    # case = "20221206.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0030_n0030"
    # case = "20221201.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0022_n0030"

    case_dir = Path(scratch, case, "run")
    print(f"COMBINE FILES IN {case_dir}")

    combine_files(120, case_dir, file_s=None, file_e=None)
    move_files(case_dir)
    remove_files(case, case_dir)


if __name__ == "__main__":
    main(cl_args=parse_args())

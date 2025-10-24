#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perform bootstrapping of different statstical tests for E3SM simulation ensembles."""

import argparse
import json
import multiprocessing as mp
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import scipy.stats as sts
import xarray as xr

import detclim


def randomise_new(ens_min, ens_max, ens_size, with_repl=False, ncases=2, uniq=False):
    ens_idx = sorted(range(ens_min, ens_max + 1))
    assert len(ens_idx) > ens_size, "ENSEMBLE SIZE MUST BE SMALLER THAN ENSEMBLE RANGE"
    if not with_repl and not uniq:
        selected = [random.sample(ens_idx, ens_size) for _ in range(ncases)]
    elif not with_repl:
        _sel = random.sample(ens_idx, ens_size * ncases)
        selected = [
            _sel[idx * ens_size : (idx + 1) * ens_size] for idx in range(ncases)
        ]
    else:
        selected = [
            [random.randint(ens_min, ens_max) for _ in range(ens_size)]
            for _ in range(ncases)
        ]
    return selected


def rolling_mean_data(data, period_len=12, time_var="time"):
    select = {time_var: period_len}
    return data.rolling(**select).mean().dropna(time_var)


def load_data(case_dirs, run_len, case_abbr, cases):
    """
    Load ensemble data into `xarray.Dataset`.

    Parameters
    ----------
    case_dirs : dict
        Dictionary mapping case name to absolute case `Path`

    run_len : str
        Length of run (e.g. 1month, 1year, etc.)

    case_abbr : list, tuple
        List or similar of abbrivated case names

    cases : dict
        Dictionary mapping (run_len, case_abbr) -> Case Long Name

    Returns
    -------
    ens_data : dictionary of `xarray.Dataset`
        Map of each case_abbr to an `xarray.Dataset`

    """
    files = {
        # _case: sorted(case_dirs[_case].glob(f"{cases[run_len][_case]}.eam*aavg.nc"))
        _case: Path(case_dirs[_case], f"{cases[run_len][_case]}.eam.h0.aavg.nc")
        for _case in case_abbr
    }
    ens_data = []
    for _case in case_abbr:
        ens_data.append(xr.open_dataset(files[_case]).load())
    ens_data = xr.concat(ens_data, dim="exp")
    ens_data["exp"] = case_abbr

    return ens_data


def ks_pval(data_x, data_y):
    _res = sts.mstats.ks_2samp(data_x, data_y)
    return _res[1]


def cvm_2samp(data_x, data_y):
    """Perform a 2 sample Cramer von Mises test, map output to a tuple."""
    _res = sts.cramervonmises_2samp(data_x, data_y)
    return _res.pvalue


def mannwhitney(data_1, data_2):
    """Perform a Wiloxon-Mann-Whitney U Test, return P-value."""
    return sts.mannwhitneyu(data_1, data_2, axis=1).pvalue


def epps_singleton(data_1, data_2):
    """Perform a 2 sample Epps Singleton test, return P-value."""
    try:
        _out = sts.epps_singleton_2samp(data_1, data_2, axis=1).pvalue
    except np.linalg.LinAlgError:
        _out = np.ones(data_1.shape[0])
    return _out


def anderson_pval(data_1, data_2):
    try:
        _res = sts.anderson_ksamp(
            [data_1, data_2], method=sts.PermutationMethod(n_resamples=500)
        )
    except ValueError:
        return 1.0
    return _res.pvalue  # pyright: ignore[reportAttributeAccessIssue]


def test_all_times(data, ens_ids, test_fcn):
    """Perform statistical test on two arrays across all times in the array.

    Parameters
    ----------
    data_1, data_2 : array_like
        Arrays of data for testing, dimension 2 (typically [ensemble, time]),
        with time dimension as the rightmost dimension.

    Returns
    -------
    test_output : `xarray.DataArray`
        Array with shape [data_n.shape[1]] of 2 sample statistical test p-value

    """
    data_1 = data.isel(exp=0, ens=ens_ids[0])
    data_2 = data.isel(exp=1, ens=ens_ids[1])
    _pval = test_fcn(data_1.T, data_2.T)
    try:
        _out = xr.DataArray(data=_pval, dims=("time",), coords={"time": data.time})
    except ValueError as _err:
        print(_err)
        return None

    return _out


def bootstrap_test(ens_ids, data, test_fcn):
    return data.apply(test_all_times, ens_ids=ens_ids, test_fcn=test_fcn)


def convert_to_array(pvals_in):
    """Convert pvals DataSet to an array of shape [n_iter, n_vars, n_times]

    Parameters
    ----------
        pvals_in : xarray.Dataset
            pvalues for each output field at each bootstrap iteration and time

    Returns
    -------
        pvals : numpy.ndarray
            pvalues array with shape [N output field, N bootstrap iteration, N times]

    """
    pvals_out = pvals_in.to_array().values
    return np.swapaxes(pvals_out, 0, 1)


def pvals_to_dataarray(pvals, test_id):
    """Convert array of p-values from a statstical test to an xarray.DataArray for output

    Parameters
    ----------
        pvals : array_like
            Array of p-values from the test of shape [n iter, n vars, n times]
        data_vars : list
            List of data variables of shape [n vars]
        times : xarray.DataArray
            Time array
        test_name : str
            Name of the statistical test

    Returns
    -------
        xarray.DataArray : DataArray with coordinates and metadata assigned

    """
    _pvals = convert_to_array(pvals)
    test_name, test_desc = test_id
    out_coords = {
        "iter": np.arange(_pvals.shape[0]),
        "vars": pvals.data_vars,
        "time": pvals.time,
    }
    return xr.DataArray(
        data=_pvals,
        coords=out_coords,
        dims=("iter", "vars", "time"),
        attrs={
            "units": "",
            "desc": f"2-sample {test_desc} p-value",
            "long_name": f"{test_desc.lower().replace(' ', '_')}_pvalue",
            "short_name": f"{test_name}_pvalue",
        },
    )


def main(
    indir,
    case_a="ctl",
    case_b="5pct",
    run_len="1year",
    n_iter=5,
    nnodes=1,
    rolling=0,
    permute=False,
    test_size=30,
    ctl_run=False,
):
    _total_time_s = time.perf_counter()
    with open(Path(detclim.data_path, "case_db.json"), "r", encoding="utf-8") as _cdb:
        cases = json.loads(_cdb.read())

    case_dirs = {
        _case: Path(indir, cases[run_len][_case], "run") for _case in [case_a, case_b]
    }

    print(f"LOAD DATA for {case_a} x {case_b}")
    _timers = time.perf_counter()
    dvars = json.loads(open("new_vars.json", "r", encoding="utf-8").read())["default"]
    ens_data = load_data(case_dirs, run_len, [case_a, case_b], cases)
    print(f"       IN {time.perf_counter() - _timers:.2f}s")
    print("     PERFORM ROLLING MEAN")
    _timers = time.perf_counter()
    if rolling != 0:
        ens_data = rolling_mean_data(ens_data[dvars], period_len=rolling)

    _emin = ens_data.ens.values.min()
    _emax = ens_data.ens.values.max()

    # If the two ensembles are the same, don't repeat between the two
    if case_a == case_b:
        unique = True
    else:
        unique = False

    ens_sel = [
        randomise_new(_emin, _emax, ens_size=test_size, ncases=2, uniq=unique)
        for _ in range(n_iter)
    ]
    print(f"       IN {time.perf_counter() - _timers:.2f}s")

    ks_test_vec = np.vectorize(ks_pval, signature="(n),(n)->()")
    cvm_test_vec = np.vectorize(cvm_2samp, signature="(n),(n)->()")
    # anderson_test_vec = np.vectorize(anderson_pval, signature="(n),(n)->()")

    print("     PERFORM BOOTSTRAPS")

    tests = {
        "ks": "Kolmogorov-Smirnov",
        "cvm": "Cramer von Mises",
        "mw": "Mann-Whitney",
        "es": "Epps Singleton",
    }

    partials = {
        "ks": partial(bootstrap_test, data=ens_data[dvars], test_fcn=ks_test_vec),
        "cvm": partial(bootstrap_test, data=ens_data[dvars], test_fcn=cvm_test_vec),
        "mw": partial(bootstrap_test, data=ens_data[dvars], test_fcn=mannwhitney),
        "es": partial(bootstrap_test, data=ens_data[dvars], test_fcn=epps_singleton),
    }

    _poolsize = min([mp.cpu_count() - 1, n_iter])
    with mp.Pool(_poolsize) as pool:
        pvals_all = {}
        for test_name in tests:
            _timers = time.perf_counter()
            print(f"     BOOTSTRAP {test_name} TEST")
            pvals_all[test_name] = xr.concat(
                pool.map(partials[test_name], ens_sel), dim="iter"
            )
            print(f"      IN {time.perf_counter() - _timers:.2f}s")

    for test_name in pvals_all:
        pvals_all[test_name] = pvals_to_dataarray(
            pvals_all[test_name], (test_name, tests[test_name])
        )
    ds_out = xr.Dataset(pvals_all)

    if ctl_run:
        out_dir = Path("bootstrap_data_ctl")
    else:
        out_dir = Path("bootstrap_data")

    if rolling == 0:
        run_shape = run_len
    else:
        run_shape = f"{run_len}_{rolling}avg"
    run_shape += f"_ts{test_size}"

    ds_out.to_netcdf(
        Path(
            out_dir,
            f"bootstrap_output.{run_shape}.{case_a}_{case_b}_n{n_iter}.nc",
        )
    )
    print(
        f"COMPLETED: TOTAL TIME: {(time.perf_counter() - _total_time_s) / 60:.2f} min"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--base", default="ctl", help="Short name of the base case.")
    parser.add_argument("--test", default="5pct", help="Short name of the test case.")
    parser.add_argument(
        "--runlen", default="1year", help="Length of model run, matching case_db.json"
    )
    parser.add_argument(
        "--iter", default=5, help="Number of bootstrap iterations to perform."
    )
    parser.add_argument("--nodes", default=1, help="Number of nodes used.")
    parser.add_argument(
        "--rolling",
        default=0,
        help="Perform a rolling time mean with specified period length.",
        type=int,
    )
    parser.add_argument(
        "-p",
        "--permute",
        action="store_true",
        default=False,
        help="Use permutation method for control",
    )
    parser.add_argument(
        "--esize",
        default=30,
        type=int,
        help="Ensemble size, default=30",
    )
    parser.add_argument(
        "--ctl",
        action="store_true",
        default=False,
        help="Write output to bootstrap_data_ctl for use as control bootstrap for threshold finding",
    )
    parser.add_argument(
        "-i",
        "--indir",
        help="Data input directory, if not on LCRC",
        type=Path,
        default=Path("/lcrc/group/e3sm/ac.mkelleher/scratch/chrys/"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    cl_args = parse_args()
    # case_a = cl_args.base
    main(
        indir=cl_args.indir,
        case_a=cl_args.base,
        case_b=cl_args.test,
        run_len=cl_args.runlen,
        n_iter=int(cl_args.iter),
        nnodes=int(cl_args.nodes),
        rolling=int(cl_args.rolling),
        permute=cl_args.permute,
        test_size=cl_args.esize,
        ctl_run=cl_args.ctl,
    )

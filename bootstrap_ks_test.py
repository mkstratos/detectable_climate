"""Perform many K-S tests comparing two E3SM ensembles.
"""
import json
import time
import random
from pathlib import Path
import argparse

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
import xarray as xr
from dask.distributed import Client

plt.style.use("ggplot")

# CASES = {
#     "ctl": "20230124.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00200_n0120",
#     "ctl-2mo": "20230126.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00200_n0120",
#     "1pct-2mo": "20230126.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00202_n0120",
#     "0p5pct": "20221205.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00201_n0030",
#     "2p5pct": "20230124.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00205_n0120",
#     "5pct": "20221205.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00201_n0030",
#     "10pct": "20221201.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0022_n0030",
#     "50pct": "20221206.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0030_n0030",
#     "old-ctl": "20221130.F2010.ne4_oQU240.dtcl_control_n0030",
#     "old-2p5pct": "20230123.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00205_n0120",
#     "new-ctl": "20230403.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00200_n0120",
#     "new-2p5pct": "20230325.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00205_n0120",
#     "new-5pct": "20230403.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00210_n0120",
#     "new-10pct": "20230329.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00220_n0120",
#     "c1-1pct": "20230422.F2010.ne4_oQU240.dtcl_clubb_c1_2p4240_n0120",
#     "gworo-10pct": "20230515.F2010.ne4_oQU240.dtcl_effgw_oro_0p4125_n0120",
#     "gworo-1pct": "20230517.F2010.ne4_oQU240.dtcl_effgw_oro_0p37875_n0120",
#     "pl_1e-10": "20230321.F2010.ne4_oQU240.dtcl_pertlim_1e-10_n0120",
#     "pl_1e-14": "20230322.F2010.ne4_oQU240.dtcl_pertlim_1e-14_n0120",
#     "gworo-1yr-0p5pct": "20230615.F2010.ne4_oQU240.dtcl_effwg_oro_0p376875_n0120",
#     "gworo-1yr-1pct": "20230613.F2010.ne4_oQU240.dtcl_effgw_oro_0p3787_n0120",
#     "gworo-1yr-10pct": "20230613.F2010.ne4_oQU240.dtcl_effgw_oro_0p4125_n0120",
# }

REJECT_THR = 0.05


# @dask.delayed
def ks_all_times_nv(data_1, data_2):
    """Perform K-S test on two arrays across all times in the array.

    Parameters
    ----------
    data_1, data_2 : array_like
        Arrays of data for testing, dimension 2 (typically [ensemble, time]), with time
        dimension as the rightmost dimension.

    Returns
    -------
    ks_test_output : `da.array`
        Dask array with shape [data_n.shape[1], 2] of 2 sample K-S test
        results (statstic, p-value)

    """
    return da.array( # type: ignore
        [
            sts.ks_2samp(data_1[:, _tix], data_2[:, _tix], method="asymp")
            for _tix in range(data_1.shape[1])
        ]
    )


def ks_all_times(data_1, data_2):
    """Perform K-S test on two arrays across all times in the array.

    Parameters
    ----------
    data_1, data_2 : array_like
        Arrays of data for testing, dimension 2 (typically [ensemble, time]), with time
        dimension as the rightmost dimension.

    Returns
    -------
    ks_test_output : `da.array`
        Dask array with shape [data_n.shape[1], 2] of 2 sample K-S test
        results (statstic, p-value)

    """
    # ks_test = np.vectorize(sts.mstats.ks_2samp, signature="(n),(n)->(),()", excluded=["method"])
    # ks_stat, ks_pval = ks_test(data_1.T, data_2.T, method="asymp")
    ks_test = np.vectorize(sts.mstats.ks_2samp, signature="(n),(n)->(),()")
    ks_stat, ks_pval = ks_test(data_1.T, data_2.T)
    return da.array(ks_stat), da.array(ks_pval)  # type: ignore
    # return da.array(ks_test(data_1, data_2, method="asymp"))
    # return da.array(
    #     [
    #         sts.ks_2samp(data_1[:, _tix], data_2[:, _tix], method="asymp")
    #         for _tix in range(data_1.shape[1])
    #     ]
    # )


def randomise_sample(ens_data, case_abbr, ens_size=30, niter=1):
    """Generate a random sample of ensemble members.

    Parameters
    ----------
    ens_data : dict
        Dict of xarray.Dataset
    case_abbr : list
        Case name abbreviations used to select cases from `ens_data`
    ens_size : int
        Size of ensemble sample, default=30
    niter : int
        Number of iterations for bootstraping. Default=1

    Returns
    -------
    selected, left_out : dict
        Indicies of selected and left-out ensemble members for each case in `case_abbr`

    """
    assert all(_abbr in ens_data for _abbr in case_abbr), "All cases not in ens_data"

    # Check that both input ensembles are the same size, if they are, then the sample
    # should not repeat between the two (in case of control run, there's no duplication)
    if (
        not set(ens_data[case_abbr[0]].ens.values).difference(
            ens_data[case_abbr[1]].ens.values
        )
        and ens_size <= ens_data[case_abbr[0]].ens.values.shape[0] // 2
    ):
        # selected = {_case: [] for _case in case_abbr}
        selected_0 = []
        selected_1 = []
        for _itr in range(niter):
            # Generate one sample, so there is no replacement between the two ensembles
            # Since they're the same, then just use the values from case_abbr[0]
            sample = random.sample(
                list(ens_data[case_abbr[0]].ens.values), ens_size * 2
            )
            # First half of the sample is for first case
            selected_0.append(sample[:ens_size])
            # Second half of the sample is for second case
            selected_1.append(sample[ens_size:])
        selected = [selected_0, selected_1]
    else:
        selected = [
            [
                random.sample(list(ens_data[_case].ens.values), ens_size)
                for _ in range(niter)
            ]
            for _case in case_abbr
        ]

    left_out = [[] for _case in case_abbr]
    for _cix, _case in enumerate(case_abbr):
        case_ens = set(ens_data[_case].ens.values)
        # shuf_ens = [set(selected[_case][idx]) for idx in range(niter)]

        for idx in range(niter):
            shuf_ens = set(selected[_cix][idx])

            if len(case_ens) >= len(shuf_ens):
                avail = list(case_ens.difference(shuf_ens))
            else:
                avail = list(shuf_ens.difference(case_ens))

            left_out[_cix].append(random.sample(avail, 1)[0])

    if niter == 1:
        selected = [selected[_cix][0] for _cix, _ in enumerate(case_abbr)]
        left_out = [left_out[_cix][0] for _cix, _ in enumerate(case_abbr)]

    return selected, left_out


def plot_single_var_summary(ens_data, case_abbr, test_var="T", group_mean=False):
    """Create an example plot illustrating the testing process for a single variable.

    Parameters
    ----------
    ens_data : dict
        Dictionary of `xarray.Dataset`s
    case_abbr : list or tuple
        Length two list or tuple of cases in `ens_data` to compare
    test_var : str, optional
        Variable to plot, by default "T"
    group_mean : bool, l
        Plot against overall ensemble mean rather than a left-out member, by default False

    """
    # (ens_data - ens_data.mean(dim="ens"))["U"].plot.line(x="time")
    ens_shuffle, ens_loo = randomise_sample(ens_data, case_abbr)

    data_a = ens_data[case_abbr[0]][test_var].isel(ens=ens_shuffle[0])
    data_b = ens_data[case_abbr[1]][test_var].isel(ens=ens_shuffle[1])
    # times = ens_data[case_abbr[0]].time.values
    times = np.arange(data_a.time.shape[0])

    _, axes = plt.subplots(2, 2, figsize=(10, 8))

    if group_mean:
        # Plot against mean for group
        axes[0, 0].plot(
            times, (data_a - data_a.mean(dim="ens")).T, color="C0", label=case_abbr[0]
        )
        axes[0, 0].plot(
            times, (data_b - data_b.mean(dim="ens")).T, color="C1", label=case_abbr[1]
        )
    else:
        # Plot against leave-one-out for each case (if n_test < (n_ens-1) then
        # it's the first one left out)
        axes[0, 0].plot(
            times,
            (data_a - ens_data[case_abbr[0]][test_var].isel(ens=ens_loo[0])).values.T,
            label=case_abbr[0],
            color="C0",
        )
        axes[0, 0].plot(
            times,
            (data_b - ens_data[case_abbr[1]][test_var].isel(ens=ens_loo[1])).values.T,
            label=case_abbr[0],
            color="C1",
        )

    axes[0, 0].set_title(f"{test_var} ensemble spread")
    axes[0, 0].set_xlabel("Time step")

    (aline,) = axes[0, 1].plot(times, data_a.mean(dim="ens"), label=case_abbr[0])
    (bline,) = axes[0, 1].plot(times, data_b.mean(dim="ens"), label=case_abbr[1])

    ax_diff = axes[0, 1].twinx()
    (diffline,) = ax_diff.plot(
        times,
        (data_a.mean(dim="ens") - data_b.mean(dim="ens")).pipe(np.abs),
        label="Difference",
        color="grey",
    )
    ax_diff.set_ylabel(f"{test_var} difference")
    axes[0, 1].legend(handles=[aline, bline, diffline])
    axes[0, 1].set_title(f"{test_var} mean")

    axes[1, 0].plot(times, data_a.std(dim="ens"), label=case_abbr[0])
    axes[1, 0].plot(times, data_b.std(dim="ens"), label=case_abbr[1])
    axes[1, 0].set_title(f"{test_var} std dev")

    ks_stat, ks_pval = ks_all_times(data_a, data_b)

    ax_pval = axes[1, 1].twinx()

    (ks_line,) = axes[1, 1].plot(times, ks_stat, label="Statistic", lw=1)
    (pv_line,) = ax_pval.plot(times, ks_pval, color="C1", label="P-value", lw=1)
    _ = ax_pval.plot(
        times[ks_pval < REJECT_THR], ks_pval[ks_pval < REJECT_THR], "C1o", ms=2
    )
    axes[1, 1].set_ylim([0, 1.0])

    ax_pval.axhline(REJECT_THR, color="C1", ls="--", alpha=0.5)

    axes[1, 1].set_title(f"{test_var} K-S Test")
    axes[1, 1].legend(handles=[ks_line, pv_line])
    axes[1, 1].set_ylabel("Test statistic", color=ks_line.get_color())
    ax_pval.set_ylabel("Test p-value", color=pv_line.get_color())
    for _ax in axes.flatten():
        _ax.grid(visible=True)
    plt.tight_layout()
    plt.savefig(f"plt_{test_var}_{case_abbr[0]}x{case_abbr[1]}_compare.png")


def check_ensemble_vars(ens_data, common_vars, case):
    """Check all common_vars to make sure they're not constant across the ensemble."""
    const_vars = []
    for _var in common_vars:
        ens_std = ens_data[case][_var].mean(dim="time").std(dim="ens").values
        if ens_std == 0.0:
            const_vars.append(_var)
    return const_vars


def ks_bootstrap_serial(ens_data, case_abbr, data_vars, permute):
    """Perform multiple K-S tests on ensemble data.

    Parameters
    ----------
    ens_data : dict
        Dict of `xarray.Dataset`s of ensemble data
    case_abbr : list, tuple
        Length two list or tuple of cases in `ens_data` to compare
    data_vars : list
        List of data variables on which to perform the test
    permute : logical
        Use permutation to create control ensemble

    Returns
    -------
    ks_stat, ks_pval : array_like
        Arrays of K-S test statistic and p-value for each bootstrap iteration.

    """
    ens_shuffle, _ = randomise_sample(ens_data, case_abbr, ens_size=30, niter=1)
    # ens_shuffle = {
    # _case: random.sample(list(ens_data[_case].ens.values), 30) for _case in case_abbr
    # }
    # _ = {
    #     _case: list(
    #         set(ens_data[_case].ens.values).difference(ens_shuffle[_case])
    #     )  # [0]
    #     for _case in case_abbr
    # }
    # print(ens_loo)
    ks_stat_i = []
    ks_pval_i = []

    for test_var in data_vars:
        data_a = ens_data[case_abbr[0]][test_var].isel(ens=ens_shuffle[0])
        data_b = ens_data[case_abbr[1]][test_var].isel(ens=ens_shuffle[1])

        if hasattr(data_a, "time"):
            _stat, _pval = ks_all_times(data_a.data, data_b.data)
            ks_stat_i.append(_stat)
            ks_pval_i.append(_pval)

    return np.array(ks_stat_i), np.array(ks_pval_i)


def ks_bootstrap(ens_data, case_abbr, dask_client, n_iter=5, test_size=30, permute=False):
    """Perform multiple K-S tests on selected variables in two ensembles.

    Parameters
    ----------
    ens_data : dict
        Dict of `xarray.Dataset`s of ensemble data
    case_abbr : list, tuple
        Length two list or tuple of cases in `ens_data` to compare
    dask_client : `dask.distributed.Client`
        Dask client on which to submit
    n_iter : int, optional
        Number of boostrap tests to perform, by default 5
    test_size : int, optional
        Number of ensemble members for each K-S test, by default 30
    permute : logical
        Use permutation to create control ensemble

    Returns
    -------
    ks_stat, ks_pval : `dask.array`
        Dask arrays for the K-S test statistic and p-value
    idx_0, idx_1 : array_like
        Array of shape [`n_iter`, `test_size`] of ensemble indicies used for each
        bootstrap iteration

    """
    futures = []
    with open("run_scripts/new_vars.json", "r", encoding="utf-8") as _vf_in:
        data_vars = sorted(json.load(_vf_in)["default"])
    vars_out = []
    # random_index = [
    #     [
    #         random.sample(
    #             list([_ for _ in range(ens_data[case_abbr[0]]["ens"].shape[0])]),
    #             test_size,
    #         )
    #         for _ in range(n_iter)
    #     ],
    #     [
    #         random.sample(
    #             list([_ for _ in range(ens_data[case_abbr[1]]["ens"].shape[0])]),
    #             test_size,
    #         )
    #         for _ in range(n_iter)
    #     ],
    # ]

    # random_index = [
    #     [
    #         random.sample(
    #             list(range(ens_data[_abbr]["ens"].shape[0])),
    #             test_size,
    #         )
    #         for _ in range(n_iter)
    #     ]
    #     for _abbr in case_abbr
    # ]
    if not permute:
        # Get random sample, outputs a dict: {case_a: [i1, i2, ...], case_b: [i1, i2, ...]}
        random_index, _ = randomise_sample(ens_data, case_abbr, test_size, niter=n_iter)
        print(f"RANDOM INDEX SIZE: {np.array(random_index).shape}")

    for rse in range(n_iter):
        var_futures = []
        data_0 = ens_data[case_abbr[0]].isel(**{"ens": random_index[0][rse]})
        data_1 = ens_data[case_abbr[1]].isel(**{"ens": random_index[1][rse]})

        for test_var in data_vars:
            if test_var in data_0.data_vars and test_var in data_1.data_vars:
                var_futures.append(
                    dask_client.submit(ks_all_times, data_0[test_var], data_1[test_var])
                )
                # Keep a list of all the variables tested, but only one copy
                if test_var not in vars_out:
                    vars_out.append(test_var)
        futures.append(var_futures)

    results = da.array(dask.compute(*dask_client.gather(futures)))  # type: ignore
    ks_stat = results[..., 0, :]
    ks_pval = results[..., 1, :]

    return ks_stat, ks_pval, np.array(random_index), vars_out


def output_data(ks_stat, ks_pval, rnd_idx, times, data_vars):
    """_summary_

    Parameters
    ----------
    ks_stat : _type_
        _description_
    ks_pval : _type_
        _description_
    ens_data : _type_
        _description_
    """

    out_coords = {
        "iter": np.arange(ks_stat.shape[0]),
        "vars": data_vars,
        "time": times,
    }
    out_dims = ("iter", "vars", "time")
    ks_stat_xr = xr.DataArray(
        np.array(ks_stat),
        coords=out_coords,
        dims=out_dims,
        attrs={
            "units": "",
            "desc": "2-sample K-S test statistic",
            "long_name": "kolmogorov_smirnov_test_statistic",
            "short_name": "ks_pval",
        },
    )
    ks_pval_xr = xr.DataArray(
        np.array(ks_pval),
        coords=out_coords,
        dims=out_dims,
        attrs={
            "units": "",
            "desc": "2-sample K-S test P-value",
            "long_name": "kolmogorov_smirnov_test_p_value",
            "short_name": "ks_stat",
        },
    )
    rnd_idx = xr.DataArray(
        rnd_idx,
        coords={
            "case": [0, 1],
            "iter": out_coords["iter"],
            "index": np.arange(rnd_idx.shape[-1]),
        },
        attrs={
            "units": "",
            "desc": "Index of ensemble members for each case and iteration",
        },
    )
    return xr.Dataset({"stat": ks_stat_xr, "pval": ks_pval_xr, "rnd_idx": rnd_idx})


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
        _case: sorted(case_dirs[_case].glob(f"{cases[run_len][_case]}.eam*aavg.nc"))
        for _case in case_abbr
    }

    ens_data = {}
    for _case in case_abbr:
        ens_data[_case] = []
        for _file in files[_case]:
            ens_data[_case].append(
                xr.open_dataset(
                    _file,
                )
            )
        ens_data[_case] = xr.concat(ens_data[_case], dim="ens")

    return ens_data


def rolling_mean_data(ens_data, cases, period_len=12, time_var="time"):
    """
    Take rolling mean of an xarray Dataset.

    Parameters
    ----------
    ens_data : dict
        Dictionary mapping case names to `xarray.Datasets` for base / test cases
    cases : list, tuple
        Abbreviations for case names in `ens_data`
    period_len : int
        Number of time periods used in averaging (same units as the `time_var`)
    time_var : str

    """
    rolling_means = {_case: {} for _case in cases}
    select = {time_var: period_len}
    for _case in cases:
        for _var in ens_data[_case].data_vars:

            try:
                rolling_means[_case][_var] = (
                    ens_data[_case][_var].rolling(**select).mean().dropna(time_var)
                )
            except TypeError:
                # Happens for things like "time" variable which can be a cftime.datetime
                continue
            except KeyError:
                # Happens for coordinate variables
                continue

    return {_case: xr.Dataset(rolling_means[_case]) for _case in cases}


def main(case_a="ctl", case_b="5pct", run_len="1year", n_iter=5, nnodes=1, rolling=0, permute=False, test_size=30):
    """Perform bootstrap K-S testing of two ensembles of E3SM

    Parameters
    ----------
    case_a : str, optional
        Base case name, by default "ctl"
    case_b : str, optional
        Test case name, by default "5pct"
    run_len : str, optional
        Case run length (defauylt 1year)
    n_iter : int, optional
        Number of bootstrap iterations to perform, default 5
    permute : logical, optional
        Use permutation for control ensemble, default False
    test_size: int
        Number of ensemble members per test, default 30

    """
    with open("case_db.json", "r", encoding="utf-8") as _cdb:
        cases = json.loads(_cdb.read())

    # with open("run_scripts/new_vars.json", "r", encoding="utf-8") as _vf_in:
    #     test_vars = sorted(json.load(_vf_in)["default"])

    scratch = Path("/lcrc/group/e3sm/ac.mkelleher/scratch/chrys/")
    case_dirs = {
        _case: Path(scratch, cases[run_len][_case], "run") for _case in [case_a, case_b]
    }

    print("LOAD DATA")
    # plot_single_var_summary(ens_data, case_abbr, test_var="T")
    ens_data = load_data(case_dirs, run_len, [case_a, case_b], cases)
    if rolling != 0:
        ens_data = rolling_mean_data(ens_data, [case_a, case_b], period_len=rolling)
    print("LAUNCH CLIENT")
    # client = Client(n_workers=n_iter // 1.1)
    _workers = 36 * nnodes
    # with Client(n_workers=_workers, processes=True, interface="lo") as client:
    with Client(
        n_workers=_workers, threads_per_worker=1, processes=True, interface="ib0"
    ) as client:
        print(client)
        print(f"PERFORM {n_iter} TESTS")
        ks_stat, ks_pval, rnd_indx, test_vars = ks_bootstrap(
            ens_data, [case_a, case_b], client, n_iter=n_iter, permute=permute, test_size=test_size
        )
        time.sleep(1)
        print("OUTPUT TO FILE")
        ds_out = output_data(
            ks_stat, ks_pval, rnd_indx, ens_data[case_a]["time"], test_vars
        )
        if rolling == 0:
            run_shape = run_len
        else:
            run_shape = f"{run_len}_{rolling}avg"
        if test_size != 30:
            run_shape += f"_ts{test_size}"
        ds_out.to_netcdf(
            Path(
                "bootstrap_data",
                f"bootstrap_output.{run_shape}.{case_a}_{case_b}_n{n_iter}.nc",
            )
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
    return parser.parse_args()


if __name__ == "__main__":
    cl_args = parse_args()
    # case_a = cl_args.base
    main(
        case_a=cl_args.base,
        case_b=cl_args.test,
        run_len=cl_args.runlen,
        n_iter=int(cl_args.iter),
        nnodes=int(cl_args.nodes),
        rolling=int(cl_args.rolling),
        permute=cl_args.permute,
        test_size=30,
    )

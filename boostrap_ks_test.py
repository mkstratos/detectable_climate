import xarray as xr
import matplotlib.pyplot as plt
from cartopy import crs as ccr
from pathlib import Path
import scipy.stats as sts
import numpy as np
import random
import json

plt.style.use("ggplot")

CASES = {
    "ctl": "20221130.F2010.ne4_oQU240.dtcl_control_n0030",
    "5pct": "20221205.F2010.ne4_oQU240.dtcl_zmconv_c0_0p00201_n0030",
    "10pct": "20221201.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0022_n0030",
    "50pct": "20221205.F2010.ne4_oQU240.dtcl_zmconv_c0_0p0030_n0030",
}
REJECT_THR = 0.05


def ks_all_times(data_a, data_b):
    """Compute Kolmogorov–Smirnov test for each time in two xarray.DataArrays"""
    stat = []
    pval = []
    for _it in range(data_a.time.shape[0]):
        _stat, _pval = sts.ks_2samp(
            data_a.isel(time=_it).values, data_b.isel(time=_it).values
        )
        stat.append(_stat)
        pval.append(_pval)
    return np.array(stat), np.array(pval)


def randomise_sample(ens_data, case_abbr):

    selected = {
        _case: random.sample(list(ens_data[_case].ens.values), 30)
        for _case in case_abbr
    }

    left_out = {}
    for _case in case_abbr:
        case_ens = set(ens_data[_case].ens.values)
        shuf_ens = set(selected[_case])

        if len(case_ens) >= len(shuf_ens):
            left_out[_case] = list(case_ens.difference(shuf_ens))[0]
        else:
            left_out[_case] = list(shuf_ens.difference(case_ens))[0]

    return selected, left_out


def plot_single_var_summary(ens_data, case_abbr, test_var="T", group_mean=False):
    # (ens_data - ens_data.mean(dim="ens"))["U"].plot.line(x="time")
    ens_shuffle, ens_loo = randomise_sample(ens_data, case_abbr)

    data_a = ens_data[case_abbr[0]][test_var].isel(ens=ens_shuffle[case_abbr[0]])
    data_b = ens_data[case_abbr[1]][test_var].isel(ens=ens_shuffle[case_abbr[1]])
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
            (
                data_a
                - ens_data[case_abbr[0]][test_var].isel(ens=ens_loo[case_abbr[0]])
            ).values.T,
            label=case_abbr[0],
            color="C0",
        )
        axes[0, 0].plot(
            times,
            (
                data_b
                - ens_data[case_abbr[1]][test_var].isel(ens=ens_loo[case_abbr[1]])
            ).values.T,
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
    pv_points = ax_pval.plot(
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


def main(case_a="ctl", case_b="5pct"):
    test_vars = json.load(open("run_scripts/new_vars.json", "r"))["default"]

    scratch = Path("/lcrc/group/e3sm/ac.mkelleher/scratch/chrys/")
    case_abbr = [case_a, case_b]
    case_dirs = {_case: Path(scratch, CASES[_case], "run") for _case in case_abbr}

    files = {
        _case: sorted(case_dirs[_case].glob(f"{CASES[_case]}.eam*aavg.nc"))
        for _case in case_abbr
    }

    ens_data = {}
    for _case in case_abbr:
        ens_data[_case] = xr.open_mfdataset(
            files[_case], combine="nested", concat_dim="ens", parallel=True
        )
    # plot_single_var_summary(ens_data, case_abbr, test_var="T")


if __name__ == "__main__":
    main()

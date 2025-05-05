"""Plot the results of bootstrapping tests.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from statsmodels.stats import multitest as smm

plt.style.use("ggplot")
REJECT_THR = 0.05


def fmt_case(case):
    if case == "ctl":
        _out = "Control"
    elif case == "ctl_ts40":
        _out = "Control: n40"
    elif case == "ctl-2mo":
        _out = "Control [2 mo]"
    elif case == "ctl-v3":
        _out = "Control v3"
    elif case == "new-ctl":
        _out = "Control [new]"
    elif case == "ctl-next":
        _out = "Control [next]"
    elif "old" in case:
        num = float(case.replace("old-", "").replace("pct", "").replace("p", "."))
        _out = f"old {num:.1f}%"
    elif "new" in case:
        num = float(case.replace("new-", "").replace("pct", "").replace("p", "."))
        _out = f"new {num:.1f}%"
    elif "c1" in case:
        num = float(
            case.split("-")[1]
            .replace("pct", "")
            .replace("p", ".")
            .replace("-2mo", "")
            .replace("_ts40", "")
        )
        _out = f"clubb_c1 {num:.1f}%"
    elif "gworo" in case or "effgw_oro" in case:
        if "yr" in case:
            num = float(
                case.split("-")[-1]
                .replace("pct", "")
                .replace("ct", "")
                .replace("p", ".")
                .replace("-2mo", "")
                .replace("_ts40", "")
            )
            _out = f"GW orog {num:.1f}%"
        else:
            num = float(
                case.split("-")[1]
                .replace("pct", "")
                .replace("ct", "")
                .replace("p", ".")
                .replace("-2mo", "")
                .replace("_ts40", "")
            )
            _out = f"GW orog {num:.1f}%"
    elif "pl" in case:
        num = float(
            case.split("_")[1].replace("pct", "").replace("p", ".").replace("-2mo", "")
        )
        _out = f"PertLim {num:.1e}"
    elif "opt" in case or "fastmath" in case:
        _out = case
    else:
        num = float(case.replace("pct", "").replace("p", ".").replace("-2mo", ""))
        _out = f"{num:.1f}%"
    if "ts" in case:
        _out += "_n40"
    return _out


def bar_plots(
    all_case_data: dict, thr_uc: int = 13, out_dir: Path = Path("figures")
) -> None:
    """
    Plot upper quantile of number of rejected variables and number of failed tests.

    Parameters
    ----------
    all_case_data : dict
        Dictionary with dictionaries of data for each case
    thr_uc : int
        Threshold for test failure of un-corrected MVK test

    """
    for case in all_case_data:
        case_a, case_b = case
        case_data = all_case_data[case]

        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        time_step = np.arange(case_data["n_times"])
        n_reject_uq = case_data["n_reject_qtiles"][-1]

        rejections = {
            f"{100 * (1 - REJECT_THR)}%": n_reject_uq,
            f"{100 * (1 - REJECT_THR)}% [Corrected]": case_data["n_reject_cr"].max(
                axis=0
            ),
        }

        width = 0.5
        mult = 0
        for name, nreject in rejections.items():
            offset = width * mult
            rect = axes[0].bar(time_step + offset, nreject, width=0.45, label=name)
            axes[0].bar_label(rect, padding=3, color=rect[-1].get_facecolor())
            # axes[0].bar_label(rect, padding=3)
            mult += 1

        axes[0].set_xticks(time_step, ["1-12", "2-13", "3-14"])

        axes[0].axhline(
            REJECT_THR * case_data["n_vars"],
            color="#343",
            ls="-.",
            label=f"{REJECT_THR * 100}% of variables",
            zorder=1,
        )
        axes[0].legend()

        axes[0].set_title(
            f"Number of variables rejected at {(1 - REJECT_THR) * 100}% confidence"
        )
        axes[0].set_xlabel("Simulated month")
        axes[0].set_ylabel("N variables")

        test = {
            "Un-corrected": (case_data["n_reject"] > thr_uc).sum(axis=0),
            "Corrected": (
                case_data["n_reject_cr"] > case_data["n_vars"] * REJECT_THR
            ).sum(axis=0),
        }

        mult = 0
        for name, itest in test.items():
            offset = width * mult
            rect = axes[1].bar(time_step + offset, itest, width=0.45, label=name)
            axes[1].bar_label(
                rect, padding=3, color=rect[-1].get_facecolor(), zorder=10
            )
            mult += 1

        axes[1].axhline(
            REJECT_THR * case_data["n_iter"],
            color="#343",
            ls="-.",
            label=f"{REJECT_THR * 100}% of tests",
            zorder=1,
        )
        axes[1].legend()
        axes[1].set_xticks(time_step, ["1-12", "2-13", "3-14"])
        axes[1].set_xlabel("Simulated month")

        axes[1].set_title(f'Number of tests (of {case_data["n_iter"]}) "failing"')

        _reject = f"{REJECT_THR:.2f}".replace(".", "p")
        fig.suptitle(f"{case_a} x {case_b}")
        plt.tight_layout()
        plt.savefig(Path(out_dir, f"plt_{case_a}-{case_b}_n{case_data['n_iter']}.png"))


def load_data(file: Path, reject_thr: float = 0.95) -> dict:
    """
    Load bootstrap data, correct p-values, get lower, median, and upper quantiles.

    Parameters
    ----------
    case : tuple
        Tuple of case names (Base, Test)
    file : Path
        Input bootstrap file path
    reject_thr : float
        Threshold for statstical rejection of H0 (Base == Test), optional. Default=0.95

    Returns
    -------
    case_data : dict
        Dictionary of processed case data with the keys:
        - n_reject: number of rejected variables for each bootstrap instance at
            alpha=reject_thr. Shape: (n_iter, n_times)
        - n_reject_cr : n_reject after FDR correction. Shape: (n_iter, n_times)
        - n_reject_qtiles : Upper, median, lower quantiles for number of rejected vars
            across bootstrap instances. Shape: (n_times, 3)
        - n_reject_qtiles_cr: Quantiles of number of rejected variables after FDR
            correction. Shape: (n_times, 3)
        - n_iter, n_vars, n_times: Data shape: number of bootstrap iterations, number of
            variables tested, and number of times, respectively

    """
    ks_res = xr.open_dataset(file)
    ks_pval = ks_res["pval"].values

    quantile = reject_thr * 100
    ks_pval_cr = np.array(
        [
            smm.fdrcorrection(
                ks_pval[jdx].flatten(),
                alpha=REJECT_THR,
                method="indep",
                is_sorted=False,
            )[1].reshape(ks_pval[jdx].shape)
            for jdx in range(ks_pval.shape[0])
        ]
    )

    n_reject = np.array((ks_pval < reject_thr).sum(axis=1))
    nreject_qtiles = np.percentile(n_reject, [100 - quantile, 50, quantile], axis=0)

    n_reject_cr = np.array((ks_pval_cr < REJECT_THR).sum(axis=1))
    nreject_qtiles_cr = np.percentile(
        n_reject_cr, [100 - quantile, 50, quantile], axis=0
    )
    n_iter, n_vars, n_times = ks_pval.shape
    case_data = {
        "n_reject": n_reject,
        "n_reject_qtiles": nreject_qtiles,
        "n_reject_cr": n_reject_cr,
        "n_reject_qtiles_cr": nreject_qtiles_cr,
        "n_iter": n_iter,
        "n_vars": n_vars,
        "n_times": n_times,
    }
    return case_data


def get_bs_file(
    case: tuple, niter: int, run_len: str, base_dir: Path = Path("bootstrap_data")
) -> Path:
    """
    Get bootstrap output file path given two case names.

    Parameters
    ----------
    case : tuple
        Tuple of cases (Base, Test)
    niter : int
        Number of bootstrap iterations in the input file
    base_dir : Path
        Location of bootstrap output data

    Returns
    -------
    case_file : Path
        Path to bootstrap output

    """
    base_name = "bootstrap_output"
    case_file = Path(base_dir, f"{base_name}.{run_len}.{case[0]}_{case[1]}_n{niter}.nc")
    return case_file


def main():
    """Main method: Identify cases, load data, create plots."""
    params = {"effgw_oro": [0.5, 1, 10, 20, 30, 40, 50], "clubb_c1": [1, 5, 10]}

    # Type hit is that `cases`` is a list of tuples of two strings each
    cases: list[tuple[str, str]] = [("ctl", "ctl")]
    case_data = {}

    for param, levels in params.items():
        _case = [
            ("ctl", f"{param}-{f'{pct:.1f}'.replace('.', 'p')}pct") for pct in levels
        ]
        cases.extend(_case)

    # Add additional test not included in the parameter sweep
    cases.append(("ctl", "ctl-next"))

    print("LOAD DATA")
    for case in cases:
        _file = get_bs_file(case, niter=500, run_len="1year")
        case_data[case] = load_data(_file)

    print("MAKE PLOTS")
    bar_plots(case_data, out_dir=Path("figures"))


if __name__ == "__main__":
    main()

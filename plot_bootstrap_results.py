#!/usr/bin/env python
# coding: utf-8
"""Plot results from K-S test bootstrapping, perform FDR Corrections.
"""

import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats import multitest as smm
from matplotlib.ticker import ScalarFormatter
from results_plot import fmt_case

plt.style.use("default")

ALPHA = 0.05
_THR = {0.01: 7, 0.05: 11}

ctl_key = tuple(["PertLim 1.0e-10"] * 2)
REJECT_UC_THR = _THR[ALPHA]


def reject_vars(pvals: np.ndarray, alpha: float = 0.05):
    """Determine number of fields for which null hypothesis is rejected.

    Parameters
    ----------
    pvals : np.array
        Array of p-values from bootstrap results, shape is: (bootstrap iter, var, time)
    alpha : float, optional
        P-value at which to reject H0, by default 0.05

    Returns
    -------
    Array of shape (bootstrap iters, times) with number of fields with p-val below alpha
        _description_
    """
    return np.array((pvals < alpha).sum(axis=1))


class CaseData:
    methods = {"uncor": "Un-corrected", "p": "Corr [B-H]", "n": "Corr [B-Y]"}

    def __init__(
        self,
        btsp_file: Path,
        case_a: str,
        case_b: str = "",
        uc_thr: int = 11,
        alpha: float = 0.05,
        pct_change: float = 0,
    ):
        self.alpha = alpha
        if case_b == "":
            case_b = case_a
        self.pct_change = pct_change
        self.cases = (case_a, case_b)
        self.n_iter = int(btsp_file.stem.split("_")[-1][1:])
        _ks_res = xr.open_dataset(btsp_file)
        self.time_step = np.arange(_ks_res.time.shape[0])

        self.ks_pval = _ks_res["pval"].values
        self.pval_cr = {_method: self._correct_pvals(_method) for _method in ["p", "n"]}
        self.nreject = {"uncor": reject_vars(self.ks_pval, self.alpha)}

        for _method in ["p", "n"]:
            self.nreject[_method] = reject_vars(self.pval_cr[_method])

        self.thrs = {"uncor": uc_thr, "p": 0, "n": 0}
        self.reject_qtiles = self._compute_stats()

        # Number of failed tests based on threshold, set by control run for
        # un-corrected, 0 for corrected
        self.ntests = {
            _method: (self.nreject[_method] > self.thrs[_method]).sum(axis=0)
            for _method in self.nreject
        }

    def _compute_stats(self):
        qtiles = {}
        for _qtile in [100 * (1 - self.alpha), 50, 100 * self.alpha]:
            qtiles[_qtile] = {
                _method: np.percentile(self.nreject[_method], _qtile, axis=0)
                for _method in self.nreject
            }
        return qtiles

    def _correct_pvals(self, method: str = "n"):
        _pval_cr = []
        for jdx in range(self.ks_pval.shape[0]):
            _pval_cr.append(
                smm.fdrcorrection(
                    self.ks_pval[jdx].flatten(),
                    alpha=self.alpha,
                    method=method,
                    is_sorted=False,
                )[1].reshape(self.ks_pval[jdx].shape)
            )
        return np.array(_pval_cr)

    def plot_bars(self, qtile: str = "uq", file_ext: str = "png"):
        if qtile.lower() in ["uq", "upper", "u"]:
            _qtile = 100 * (1 - self.alpha)
        elif qtile.lower() in ["lq", "lower", "l"]:
            _qtile = 100 * self.alpha
        else:
            _qtile = 50

        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        width = 0.333
        mult = 0

        for name, nreject in self.reject_qtiles[_qtile].items():
            offset = width * mult
            rect = axes[0].bar(
                self.time_step + offset,
                nreject,
                width=width * 0.9,
                label=self.methods[name],
            )
            axes[0].bar_label(rect, padding=3, color=rect[-1].get_facecolor())
            mult += 1

        axes[0].bar_label(rect, padding=3)
        axes[0].axhline(
            _THR[self.alpha],
            color="#600",
            ls="--",
            label=f"Failure thr [uncorrected]",
            zorder=1,
        )
        axes[0].axhline(
            1,
            color="#060",
            ls="-.",
            label=f"Failure thr [corrected]",
            zorder=1,
        )
        axes[0].legend()

        axes[0].set_title(
            f"Number of variables rejected at {(1 - self.alpha) * 100}% confidence"
        )
        axes[0].set_xlabel("Simulated month")
        axes[0].set_ylabel("N variables")

        mult = 0
        for name, itest in self.ntests.items():
            offset = width * mult
            rect = axes[1].bar(
                self.time_step + offset,
                itest,
                width=width * 0.9,
                label=self.methods[name],
            )
            axes[1].bar_label(
                rect, padding=3, color=rect[-1].get_facecolor(), zorder=10
            )
            mult += 1

        axes[1].axhline(
            self.alpha * self.ks_pval.shape[0],
            color="#343",
            ls="-.",
            label=f"{self.alpha * 100}% of tests",
            zorder=1,
        )
        axes[1].legend()

        for _ax in axes:
            _ax.set_xticks(
                self.time_step,
                [f"{xi}-{xi + 11}" for xi in range(1, len(self.time_step) + 1)],
            )

        axes[1].set_xlabel("Simulated month")

        axes[1].set_title(f'Number of tests (of {self.ks_pval.shape[0]}) "failing"')

        _reject = f"{self.alpha:.2f}".replace(".", "p")
        fig.suptitle(f"{self.cases[0]} x {self.cases[1]}")
        plt.tight_layout()
        plt.savefig(f"plt_{self.cases[0]}-{self.cases[1]}_n{self.n_iter}.{file_ext}")


def main():

    run_len = "1year"
    rolling = 12
    niter = 1000

    pcts = {"effgw_oro": [0.5, 1, 5, 10, 20, 30, 40, 50], "clubb_c1": [1, 3, 5, 10]}
    params_hum = {"effgw_oro": "GW Orog", "clubb_c1": "CLUBB C1"}

    # Initialize list of case with control vs. control test
    cases = [("ctl", "ctl")]
    cases_hum = [
        [
            "Control",
        ]
        * 2
    ]
    case_data = {}
    _casefile = "bootstrap_output.{runlen}_{rolling}avg.{case[0]}_{case[1]}_n{niter}.nc"

    # Add the cases for all the tested parameters
    for _param in pcts:
        case_data[_param] = []

        for _pct in pcts[_param]:
            _pct_str = f"{_pct:.1f}".replace(".", "p")
            cases.append(("ctl", f"{_param}-{_pct_str}pct"))
            if _pct < 1:
                cases_hum.append(("Control", f"{params_hum[_param]} {_pct:.1f}%"))
            else:
                cases_hum.append(("Control", f"{params_hum[_param]} {_pct:.0f}%"))

            _file = Path(
                "bootstrap_data",
                _casefile.format(
                    runlen=run_len, rolling=rolling, case=cases[-1], niter=niter
                ),
            )
            case_data[_param].append(
                CaseData(_file, *cases[-1], _THR[ALPHA], ALPHA, pct_change=_pct)
            )

    # files = [
    #     Path(
    #         "bootstrap_data/bootstrap_output.{}_{}avg.{}_{}_n{}.nc".format(
    #             run_len, rolling, *case, niter
    #         )
    #     )
    #     for case in cases
    # ]

    # assert all([_file.exists() for _file in files]), f"FILES MISSING: {files}"

    # print("[")
    # for _file in files:
    #     print(f"\t{_file.name:>60s}\t\t{_file.exists()}")
    # print("]")

    fig_dpi = 300
    fig_width = 12.5 / 2.54
    fig_height = fig_width / 3.75
    fig_extn = "pdf"

    ctl_file = Path(
        "bootstrap_data",
        _casefile.format(runlen=run_len, rolling=rolling, case=cases[0], niter=niter),
    )
    case_data["ctl"] = CaseData(ctl_file, *cases[0], _THR[ALPHA], ALPHA)

    # case_data = [
    #     CaseData(files[idx], case_a, case_b, _THR[ALPHA], ALPHA)
    #     for idx, (case_a, case_b) in enumerate(cases)
    # ]

    return case_data


def reject_qtiles(case_data: list, idx: int = 0, qtile: float = 50):
    _data = pd.DataFrame(
        [
            {
                _method: _case.reject_qtiles[qtile][_method][idx]
                for _method in _case.reject_qtiles[qtile]
            }
            for _case in case_data
        ]
    )
    _data["pct_change"] = [_case.pct_change for _case in case_data]
    return _data


def ntests(case_data: list, idx: int = 0):
    _data = pd.DataFrame(
        [
            {_method: _case.ntests[_method][idx] for _method in _case.ntests}
            for _case in case_data
        ]
    )
    _data["pct_change"] = [_case.pct_change for _case in case_data]
    return _data


if __name__ == "__main__":
    case_data = main()

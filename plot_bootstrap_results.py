#!/usr/bin/env python
# coding: utf-8
"""Plot results from K-S test bootstrapping, perform FDR Corrections.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.ticker import ScalarFormatter
from statsmodels.stats import multitest as smm

plt.style.use("default")

REJ_THR = {0.01: 6, 0.05: 12}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--alpha", default=0.05, type=float, help="Significance level")
    parser.add_argument("--ext", default="png", type=str, help="Image file extension")
    return parser.parse_args()


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
    methods = {
        "uncor": "Un-corrected",
        "fdr_bh": "Corr [B-H]",
        # "fdr_by": "Corr [B-Y]",
    }

    def __init__(
        self,
        btsp_file: Path,
        case_a: str,
        case_b: str = "",
        uc_thr: int = 11,
        alpha: float = 0.05,
        pct_change: float = 0,
    ):
        _corr_methods = [
            "fdr_bh",
            # "fdr_by",
            # "bonferroni",
            # "fdr_tsbh",
            # "fdr_tsbky",
        ]
        self.alpha = alpha
        if case_b == "":
            case_b = case_a
        self.pct_change = pct_change
        self.cases = (case_a, case_b)
        self.n_iter = int(btsp_file.stem.split("_")[-1][1:])
        _ks_res = xr.open_dataset(btsp_file)
        self.time_step = np.arange(_ks_res.time.shape[0])

        self.ks_pval = _ks_res["pval"].values
        self.pval_cr = {}
        self.pval_cr = {
            _method: self._correct_pvals(_method) for _method in _corr_methods
        }
        self.nreject = {"uncor": reject_vars(self.ks_pval, self.alpha)}

        for _method in _corr_methods:
            self.nreject[_method] = reject_vars(self.pval_cr[_method])

        self.thrs = {"uncor": uc_thr}
        for _method in _corr_methods:
            self.thrs[_method] = 0

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

    def _correct_pvals(self, method: str = "fdr_bh"):
        _pval_cr = []
        for jdx in range(self.ks_pval.shape[0]):
            for kdx in range(self.ks_pval.shape[-1]):
                _pval_cr.append(
                    smm.multipletests(
                        pvals=self.ks_pval[jdx, :, kdx],
                        alpha=self.alpha,
                        method=method,
                        is_sorted=False,
                    )[1]
                )

        return np.array(_pval_cr).reshape(self.ks_pval.shape)

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
            REJ_THR[self.alpha],
            color="#600",
            ls="--",
            label="Failure thr [uncorrected]",
            zorder=1,
        )
        axes[0].axhline(
            1,
            color="#060",
            ls="-.",
            label="Failure thr [corrected]",
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


def plot_rej_vars(
    case_data: dict, idx: int, qtile: float, params: dict, fig_spec: dict, style: dict
):
    """
    Plot the number of rejected variables

    Parameters
    ----------
    case_data : list
        _description_
    idx : int
        _description_
    qtile : float
        _description_
    params : dict
        _description_

    """
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * fig_spec["width"], fig_spec["width"]), dpi=fig_spec["dpi"]
    )

    for ixp, _param in enumerate(params):
        _nvars = reject_qtiles(case_data[_param], idx, qtile)
        axis = axes[ixp]
        for ixm, _method in enumerate(case_data[_param][0].methods):
            _method_hum = case_data[_param][0].methods[_method]
            axis.plot(
                _nvars["pct_change"],
                _nvars[_method],
                color=style["colors"][ixp],
                linestyle=style["lstyle"][ixm % 3],
                marker=style["markers"][ixm],
                label=f"{params[_param]}: {_method_hum} {qtile:.0f}%",
            )

        axis.legend(fontsize=style["legend_fontsize"])

        # Put a horizontal line for the test failure threshold
        axis.axhline(
            1,
            color="k",
            ls="--",
            lw=style["linewidth"],
            label="Global test failure threshold [cor]",
        )
        axis.axhline(
            REJ_THR[case_data[_param][0].alpha],
            color="k",
            ls="-",
            lw=style["linewidth"],
            label="Global test failure threshold [unc]",
        )

        axis.set_xlabel("Parameter Change [%]", fontsize=style["label_fontsize"])
        axis.set_ylabel(
            f"Number of variables\nrejected [p < {case_data[_param][0].alpha:.2f}]",
            fontsize=style["label_fontsize"],
        )
        axis.legend(fontsize=style["legend_fontsize"])
        style_axis(axis, style)

    fig.tight_layout()
    fig.savefig(f"plt_nrej_vars_{idx}_q{qtile:.0f}.{fig_spec['ext']}")


def plot_tests_failed(
    case_data: dict, idx: int, params: dict, fig_spec: dict, style: dict
):
    """
    Plot the number of rejected variables

    Parameters
    ----------
    case_data : list
        _description_
    idx : int
        _description_
    params : dict
        _description_

    """
    fig, axis = plt.subplots(
        1,
        2,
        figsize=(2 * fig_spec["width"], fig_spec["width"]),
        dpi=fig_spec["dpi"],
        sharey=True,
    )
    axis = axis.flatten()

    for ixp, _param in enumerate(params):
        _ntests = ntests(case_data[_param], idx)
        for ixm, _method in enumerate(case_data[_param][0].methods):
            _method_hum = case_data[_param][0].methods[_method]
            axis[ixp].semilogy(
                _ntests["pct_change"],
                _ntests[_method],
                color=style["colors"][ixp],
                linestyle=style["lstyle"][ixm % 3],
                marker=style["markers"][ixm],
                label=f"{params[_param]}: {_method_hum}",
            )
        axis[ixp].set_ylim([10, 1500])
        axis[ixp].legend(fontsize=style["legend_fontsize"])

        # Put a horizontal line for alpha*niter% of bootstrap iterations
        _alphapct = case_data[_param][0].alpha * case_data[_param][0].n_iter
        axis[ixp].axhline(
            _alphapct,
            color="k",
            ls="--",
            lw=style["linewidth"],
            label=f"{case_data[_param][0].alpha*100:.0f}% of tests ({_alphapct:.0f})",
        )

        axis[ixp].set_xlabel("Parameter Change [%]", fontsize=style["label_fontsize"])
        if ixp == 0:
            axis[ixp].set_ylabel(
                (
                    f"Tests with global\n"
                    f"significance [p < {case_data[_param][0].alpha:.2f}]"
                ),
                fontsize=style["label_fontsize"],
            )
        axis[ixp].legend(fontsize=style["legend_fontsize"])

        for _ax in [axis[ixp].xaxis, axis[ixp].yaxis]:
            _ax.set_major_formatter(ScalarFormatter(useOffset=True))

        style_axis(axis[ixp], style)

    fig.tight_layout()
    _alphastr = f"{case_data[_param][0].alpha:.02f}".replace(".", "p")
    fig.savefig(f"plt_nfailed_tests_{idx}_a{_alphastr}.{fig_spec['ext']}")


def style_axis(_ax, style):
    _ax.grid(visible=True, ls="dotted", lw=style["linewidth"], color="grey")
    sns.despine(ax=_ax, offset=10)
    _ax.tick_params(labelsize=style["tick_fontsize"])


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


def main(args):

    run_len = "1year"
    rolling = 12
    niter = 1000
    alpha = args.alpha
    assert 0.0 < alpha < 1.0, f"ALPHA {alpha} not in [0, 1]"

    pcts = {"effgw_oro": [1, 10, 20, 30, 40, 50], "clubb_c1": [1, 3, 5, 10]}
    params_hum = {"effgw_oro": "GW Orog", "clubb_c1": "CLUBB C1"}

    # Initialize list of case with control vs. control test
    cases = []
    cases_hum = []
    case_data = {}
    _casefile = "bootstrap_output.{runlen}_{rolling}avg.{case[0]}_{case[1]}_n{niter}.nc"

    # Add the cases for all the tested parameters
    for _param in pcts:
        case_data[_param] = []

        for _pct in pcts[_param]:
            _pct_str = f"{_pct:.1f}".replace(".", "p")
            _case = ("ctl", f"{_param}-{_pct_str}pct")
            cases.append(_case)

            if _pct < 1:
                _casehum = ("Control", f"{params_hum[_param]} {_pct:.1f}%")
            else:
                _casehum = ("Control", f"{params_hum[_param]} {_pct:.0f}%")
            cases_hum.append(_casehum)

            _file = Path(
                "bootstrap_data",
                _casefile.format(
                    runlen=run_len, rolling=rolling, case=cases[-1], niter=niter
                ),
            )
            case_data[_param].append(
                CaseData(_file, *_case, REJ_THR[alpha], alpha, pct_change=_pct)
            )

    fig_width = 12.5 / 2.54
    fig_spec = {
        "dpi": 300,
        "width": fig_width,
        "height": fig_width / 3.75,
        "ext": args.ext,
    }

    _case = ("ctl", "ctl")
    cases.append(_case)
    _casehum = ("Control", "Control")
    cases_hum.append(_casehum)
    ctl_file = Path(
        "bootstrap_data",
        _casefile.format(runlen=run_len, rolling=rolling, case=cases[0], niter=niter),
    )

    case_data["ctl"] = CaseData(ctl_file, *_case, REJ_THR[alpha], alpha)
    style = {
        "linewidth": 1.0,
        "label_fontsize": 12,
        "legend_fontsize": 8,
        "tick_fontsize": 10,
        "colors": ["C1", "C4", "C6"],
        "lstyle": ["-", "--", "-."],
        "markers": ["o", "x", "+", "h", ".", "*"],
    }
    for _ti in [0, 1, 2]:
        plot_rej_vars(
            case_data,
            _ti,
            100 * (1 - alpha),
            params=params_hum,
            fig_spec=fig_spec,
            style=style,
        )
        plot_tests_failed(
            case_data, _ti, params=params_hum, fig_spec=fig_spec, style=style
        )

    return case_data, params_hum, fig_spec


if __name__ == "__main__":
    case_data = main(parse_args())

#!/usr/bin/env python
# coding: utf-8
"""Plot results from K-S test bootstrapping, perform FDR Corrections."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.ticker import ScalarFormatter
from statsmodels.stats import multitest as smm

import detclim

plt.style.use("default")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--alpha", default=0.05, type=float, help="Significance level")
    parser.add_argument("--ext", default="png", type=str, help="Image file extension")
    parser.add_argument(
        "--vert", action="store_true", default=False, help="Orient figures in vertical"
    )
    parser.add_argument(
        "--test-size", default=30, type=int, help="Ensemble size, default=30"
    )
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
        "uncor": "MVK",
        "fdr_bh": "B-H FDR",
        "fdr_by": "B-Y FDR",
        "bonferroni": "Bonf. FDR",
    }
    stests = {"ks": "K-S", "cvm": "CVM", "mw": "M-W", "wsr": "WSR"}

    def __init__(
        self,
        btsp_file: Path,
        case_a: str,
        case_b: str = "",
        alpha: float = 0.05,
        pct_change: float = 0,
        esize: int = 30,
    ):
        _corr_methods = [
            "fdr_bh",
            "fdr_by",
            "bonferroni",
            # "fdr_tsbh",
            # "fdr_tsbky",
        ]
        self.alpha = alpha
        if case_b == "":
            case_b = case_a
        self.pct_change = pct_change
        self.cases = (case_a, case_b)
        self.n_iter = int(btsp_file.stem.split("_")[-1][1:])
        _ks_res = xr.open_dataset(btsp_file).load()
        self.time_step = np.arange(_ks_res.time.shape[0])
        self.esize = esize

        self.ks_pval = _ks_res["cvm"].values
        self.pvals = {_test: _ks_res[_test].values for _test in self.stests}

        self.pval_cr = {
            _test: {
                _method: self._correct_pvals(_test, _method)
                for _method in _corr_methods
            }
            for _test in self.stests
        }
        # self.nreject = {"uncor": reject_vars(self.ks_pval, self.alpha)}
        self.nreject = {
            _test: {"uncor": reject_vars(self.pvals[_test], self.alpha)}
            for _test in self.stests
        }
        for _test in self.stests:
            for _method in _corr_methods:
                self.nreject[_test][_method] = reject_vars(self.pval_cr[_test][_method])

        self.thrs = {"uncor": detclim.REJ_THR[alpha][self.esize]}
        for _method in _corr_methods:
            self.thrs[_method] = {_stest: 0 for _stest in self.stests}

        self.reject_qtiles = self._compute_stats()

        # Number of failed tests based on threshold, set by control run for
        # un-corrected, 0 for corrected
        self.ntests = {
            _test: {
                _method: (self.nreject[_test][_method] > self.thrs[_method][_test]).sum(
                    axis=0
                )
                for _method in self.nreject[_test]
            }
            for _test in self.stests
        }

    def _compute_stats(self):
        qtiles = {}
        for _test in self.stests:
            qtiles[_test] = {}
            for _qtile in [100 * (1 - self.alpha), 50, 100 * self.alpha]:
                qtiles[_test][_qtile] = {
                    _method: np.percentile(self.nreject[_test][_method], _qtile, axis=0)
                    for _method in self.nreject[_test]
                }
        return qtiles

    def _correct_pvals(self, stest: str = "ks", method: str = "fdr_bh"):
        _pval_cr = []
        for jdx in range(self.pvals[stest].shape[0]):
            for kdx in range(self.pvals[stest].shape[-1]):
                _pval_cr.append(
                    smm.multipletests(
                        pvals=self.pvals[stest][jdx, :, kdx],
                        alpha=self.alpha,
                        method=method,
                        is_sorted=False,
                    )[1]
                )

        return np.array(_pval_cr).reshape(self.pvals[stest].shape)

    def plot_bars(
        self,
        qtile: str = "uq",
        file_ext: str = "png",
        vert: bool = False,
        stest: str = "ks",
    ):
        stest = "ks"
        if qtile.lower() in ["uq", "upper", "u"]:
            _qtile = 100 * (1 - self.alpha)
        elif qtile.lower() in ["lq", "lower", "l"]:
            _qtile = 100 * self.alpha
        else:
            _qtile = 50

        if vert:
            fig, axes = plt.subplots(2, 1, figsize=(4, 15))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 4))

        width = 0.333
        mult = 0

        for name, nreject in self.reject_qtiles[stest][_qtile].items():
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
            detclim.REJ_THR[self.alpha][self.esize][stest],
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
        for name, itest in self.ntests[stest].items():
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
            self.alpha * self.pvals[stest].shape[0],
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

        axes[1].set_title(
            f'Number of tests (of {self.pvals[stest].shape[0]}) "failing"'
        )

        _reject = f"{self.alpha:.2f}".replace(".", "p")
        fig.suptitle(f"{self.cases[0]} x {self.cases[1]}")
        plt.tight_layout()
        plt.savefig(
            f"plt_{self.cases[0]}-{self.cases[1]}_{stest}_n{self.n_iter}_ts{self.esize}.{file_ext}"
        )


def plot_rej_vars(
    case_data: dict,
    idx: int,
    qtile: float,
    params: dict,
    fig_spec: dict,
    style: dict,
    stest: str,
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
    nparams = len(params)
    if "vert" in fig_spec["orient"].lower():
        fig, axes = plt.subplots(
            nparams,
            1,
            figsize=(fig_spec["width"], nparams * fig_spec["width"]),
            dpi=fig_spec["dpi"],
        )
    else:
        fig, axes = plt.subplots(
            1,
            nparams,
            figsize=(nparams * fig_spec["width"], fig_spec["width"]),
            dpi=fig_spec["dpi"],
        )

    for ixp, _param in enumerate(params):
        _nvars = reject_qtiles(case_data[_param], idx, qtile)[stest]
        axis = axes[ixp]
        for ixm, _method in enumerate(case_data[_param][0].methods):
            _method_hum = case_data[_param][0].methods[_method]
            axis.plot(
                _nvars["pct_change"],
                _nvars[_method],
                color=style["colors"][ixp],
                linestyle=style["lstyle"][ixm % 4],
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
            label="Global test failure threshold [MVK]",
        )
        _case0 = case_data[_param][0]
        axis.axhline(
            detclim.REJ_THR[_case0.alpha][_case0.esize][stest],
            color="k",
            ls="-",
            lw=style["linewidth"],
            label="Global test failure threshold [B-H FDR]",
        )

        if "vert" in fig_spec["orient"].lower():
            x_check = ixp == len(params) - 1
            y_check = True
        else:
            x_check = True
            y_check = ixp == 0

        if x_check:
            axis.set_xlabel("Parameter Change [%]", fontsize=style["label_fontsize"])

        if y_check:
            axis.set_ylabel(
                f"Number of variables\nrejected [p < {case_data[_param][0].alpha:.2f}]",
                fontsize=style["label_fontsize"],
            )
        axis.legend(fontsize=style["legend_fontsize"])
        style_axis(axis, style)

    fig.tight_layout()
    _esize = case_data[_param][0].esize
    fig.savefig(
        f"plt_nrej_vars_{stest}_{idx}_q{qtile:.0f}_ts{_esize}.{fig_spec['ext']}"
    )


def plot_tests_failed(
    case_data: dict,
    idx: int,
    params: dict,
    fig_spec: dict,
    style: dict,
    stest: str,
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
    nparams = len(params)
    # Horizontal orientation
    if "horiz" in fig_spec["orient"].lower():
        fig, axis = plt.subplots(
            1,
            nparams,
            figsize=(nparams * fig_spec["width"], fig_spec["width"]),
            dpi=fig_spec["dpi"],
            sharey=True,
        )
    # Vertical orientation
    else:
        fig, axis = plt.subplots(
            nparams,
            1,
            figsize=(fig_spec["width"], nparams * fig_spec["width"]),
            dpi=fig_spec["dpi"],
            sharey=True,
        )
    axis = axis.flatten()

    for ixp, _param in enumerate(params):
        _ntests = ntests(case_data[_param], idx)[stest]
        for ixm, _method in enumerate(case_data[_param][0].methods):
            _method_hum = case_data[_param][0].methods[_method]
            axis[ixp].semilogy(
                _ntests["pct_change"],
                _ntests[_method],
                color=style["colors"][ixp],
                linestyle=style["lstyle"][ixm % 4],
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
            label=f"{case_data[_param][0].alpha * 100:.0f}% of tests ({_alphapct:.0f})",
        )

        if "vert" in fig_spec["orient"].lower():
            x_check = ixp == len(params) - 1
            y_check = True
        else:
            x_check = True
            y_check = ixp == 0

        if x_check:
            axis[ixp].set_xlabel(
                "Parameter Change [%]", fontsize=style["label_fontsize"]
            )

        if y_check:
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
    _esize = case_data[_param][0].esize
    fig.savefig(
        f"plt_nfailed_tests_{stest}_{idx}_a{_alphastr}_ts{_esize}.{fig_spec['ext']}"
    )


def style_axis(_ax, style):
    _ax.grid(visible=True, ls="dotted", lw=style["linewidth"], color="grey")
    sns.despine(ax=_ax, offset=10)
    _ax.tick_params(labelsize=style["tick_fontsize"])


def reject_qtiles(case_data: list, idx: int = 0, qtile: float = 50):
    _data = {
        _test: pd.DataFrame(
            [
                {
                    _method: _case.reject_qtiles[_test][qtile][_method][idx]
                    for _method in _case.reject_qtiles[_test][qtile]
                }
                for _case in case_data
            ]
        )
        for _test in case_data[0].stests
    }
    for _test in case_data[0].stests:
        _data[_test]["pct_change"] = [_case.pct_change for _case in case_data]
    return _data


def ntests(case_data: list, idx: int = 0):
    _data = {
        _test: pd.DataFrame(
            [
                {
                    _method: _case.ntests[_test][_method][idx]
                    for _method in _case.ntests[_test]
                }
                for _case in case_data
            ]
        )
        for _test in case_data[0].stests
    }
    for _test in case_data[0].stests:
        _data[_test]["pct_change"] = [_case.pct_change for _case in case_data]
    return _data


def data_extract(case_data, qtile=95.0, t_idx=-1):
    """Convert list of case data into data for plotting all statstical test results."""
    nfailed = {}
    nrej_vars = {}
    pct_change = {}
    for param in case_data:
        if "ctl" not in param:
            nfailed[param] = {}
            nrej_vars[param] = {}
            pct_change[param] = []
            for _case in case_data[param]:
                pct_change[param].append(_case.pct_change)
                for stest in _case.ntests:
                    nfailed[param][stest] = {}
                    nrej_vars[param][stest] = {}
                    for method in _case.ntests[stest]:
                        nfailed[param][stest][method] = [
                            _case.ntests[stest][method][t_idx]
                            for _case in case_data[param]
                        ]

                        nrej_vars[param][stest][method] = [
                            _case.reject_qtiles[stest][qtile][method][t_idx]
                            for _case in case_data[param]
                        ]
    return nfailed, nrej_vars, pct_change


def plot_failed_tests_all(
    nfailed, pct_change, params_hum, fig_spec, style, esize=30, niter=1000, alpha=0.05
):
    ornl_colours = {
        "green": "#007833",
        "bgreen": "#84b641",
        "orange": "#DE762D",
        "teal": "#1A9D96",
        "red": "#88332E",
        "blue": "#5091CD",
        "gold": "#FECB00",
    }
    colors = {
        "ks": ornl_colours["green"],
        "K-S": ornl_colours["green"],
        "cvm": ornl_colours["blue"],
        "C-VM": ornl_colours["blue"],
        "wsr": ornl_colours["red"],
        "mw": ornl_colours["orange"],
        "M-W": ornl_colours["orange"],
        "C-VM, M-W": ornl_colours["orange"],
        "BH-FDR": "black",
    }

    if "horiz" in fig_spec["orient"].lower():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    else:
        _width = 12 / 2.54
        _height = _width * 2.1
        fig, axes = plt.subplots(3, 1, figsize=(_width, _height), sharey=True, dpi=300)

    dframes = []
    _failed_label = f"Tests with global\nsignificance [p < {alpha:.2f}]"
    for _param in nfailed:
        dframes.append([])
        for _stest in nfailed[_param]:
            for _mode in nfailed[_param][_stest]:
                _df = pd.DataFrame({_failed_label: nfailed[_param][_stest][_mode]})
                _df["Parameter Change [%]"] = pct_change[_param]
                _df["Method"] = detclim.METHOD_SHORT.get(_mode, _mode)
                _df["Statistical test"] = detclim.STESTS_SHORT.get(_stest, _stest)
                dframes[-1].append(_df)
        dframes[-1] = pd.concat(dframes[-1])
        dframes[-1]["param"] = _param
    dframes = pd.concat(dframes)

    dframes = dframes.query("`Statistical test` != 'WSR'").query(
        "Method == 'Uncor.' or Method == 'FDR-BH'"
    )

    for idx, param in enumerate(nfailed):
        _dframe = dframes.query("param == @param")
        if idx == len(nfailed) - 1:
            legend = "auto"
        else:
            legend = False
        sns.set_theme(
            context="paper",
            style="whitegrid",
            font="sans-serif",
            rc={"lines.linewidth": 1.5, "font.size": 8},
        )
        _plt = sns.lineplot(
            data=_dframe,
            x="Parameter Change [%]",
            y=_failed_label,
            hue="Statistical test",
            style="Method",
            palette=colors,
            markers=True,
            ax=axes[idx],
            legend=legend,
        )
        # if legend:
        # plt.setp(_plt.get_legend().get_texts(), fontsize='8')
        # plt.setp(_plt.get_legend().get_title(), fontsize='10')
        # plt.setp(_plt.yaxis.get_label(), fontsize="8")
        # plt.setp(_plt.xaxis.get_label(), fontsize="8")

        axes[idx].axhline(
            alpha * niter,
            ls="-.",
            color="k",
            lw=style["linewidth"],
            label=f"{alpha * 100}% of tests",
        )
        axes[idx].set_title(f"{params_hum[param]}", fontsize=10)
        axes[idx].set_yscale("log")
        axes[idx].set_ylim([10, 1500])
        for _ax in [axes[idx].xaxis, axes[idx].yaxis]:
            _ax.set_major_formatter(ScalarFormatter(useOffset=True))

    for axis in axes:
        style_axis(axis, style)
    # axes[-1].legend()
    fig.tight_layout()
    _alphastr = f"{alpha:.02f}".replace(".", "p")
    fig.savefig(f"plt_nfailed_tests_a{_alphastr}_ts{esize}.{fig_spec['ext']}")

    _failed_label_diff = f"Additonal {_failed_label.lower()}"
    pwr_diff = (
        dframes.query("Method == 'FDR-BH'")[_failed_label]
        - dframes.query("Method == 'Uncor.'")[_failed_label]
    )
    dframes_diff = dframes.query("Method == 'FDR-BH'").copy()
    dframes_diff[_failed_label] = pwr_diff
    dframes_diff = dframes_diff.rename(columns={_failed_label: _failed_label_diff})

    if "horiz" in fig_spec["orient"].lower():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(_width, _height), sharey=False, dpi=300)

    for idx, param in enumerate(nfailed):
        _dframe = dframes_diff.query("param == @param")
        if idx == len(nfailed) - 1:
            legend = "auto"
        else:
            legend = False

        sns.lineplot(
            data=_dframe,
            x="Parameter Change [%]",
            y=_failed_label_diff,
            hue="Statistical test",
            palette=colors,
            markers="x",
            ax=axes[idx],
            legend=legend,
        )

        axes[idx].axhline(
            0,
            ls="-",
            color="k",
            lw=style["linewidth"],
        )
        axes[idx].set_title(f"{params_hum[param]}", fontsize=8)

    for axis in axes:
        style_axis(axis, style)

    fig.tight_layout()
    _alphastr = f"{alpha:.02f}".replace(".", "p")
    fig.savefig(f"plt_nfailed_tests_diff_a{_alphastr}_ts{esize}.{fig_spec['ext']}")


def plot_rej_vars_all(
    nrej,
    pct_change,
    params_hum,
    fig_spec,
    style,
    esize=30,
    niter=1000,
    alpha=0.05,
    qtile=0.95,
):
    ornl_colours = {
        "green": "#007833",
        "bgreen": "#84b641",
        "orange": "#DE762D",
        "teal": "#1A9D96",
        "red": "#88332E",
        "blue": "#5091CD",
        "gold": "#FECB00",
    }
    colors = {
        "ks": ornl_colours["green"],
        "K-S": ornl_colours["green"],
        "cvm": ornl_colours["blue"],
        "C-VM": ornl_colours["blue"],
        "wsr": ornl_colours["red"],
        "mw": ornl_colours["orange"],
        "M-W": ornl_colours["orange"],
        "C-VM, M-W": ornl_colours["orange"],
        "BH-FDR": "black",
    }
    if "horiz" in fig_spec["orient"].lower():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    else:
        _width = 12 / 2.54
        _height = _width * 2.1
        fig, axes = plt.subplots(3, 1, figsize=(_width, _height), sharey=True, dpi=300)

    dframes = []
    for _param in nrej:
        dframes.append([])
        for _stest in nrej[_param]:
            for _mode in nrej[_param][_stest]:
                _df = pd.DataFrame({"Rejected Fields": nrej[_param][_stest][_mode]})
                _df["Parameter Change [%]"] = pct_change[_param]
                _df["Method"] = detclim.METHOD_SHORT.get(_mode, _mode)
                _df["Statistical test"] = detclim.STESTS_SHORT.get(_stest, _stest)
                dframes[-1].append(_df)
        dframes[-1] = pd.concat(dframes[-1])
        dframes[-1]["param"] = _param
    dframes = pd.concat(dframes)

    dframes = dframes.query("`Statistical test` != 'WSR'").query(
        "Method == 'Uncor.' or Method == 'FDR-BH'"
    )

    for idx, param in enumerate(nrej):
        _dframe = dframes.query("param == @param")
        if idx == len(nrej) - 1:
            legend = "auto"
        else:
            legend = False
        sns.set_theme(
            context="paper",
            style="whitegrid",
            font="sans-serif",
            rc={"lines.linewidth": 1.5, "font.size": 8},
        )
        sns.lineplot(
            data=_dframe,
            x="Parameter Change [%]",
            y="Rejected Fields",
            hue="Statistical test",
            style="Method",
            palette=colors,
            markers=True,
            ax=axes[idx],
            legend=legend,
        )

        for _stest, _thr in {
            "K-S": detclim.REJ_THR[alpha][esize]["ks"],
            "C-VM, M-W": detclim.REJ_THR[alpha][esize]["mw"],
            "BH-FDR": 1,
        }.items():
            axes[idx].axhline(
                _thr,
                ls="-.",
                color=colors[_stest],
                lw=style["linewidth"],
                label=f"Threshold [{_stest}]",
            )
        axes[idx].set_title(f"{params_hum[param]}")
    for axis in axes:
        style_axis(axis, style)
    axes[-1].legend(fontsize=style["legend_fontsize"])
    fig.tight_layout()
    _alphastr = f"{alpha:.02f}".replace(".", "p")
    _qtilestr = f"{qtile:.02f}".replace(".", "p")
    fig.savefig(f"plt_rejvars_a{_alphastr}_ts{esize}_q{_qtilestr}.{fig_spec['ext']}")


def main(args):
    """Parse command line args, make plots."""
    run_len = "1year"
    rolling = 12
    test_size = args.test_size
    niter = 1000
    alpha = args.alpha
    assert 0.0 < alpha < 1.0, f"ALPHA {alpha} not in [0, 1]"

    pcts = {
        "effgw_oro": [1, 10, 20, 30, 40, 50],
        "clubb_c1": [1, 3, 5, 10],
        "zmconv_c0_ocn": [0.5, 1, 3, 5],
    }
    params_hum = {
        "effgw_oro": "GW Orog",
        "clubb_c1": "CLUBB C1",
        "zmconv_c0_ocn": "ZM Conv C0-Ocean",
    }
    # Initialize list of case with control vs. control test
    cases = []
    cases_hum = []
    case_data = {}
    _casefile = (
        "bootstrap_output.{runlen}_{rolling}avg_"
        "ts{test_size}.{case[0]}_{case[1]}_n{niter}.nc"
    )

    _time_s = time.perf_counter()
    # Add the cases for all the tested parameters
    for _param, _pcts in pcts.items():
        case_data[_param] = []

        for _pct in _pcts:
            _pct_str = f"{_pct:.1f}".replace(".", "p")
            if "zmconv" in _param:
                _ctlcase = "ctl-miller"
            else:
                _ctlcase = "ctl"
            _case = (_ctlcase, f"{_param}-{_pct_str}pct")
            cases.append(_case)

            if _pct < 1:
                _casehum = ("Control", f"{params_hum[_param]} {_pct:.1f}%")
            else:
                _casehum = ("Control", f"{params_hum[_param]} {_pct:.0f}%")
            cases_hum.append(_casehum)

            _file = Path(
                "bootstrap_data",
                _casefile.format(
                    runlen=run_len,
                    rolling=rolling,
                    test_size=test_size,
                    case=cases[-1],
                    niter=niter,
                ),
            )
            case_data[_param].append(CaseData(_file, *_case, alpha, pct_change=_pct))

    print(f"{'loaded case data in':22s} {time.perf_counter() - _time_s:.2f}")

    fig_width = 12.5 / 2.54
    fig_spec = {
        "dpi": 300,
        "width": fig_width,
        "height": fig_width / 3.75,
        "ext": args.ext,
    }
    if args.vert:
        fig_spec["orient"] = "vert"
    else:
        fig_spec["orient"] = "horiz"

    _case = ("ctl", "ctl")
    cases.append(_case)
    _casehum = ("Control", "Control")
    cases_hum.append(_casehum)
    ctl_file = Path(
        "bootstrap_data",
        _casefile.format(
            runlen=run_len,
            rolling=rolling,
            test_size=test_size,
            case=cases[0],
            niter=niter,
        ),
    )

    case_data["ctl"] = CaseData(ctl_file, *_case, alpha)
    style = {
        "linewidth": 1.0,
        "label_fontsize": 12,
        "legend_fontsize": 7,
        "tick_fontsize": 10,
        "colors": ["C1", "C4", "C6"],
        "lstyle": ["-", "--", "-.", ":"],
        "markers": ["o", "x", "+", "h", ".", "*"],
    }

    _time_s = time.perf_counter()
    qtile = 5.0
    nfailed, nrej_vars, pct_change = data_extract(case_data, qtile=qtile, t_idx=-1)
    print(f"{'extract data in':22s} {time.perf_counter() - _time_s:.2f}")

    _time_s = time.perf_counter()
    plot_failed_tests_all(
        nfailed,
        pct_change,
        params_hum,
        fig_spec,
        style,
        esize=test_size,
        niter=1000,
        alpha=0.05,
    )
    print(f"{'plot failed tests in':22s} {time.perf_counter() - _time_s:.2f}")

    _time_s = time.perf_counter()
    plot_rej_vars_all(
        nrej_vars,
        pct_change,
        params_hum,
        fig_spec,
        style,
        esize=test_size,
        niter=1000,
        alpha=0.05,
        qtile=qtile,
    )
    print(f"{'plot rej vars in':22s} {time.perf_counter() - _time_s:.2f}")
    # for _ti in [0, 1, 2]:
    # _ti = 2

    # for stest in case_data["ctl"].stests:
    #     print(f"plot {stest}")
    #     plot_rej_vars(
    #         case_data,
    #         _ti,
    #         100 * (1 - alpha),
    #         params=params_hum,
    #         fig_spec=fig_spec,
    #         style=style,
    #         stest=stest,
    #     )
    #     plot_tests_failed(
    #         case_data,
    #         _ti,
    #         params=params_hum,
    #         fig_spec=fig_spec,
    #         style=style,
    #         stest=stest,
    #     )

    return case_data, params_hum, fig_spec


if __name__ == "__main__":
    all_case_data, _, _ = main(parse_args())

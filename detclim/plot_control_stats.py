#!/usr/bin/env python
# coding: utf-8
"""Create boxplot of control simulations, and bar graph of false positives."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from statsmodels.stats import multitest as smm

import detclim
from detclim.results_plot import fmt_case

plt.style.use("default")


ALPHA = 0.05


def correct_pvals(ks_pval, alpha=0.05, method: str = "fdr_bh"):
    _pval_cr = []
    for jdx in range(ks_pval.shape[0]):
        for kdx in range(ks_pval.shape[-1]):
            _pval_cr.append(
                smm.multipletests(
                    pvals=ks_pval[jdx, :, kdx],
                    alpha=alpha,
                    method=method,
                    is_sorted=False,
                )[1]
            )

    return np.array(_pval_cr).reshape(ks_pval.shape)


def compute_cases(cases, files, stest="ks"):
    ks_pval_cr = {}

    n_reject = {}
    reject_test = {}

    n_reject_cr = {}
    reject_test_cr = {}
    rejections = {}

    for _ix, _file in enumerate(files):
        case_a, case_b = cases[_ix]
        # n_iter = int(_file.stem.split("_")[-1][1:])
        case_a = fmt_case(case_a)
        case_b = fmt_case(case_b)
        ks_res = xr.open_dataset(_file)
        ks_pval = ks_res[stest].values

        # fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        quantile = ALPHA * 100
        # time_step = np.arange(ks_res.time.shape[0])

        # n_reject = np.array((ks_pval < ALPHA).sum(axis=1))
        n_reject[(case_a, case_b)] = np.array((ks_pval < ALPHA).sum(axis=1))

        # n_reject_mean = np.median(n_reject[(case_a, case_b)], axis=0)
        # n_reject_lq = np.percentile(n_reject[(case_a, case_b)], quantile, axis=0)
        n_reject_uq = np.percentile(n_reject[(case_a, case_b)], 100 - quantile, axis=0)

        ks_pval_cr[(case_a, case_b)] = correct_pvals(ks_pval, alpha=ALPHA)
        n_reject_cr[(case_a, case_b)] = np.array(
            (ks_pval_cr[(case_a, case_b)] < ALPHA).sum(axis=1)
        )
        # n_reject_mean_cr = np.median(n_reject_cr[(case_a, case_b)], axis=0)
        # n_reject_lq_cr = np.percentile(n_reject_cr[(case_a, case_b)], quantile, axis=0)
        n_reject_uq_cr = np.percentile(
            n_reject_cr[(case_a, case_b)], 100 - quantile, axis=0
        )

        rejections[(case_a, case_b)] = {
            f"{100 * (1 - ALPHA)}%": n_reject_uq,
            f"{100 * (1 - ALPHA)}% [Corrected]": n_reject_uq_cr,  # .max(axis=0),
        }
    return rejections, n_reject, n_reject_cr, reject_test, reject_test_cr


def main(stest="ks", test_size: int = 30, draw_plots: bool = False, ext: str = "png"):
    run_len = "1year"
    rolling = 12
    niter = 1000

    cases = [
        ("ctl", "ctl"),
        ("effgw_oro-1p0pct",) * 2,
        ("effgw_oro-10p0pct",) * 2,
        ("effgw_oro-20p0pct",) * 2,
        ("effgw_oro-30p0pct",) * 2,
        ("effgw_oro-40p0pct",) * 2,
        ("effgw_oro-50p0pct",) * 2,
        ("clubb_c1-1p0pct",) * 2,
        ("clubb_c1-3p0pct",) * 2,
        ("clubb_c1-5p0pct",) * 2,
        ("clubb_c1-10p0pct",) * 2,
        ("zmconv_c0_ocn-0p5pct",) * 2,
        ("zmconv_c0_ocn-1p0pct",) * 2,
        ("zmconv_c0_ocn-3p0pct",) * 2,
        ("zmconv_c0_ocn-5p0pct",) * 2,
        ("opt-O1",) * 2,
        ("fastmath",) * 2,
    ]

    data_path = detclim.data_path

    files = [
        Path(
            data_path,
            f"bootstrap_output.{run_len}_{rolling}avg_"
            f"ts{test_size}.{case[0]}_{case[1]}_n{niter}.nc",
        )
        for case in cases
    ]
    rejections, n_reject, n_reject_cr, reject_test, reject_test_cr = compute_cases(
        cases, files, stest=stest
    )

    reject_data = {
        "Case": [],
        "Rejected fields": [],
        "iteration": [],
        "Parameter": [],
        "Median": [],
        "Pct": [],
    }

    time_idx = 1
    for _case, nrej in n_reject.items():
        for idx, nrej in enumerate(n_reject[_case]):
            _pct = _case[0].split(" ")[-1]
            try:
                _pct = float(_pct[:-1])
            except ValueError:
                _pct = 0
            reject_data["Case"].append(_case[0])
            if _case[0] == "opt-O1" or _case[0] == "fastmath":
                reject_data["Parameter"].append("Optimization")
            else:
                reject_data["Parameter"].append(_case[0].split(" ")[0])
            reject_data["Rejected fields"].append(nrej[time_idx])
            reject_data["iteration"].append(idx)
            reject_data["Median"].append(np.median(n_reject[_case][:, time_idx]))
            reject_data["Pct"].append(_pct)

    reject_data_frame = pd.DataFrame(reject_data)

    # if compute_thr:
    ctl_thr = float(
        np.percentile(
            reject_data_frame.groupby("Case")["Rejected fields"].quantile(1 - ALPHA),
            50,
        )
    )
    if draw_plots:
        fig, axis = plt.subplots(1, 1, figsize=(12.5 / 2.54, 6.25 / 2.54), dpi=600)

        with sns.plotting_context(context="paper", font_scale=0.5, rc=None):
            _ = sns.boxplot(
                reject_data_frame,
                orient="h",
                x="Rejected fields",
                y="Case",
                hue="Parameter",
                palette="Set2",
                ax=axis,
                zorder=4,
                linewidth=0.5,
                whis=(5, 95),  # pyright: ignore[reportArgumentType]
                fliersize=1,
            )

            _labels = axis.get_yticklabels()
            _ticks = axis.get_yticks()
            _tmp = []
            _newlabels = []

            for _label in _labels:
                _txt = _label.get_text()
                if _txt.split(" ")[0] not in _tmp:
                    _tmp.append(_txt.split(" ")[0])
                else:
                    _label.set_text(_txt.split(" ")[-1])
                _newlabels.append(_label)
            axis.set_yticks(_ticks)
            axis.set_yticklabels(_newlabels)
            axis.tick_params(labelsize=8)

            axis.axvline(ctl_thr, ls="--", lw=0.6, color="k")

            axis.set_ylabel("")
            axis.set_xlabel(f"Rejected fields at \u03b1={ALPHA:.2f}", fontsize=8)
            axis.grid(visible=True, ls="--", lw=0.2, zorder=0)

            # Add label for 95th %tile
            _txtx = ctl_thr + 1
            _txty = len(cases) * 0.985
            # print(f"ADD THE TEXT AT {_txtx}, {_txty}")
            axis.text(
                float(_txtx),
                _txty,
                f"{ctl_thr:.0f}",
                horizontalalignment="right",
                verticalalignment="top",
                zorder=10,
                fontsize=6,
                bbox=dict(
                    boxstyle="round", edgecolor="white", facecolor="grey", alpha=0.6
                ),
            )
            plt.tight_layout()
            plt.savefig(f"plt_control_{stest}_nrej_ts{test_size}_a{ALPHA}.{ext}")
            plt.close()

    return ctl_thr


if __name__ == "__main__":
    thrs = {}
    for tsize in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
        thrs[tsize] = {}
        print(f"---{tsize}---")
        for stest in ["ks", "cvm", "mw", "wsr"]:
            _thr = main(stest, test_size=tsize)
            thrs[tsize][stest] = int(_thr)
            print(f"  {stest}: {_thr}")
    print(thrs)
    # One more time to get the paper plots
    _thr = main("ks", test_size=30, draw_plots=True, ext="pdf")

#!/usr/bin/env python
# coding: utf-8
"""Create bar plot of false positives."""

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
REJ_THR_PRECOMPUTE = {
    0.01: {"ks": 6, "cvm": 9, "mw": 9, "es": 8},
    0.05: {"ks": 11, "cvm": 16, "mw": 16, "es": 15},
}
STEST_NAMES = {"ks": "K-S", "cvm": "CVM", "mw": "M-W", "es": "E-S"}


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

        quantile = ALPHA * 100
        n_reject[(case_a, case_b)] = np.array((ks_pval < ALPHA).sum(axis=1))
        n_reject_uq = np.percentile(n_reject[(case_a, case_b)], 100 - quantile, axis=0)

        ks_pval_cr[(case_a, case_b)] = correct_pvals(ks_pval, alpha=ALPHA)
        n_reject_cr[(case_a, case_b)] = np.array(
            (ks_pval_cr[(case_a, case_b)] < ALPHA).sum(axis=1)
        )
        n_reject_uq_cr = np.percentile(
            n_reject_cr[(case_a, case_b)], 100 - quantile, axis=0
        )

        rejections[(case_a, case_b)] = {
            f"{100 * (1 - ALPHA)}%": n_reject_uq,
            f"{100 * (1 - ALPHA)}% [Corrected]": n_reject_uq_cr,  # .max(axis=0),
        }
    return rejections, n_reject, n_reject_cr, reject_test, reject_test_cr


def main(test_size=30, make_plots=False, ext: str = "png"):
    run_len = "1year"
    rolling = 12
    niter = 1000
    ctl_thr = detclim.REJ_THR[ALPHA][test_size]

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

    data_path = Path(*detclim.data_path.parts[:-1], "bootstrap_data_ctl")

    files = [
        Path(
            data_path,
            f"bootstrap_output.{run_len}_{rolling}avg_"
            f"ts{test_size}.{case[0]}_{case[1]}_n{niter}.nc",
        )
        for case in cases
    ]

    all_fps = []

    for stest in ["ks", "mw", "cvm"]:  # , "es"]:
        _, n_reject, n_reject_cr, _, _ = compute_cases(cases, files, stest=stest)

        false_positives = {
            _key[0]
            .replace("%", r"\%")
            .replace("_", " "): np.sum(n_reject[_key][:, :] > ctl_thr[stest], axis=0)
            for _key in n_reject
        }
        false_positives_cr = {
            _key[0]
            .replace("%", r"\%")
            .replace("_", " "): np.sum(n_reject_cr[_key][:, :] > 0, axis=0)
            for _key in n_reject_cr
        }

        false_positives = pd.DataFrame(false_positives)
        false_positives_cr = pd.DataFrame(false_positives_cr)

        false_positives = false_positives.rename({0: "0-11", 1: "1-12", 2: "2-13"}).T
        false_positives_cr = false_positives_cr.rename(
            {0: "0-11", 1: "1-12", 2: "2-13"}
        ).T

        false_positives["Mode"] = "Uncorrected"
        false_positives_cr["Mode"] = "Corrected"

        _label = f"{100 * (1 - ALPHA):.0f}"
        _label = r"\textbf{" + _label + r"th \%tile}"

        all_fps.append(pd.concat((false_positives, false_positives_cr)))
        all_fps[-1]["Statistical Test"] = STEST_NAMES[stest]
        all_fps[-1]["Sim"] = all_fps[-1].index
        all_fps[-1] = all_fps[-1].melt(
            id_vars=["Sim", "Mode", "Statistical Test"],
            var_name="Months",
            value_name="False Positives",
        )

    all_fps = pd.concat(all_fps)
    ornl_colours = {
        "green": "#007833",
        "bgreen": "#84b641",
        "orange": "#DE762D",
        "teal": "#1A9D96",
        "red": "#88332E",
        "blue": "#5091CD",
        "gold": "#FECB00",
    }

    ptile = 100 * (1 - ALPHA)

    def _estr(x):
        return np.percentile(x, ptile, method="closest_observation")

    if make_plots:
        fig, axis = plt.subplots(1, 1, figsize=(12.5 / 2.54, 10 / 2.54), dpi=120)
        _plt = sns.barplot(
            x="Statistical Test",
            y="False Positives",
            hue="Mode",
            data=all_fps.query("Months == '2-13'"),
            estimator=_estr,
            errorbar=("ci", ptile),
            ax=axis,
            palette=[
                ornl_colours.get(_color) for _color in ["teal", "orange"]
            ],  # pyright: ignore[reportArgumentType]
            saturation=1.0,
        )
        axis.axhline(niter * ALPHA, ls="--", lw=2, color="k", zorder=0)
        sns.despine(ax=axis, offset=5, trim=True)
        _ = _plt.legend().set_title("")
        _ = _plt.set_ylabel(f"False Positives {ptile:.0f}th %tile")
        plt.tight_layout()
        fig.savefig(
            f"false_pos_figs/plt_false_pos_"
            f"{f'{ALPHA:.2f}'.replace('.', 'p')}_ts{test_size}.{ext}"
        )
    all_fps["False Positive Rate"] = all_fps["False Positives"] / niter
    fprs_for_table = (
        all_fps.query("Months == '2-13'")
        .reset_index()
        .drop("Months", axis=1)
        .pivot(
            index="Sim",
            columns=["Statistical Test", "Mode"],
            values="False Positive Rate",
        )
    )
    _mean = fprs_for_table.mean()
    _pctile = fprs_for_table.quantile(1 - ALPHA)
    fprs_for_table.loc[r"\textbf{Mean}"] = _mean
    fprs_for_table.loc[_label] = _pctile
    with open(
        f"false_pos_{stest}_ts{test_size}_{f'{ALPHA:.2f}'.replace('.', 'p')}.tex",
        "w",
        encoding="utf-8",
    ) as _fout:
        _fout.write(fprs_for_table.to_latex(float_format="{:.03f}".format))

    _fprs = (
        all_fps.groupby(["Statistical Test", "Mode"])["False Positives"]
        .apply(np.percentile, 95)
        .reset_index()
    )
    _fprs["Test Size"] = test_size
    return _fprs


if __name__ == "__main__":
    fprs = []
    for _ts in [15, 20, 25, 30, 35, 45, 50, 55, 60]:
        fprs.append(main(test_size=_ts, make_plots=False))
    out_file = f"fprs_{str(ALPHA).replace('.', 'p')}.csv"
    fprs = pd.concat(fprs)
    fprs.to_csv(out_file)

    # One more time for the plots
    main(test_size=30, make_plots=True, ext="pdf")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from statsmodels.stats import multitest as smm

import detclim


def correct_pvals(pvals, method: str = "fdr_bh", alpha: float = 0.05):
    _pval_cr = []
    for idx in range(pvals.shape[0]):
        for jdx in range(pvals.shape[1]):
            for kdx in range(pvals.shape[2]):
                for ldx in range(pvals.shape[-1]):
                    _pval_cr.append(
                        smm.multipletests(
                            pvals=pvals[idx, jdx, kdx, :, ldx],
                            alpha=alpha,
                            method=method,
                            is_sorted=False,
                        )[1]
                    )
    return np.array(_pval_cr).reshape(pvals.shape)


def load_data(param, pcts, esizes):
    bst_dir = Path("bootstrap_data")
    data_out = []
    for esize in esizes:
        bstp_files = []
        for pct in pcts:
            if pct == 0:
                _param = "ctl"
            else:
                _param = f"{param}-{pct}p0pct"

            _bst_file = Path(
                bst_dir, f"bootstrap_output.1year_12avg_ts{esize}.ctl_{_param}_n1000.nc"
            )
            bstp_files.append(_bst_file)
        data_out.append(
            xr.open_mfdataset(bstp_files, combine="nested", concat_dim="pct")
        )
        data_out[-1]["pct"] = pcts

    data_out = xr.concat(data_out, dim="esize")
    data_out["esize"] = esizes
    return data_out


def main(param: str = "clubb_c1", ext: str = "png"):
    pcts = {"clubb_c1": [0, 1, 3, 5, 10], "effgw_oro": [0, 1, 5, 10, 20, 30, 40, 50]}
    pct_single = {"clubb_c1": 5, "effgw_oro": 30}
    esizes = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    ALPHA = 0.05

    colors = {"ks": "C0", "mw": "C1", "cvm": "C2", "wsr": "C3"}
    stests = ["ks", "cvm", "mw"]  # , "wsr"]
    data_out = load_data(param, pcts[param], esizes)
    data_out_cr = data_out.copy()

    for stest in ["ks", "mw", "cvm"]:
        data_out_cr[stest] = (
            data_out[stest].dims,
            correct_pvals(data_out[stest].values),
        )

    ctl_thr = (data_out.sel(pct=0) < ALPHA).sum(dim="vars").quantile(q=0.95, dim="iter")

    failed_tests = (
        ((data_out.sel(pct=pcts[param][1:]) < ALPHA).sum(dim="vars") > ctl_thr)
        .sum(dim="iter")
        .isel(time=2)
    )

    failed_tests_cr = (
        ((data_out_cr.sel(pct=pcts[param][1:]) < ALPHA).sum(dim="vars") > 0)
        .sum(dim="iter")
        .isel(time=2)
    )

    failed_tests /= data_out.iter.shape[0]
    failed_tests_cr /= data_out.iter.shape[0]

    fig, axis = plt.subplots(1, 4, figsize=(16, 7))
    for idx, stest in enumerate(stests):
        failed_tests[stest].plot.line(x="pct", ax=axis[idx])
        failed_tests_cr[stest].plot.line(
            x="pct",
            ax=axis[idx],
            ls="--",
        )
        axis[idx].set_title(stest)
        axis[idx].set_yscale("log")
        axis[idx].grid(visible=True, ls="--", color="grey")

    fig.tight_layout()
    plt.savefig(f"plt_enssize_power_{param}.{ext}")

    if len(pcts[param][1:]) == 4:
        fig, axis = plt.subplots(2, 2, figsize=(10, 6))
    else:
        fig, axis = plt.subplots(2, 4, figsize=(10, 6))

    axis = axis.flatten()
    for idx, pct in enumerate(pcts[param][1:]):
        for stest in stests:
            failed_tests[stest].sel(pct=pct).plot(
                x="esize",
                ax=axis[idx],
                label=detclim.STESTS[stest],
                color=colors[stest],
            )
            failed_tests_cr[stest].sel(pct=pct).plot(
                x="esize",
                ax=axis[idx],
                label=f"{detclim.STESTS[stest]} BH-FDR",
                ls="--",
                color=colors[stest],
            )
        axis[idx].set_title(f"{pct}% change")
        axis[idx].grid(visible=True, ls="--", color="grey")

    fig.tight_layout()
    plt.legend()
    plt.savefig(f"plt_enssize_power_{param}_bypct.{ext}")

    fig, axis = plt.subplots(1, 1, figsize=(12.5 / 2.54, 10 / 2.54), dpi=120)

    pct = pct_single[param]
    for stest in stests:
        failed_tests[stest].sel(pct=pct).plot(
            x="esize", ax=axis, label=detclim.STESTS[stest], color=colors[stest]
        )
        failed_tests_cr[stest].sel(pct=pct).plot(
            x="esize", ax=axis, ls="--", color=colors[stest]
        )
    axis.set_xlabel("Sub-ensemble size")
    axis.set_ylabel("Fraction of rejected tests")
    axis.set_title(f"{pct}% change in {param}")
    axis.grid(visible=True, ls="--", color="grey")

    fig.tight_layout()
    plt.legend()
    plt.savefig(f"plt_enssize_power_{param}_single.{ext}")


if __name__ == "__main__":
    for _param in ["clubb_c1", "effgw_oro"]:
        main(param=_param, ext="pdf")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

import detclim.results_plot as rplt


def main(dvar):
    scratch_dir = Path("/lustre/storm/nwp501/proj-shared/mkelleher/e3sm_scratch")
    ctl = "ctl"
    test = "clubb_c1-5p0pct"

    cases = {
        ctl: "2025-09-16.F2010.ne30pg2_r05_oECv3_control",
        test: "2025-09-16.F2010.ne30pg2_r05_oECv3_clubb_c1_2p520000",
    }

    ens_dirs = {_ens: Path(scratch_dir, _name, "run") for _ens, _name in cases.items()}

    file_pattern = "{case}.eam.h0.aavg.nc"
    ens_file = {
        _ens: Path(ens_dirs[_ens], file_pattern.format(case=_name))
        for _ens, _name in cases.items()
    }
    data = {_ens: xr.open_dataset(_file) for _ens, _file in ens_file.items()}

    _figwidth = 10 / 2.54
    _figheight = _figwidth * 0.9

    data_ref = data[ctl][dvar].mean(dim="time")
    data_test = data[test][dvar].mean(dim="time")

    n_q = data_ref.shape[0] // 2

    all_ = np.concatenate((data_test.values, data_ref.values))
    min_all = np.min(all_)
    max_all = np.max(all_)

    norm_ref = (data_ref - min_all) / (max_all - min_all)
    norm_test = (data_test - min_all) / (max_all - min_all)

    bins = np.linspace(0, 1, n_q)

    freq_ref, _ = np.histogram(norm_ref, bins=bins)
    freq_test, _ = np.histogram(norm_test, bins=bins)

    cdf_ref = freq_ref.cumsum()
    cdf_test = freq_test.cumsum()

    _xlabel = "Normalized annual global means"
    _ylabel = "N Ensemble members"

    df_all = pd.concat(
        [
            pd.DataFrame(
                {
                    _xlabel: bins,
                    _ylabel: [0, *cdf_ref],
                    "Simulation": rplt.fmt_case(ctl),
                }
            ),
            pd.DataFrame(
                {
                    _xlabel: bins,
                    _ylabel: [0, *cdf_test],
                    "Simulation": rplt.fmt_case(test),
                }
            ),
        ]
    )

    fig, axes = plt.subplots(
        1,
        1,
        figsize=(_figwidth, _figheight),
        dpi=300,
    )
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="sans-serif",
        rc={"lines.linewidth": 1.5, "font.size": 8},
    )
    sns.lineplot(x=_xlabel, y=_ylabel, hue="Simulation", data=df_all, ax=axes)
    plt.tight_layout()
    fig.savefig(f"plt_cdf_{dvar}.pdf")
    fig.savefig(f"plt_cdf_{dvar}.png")


if __name__ == "__main__":
    main(dvar="CLDLIQ")

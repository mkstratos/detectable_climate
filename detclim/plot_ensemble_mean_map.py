from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy import crs as ccrs

import detclim.results_plot as rplt


def main():
    scratch_dir = Path("/lustre/storm/nwp501/proj-shared/mkelleher/e3sm_scratch")
    ctl = "ctl"
    test = "clubb_c1-5p0pct"
    params_hum = {
        "effgw_oro": "GW Orog",
        "clubb_c1": "CLUBB C1",
        "zmconv_c0_ocn": "ZM Conv C0-Ocean",
        "ctl": "Control",
    }

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

    files_2d = {
        _ens: Path(
            ens_file[_ens].parent,
            f"{cases[_ens]}.eam.h0.0001-03_0002-02_climo_ANN_r05_ensavg.nc",
        )
        for _ens in cases
    }

    data_2d = {_ens: xr.open_dataset(_file) for _ens, _file in files_2d.items()}

    proj = ccrs.Robinson()
    src_crs = ccrs.PlateCarree()

    dvar = "CLDLIQ"
    title_props = {"fontsize": 8}
    _figwidth = 12.5 / 2.54
    plot_3panel = False

    cbar_props = {"location": "right", "pad": 0.01, "shrink": 0.8}
    _figheight = _figwidth * 1.1
    fig = plt.figure(
        figsize=(_figwidth, _figheight),
        dpi=300,
    )
    axes = [
        fig.add_subplot(
            2,
            1,
            idx + 1,
            projection=proj,
        )
        for idx in range(2)
    ]

    diff_data = {}
    for _ens, _data in data_2d.items():
        _pltdata = _data.isel(time=0)
        if "lev" in _pltdata.coords:
            _pltdata = _pltdata.isel(lev=-1)
        else:
            _pltdata = _pltdata.squeeze()
        _pltdata = _pltdata.eval(dvar)
        if dvar in _data:
            _name = _data[dvar].long_name
            _units = _data[dvar].units
        diff_data[_ens] = _pltdata

    _cf = axes[0].pcolormesh(
        _data["lon"],
        _data["lat"],
        diff_data[ctl],
        transform=src_crs,
        cmap="plasma",
        rasterized=True,
    )
    axes[0].set_title(f"{rplt.fmt_case(ctl)} mean", fontsize=title_props["fontsize"])
    _cax = fig.colorbar(_cf, ax=axes[0], **cbar_props)

    _cax.set_label(_units, fontsize=title_props["fontsize"])

    diff_2d = diff_data[test] - diff_data[ctl]
    _absmax = np.max(np.abs(diff_2d.quantile([0.01, 0.99])))
    _cf = axes[1].pcolormesh(
        data_2d[test]["lon"],
        data_2d[test]["lat"],
        diff_2d,
        vmin=-_absmax,
        vmax=_absmax,
        cmap="BrBG",
        transform=src_crs,
        rasterized=True,
    )
    axes[1].set_title(
        f"Difference {rplt.fmt_case(test)} - {rplt.fmt_case(ctl)}",
        fontsize=title_props["fontsize"],
    )
    _cax = fig.colorbar(_cf, ax=axes[1], **cbar_props)
    _cax.set_label(_units, fontsize=title_props["fontsize"])
    for axis in axes:
        axis.coastlines(linewidth=0.5)
        axis.gridlines(linestyle="--", linewidth=0.5, color="k")

    fig.suptitle(f"{_name} [{_units}]", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"test_diff_{_name.replace(' ', '_').replace('(', '').replace(')', '')}_"
        f"{ctl}-{test}.png"
    )
    plt.savefig(
        f"test_diff_{_name.replace(' ', '_').replace('(', '').replace(')', '')}_"
        f"{ctl}-{test}.pdf"
    )


if __name__ == "__main__":
    main()
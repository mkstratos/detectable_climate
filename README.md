# Detectable Climate

Scripts and notebooks related to the manuscript titled:
"Improving Climate Reproducibility Testing with False Discovery Rate Correction".

Submitted to [Earth System Dynamics](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-2311/) special issue on "Theoretical and computational aspects of ensemble design, implementation, and interpretation in climate science"


## About
This investigation was set to improve the usability of non bit-for-bit testing in
the E3SM Earth system model primarily by reducing the number of false positive tests
(i.e. a statistical type I error where the test claims two ensembles have different
simulated climates when in fact they are not)


## Status
[![Pre-Commit](https://github.com/mkstratos/detectable_climate/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/mkstratos/detectable_climate/actions/workflows/pre-commit.yml)
[![Python Tests](https://github.com/mkstratos/detectable_climate/actions/workflows/python-tests.yml/badge.svg)](https://github.com/mkstratos/detectable_climate/actions/workflows/python-tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15343054.svg)](https://doi.org/10.5281/zenodo.15343054)


## Package directories

- The `run_scripts` directory contains scripts nessecary for setup and execution of
E3SM (specific to the LCRC cluster "chrysalis")

- The `detclim` directory contains analysis scripts

- The `detclim/notebooks` directory contains exploratory notebooks


## Requirements
As described in `pyproject.toml`:
- setuptools
- numpy
- scipy
- pandas>=2.0.1
- matplotlib
- statsmodels
- xarray
- cartopy
- jupyter
- cftime
- dask
- nco
- nc-time-axis
- dask-mpi
- seaborn
- hvplot
- bokeh
- holoviews

from pathlib import Path

__version_info__ = (0, 1, 0)
__version__ = ".".join(str(vi) for vi in __version_info__)
data_path = Path(Path(__file__).parent, "bootstrap_data").resolve()

REJ_THR = {
    0.05: {
        15: {"ks": 9, "cvm": 16, "mw": 16, "wsr": 16},
        20: {"ks": 11, "cvm": 16, "mw": 17, "wsr": 16},
        25: {"ks": 12, "cvm": 17, "mw": 17, "wsr": 16},
        30: {"ks": 11, "cvm": 16, "mw": 16, "wsr": 17},
        35: {"ks": 11, "cvm": 16, "mw": 17, "wsr": 17},
        40: {"ks": 10, "cvm": 16, "mw": 16, "wsr": 16},
        45: {"ks": 14, "cvm": 16, "mw": 16, "wsr": 16},
        50: {"ks": 12, "cvm": 16, "mw": 17, "wsr": 17},
        55: {"ks": 11, "cvm": 17, "mw": 17, "wsr": 17},
        60: {"ks": 14, "cvm": 16, "mw": 17, "wsr": 17},
    },
    0.01: {
        15: {"ks": 7, "cvm": 9, "mw": 8, "wsr": 8},
        20: {"ks": 5, "cvm": 9, "mw": 9, "wsr": 9},
        25: {"ks": 6, "cvm": 9, "mw": 9, "wsr": 9},
        30: {"ks": 6, "cvm": 9, "mw": 9, "wsr": 10},
        35: {"ks": 7, "cvm": 10, "mw": 10, "wsr": 10},
        40: {"ks": 6, "cvm": 10, "mw": 10, "wsr": 9},
        45: {"ks": 6, "cvm": 9, "mw": 10, "wsr": 10},
        50: {"ks": 6, "cvm": 10, "mw": 10, "wsr": 10},
        55: {"ks": 6, "cvm": 10, "mw": 10, "wsr": 9},
        60: {"ks": 7, "cvm": 9, "mw": 9, "wsr": 9},
    },
}

STESTS = {
    "ks": "Kolmogorov-Smirnov",
    "cvm": "Cramer von Mises",
    "mw": "Mann-Whitney",
    "es": "Epps Singleton",
    "wsr": "Wilcoxon signed rank",
}

STESTS_SHORT = {
    "ks": "K-S",
    "cvm": "C-VM",
    "mw": "M-W",
    "es": "E-S",
    "wsr": "WSR",
}

METHOD_SHORT = {
    "uncor": "Uncor.",
    "fdr_bh": "FDR-BH",
    "fdr_by": "FDR-BY",
}

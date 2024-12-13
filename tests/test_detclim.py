import detclim
from detclim import results_plot


def test_module():
    assert detclim.data_path.exists(), "DATA DIRECTORY DOES NOT EXIST"


def test_casefmt():
    case_short = "effgw_oro-50p0pct"
    case_result = "GW orog 50.0%"
    assert results_plot.fmt_case(case_short) == case_result
    try:
        bad_value = results_plot.fmt_case("TEST ME")
    except ValueError:
        bad_value = True
    assert bad_value, "SOMEHOW INVALID CASE GOT CONVERTED???"
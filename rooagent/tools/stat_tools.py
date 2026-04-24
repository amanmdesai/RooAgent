from langchain_core.tools import tool
from .utils import (
    _counting_window_inputs,
    _roostats_upper_limit,
    _stat_summary,
)


@tool
def histogram_significance_and_limits(
    file_path: str,
    data_name: str = "",
    bkg_name: str = "bkg",
    sig_name: str = "sig",
    center: float = 50.0,
    window: float = 4.0,
) -> str:
    """Compute RooStats counting statistics from TH1 yields.
    Reports expected S+B Asimov p0, one-sided Z from p0, and CLs metrics.
    If data is provided (or a data histogram exists), also reports observed values.
    """
    inputs, err = _counting_window_inputs(
        file_path=file_path,
        bkg_name=bkg_name,
        sig_name=sig_name,
        data_name=data_name,
        center=center,
        window=window,
    )
    if err:
        return err

    n_bkg = inputs["n_bkg"]
    n_sig = inputs["n_sig"]
    n_obs = inputs["n_obs"]

    header = f"Window: [{center - window:.4g}, {center + window:.4g}]  N_bkg={n_bkg:.4g}  N_sig={n_sig:.4g}"
    return f"{header} | {_stat_summary(n_bkg, n_sig, n_obs)}"


@tool
def histogram_upper_limit(
    file_path: str,
    bkg_name: str = "bkg",
    sig_name: str = "sig",
    data_name: str = "",
    center: float = 50.0,
    window: float = 4.0,
    cl: float = 0.95,
) -> str:
    """Compute RooStats CLs upper limits on signal strength mu from TH1 yields.
    Reports expected B-only Asimov mu_up from signal and background yields.
    If data is provided (or a data histogram exists), also reports observed mu_up.
    """
    inputs, err = _counting_window_inputs(
        file_path=file_path,
        bkg_name=bkg_name,
        sig_name=sig_name,
        data_name=data_name,
        center=center,
        window=window,
    )
    if err:
        return err

    n_bkg = inputs["n_bkg"]
    n_sig = inputs["n_sig"]
    n_obs = inputs["n_obs"]
    if n_sig <= 0.0:
        return "Error: signal yield in window is zero; cannot compute upper limit."

    n_exp = max(0, int(round(n_bkg)))  # B-only Asimov
    mu_exp = _roostats_upper_limit(n_exp, n_bkg, n_sig, cl=cl)

    lines = [
        f"CL={cl:.0%}  N_bkg={n_bkg:.4g}  N_sig={n_sig:.4g}",
        f"Expected(B Asimov): N={n_exp}  mu_up={mu_exp:.4g}  yield_up={mu_exp * n_sig:.4g}",
    ]
    if n_obs is not None:
        mu_obs = _roostats_upper_limit(n_obs, n_bkg, n_sig, cl=cl)
        lines.append(f"Observed: N={n_obs}  mu_up={mu_obs:.4g}  yield_up={mu_obs * n_sig:.4g}")
    return " | ".join(lines)
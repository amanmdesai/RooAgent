import ROOT
from langchain_core.tools import tool
from .utils import (
    _open_root_file,
    _get_hist,
    _poisson_tail_ge,
    _poisson_tail_le,
    _fractional_integral,
    _significance_from_pvalue,
    _upper_limit_bisect,
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
    """Compute counting p-values, a significance, and a CLs summary from TH1 histograms.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing histograms.
    data_name : str
        Name of the data histogram. If empty, expected sensitivity is computed
        assuming N = S + B.
    bkg_name, sig_name : str
        Names of the background and signal histograms in the file.
    center, window : float
        Center and half-width (same units as histogram x-axis) of the counting
        window.

    The histograms are integrated as stored; no rebinning is applied.

    Returns
    -------
    str
        One-line summary with discovery p-values, significance, and CLs.
    """
    if not file_path:
        return "Error: file_path is required."

    f = _open_root_file(file_path)
    if not f:
        return f"Error: cannot open ROOT file '{file_path}'"

    hbkg = _get_hist(f, bkg_name)
    hsig = _get_hist(f, sig_name)
    hdata = _get_hist(f, data_name) if data_name else None

    missing = []
    if hbkg is None:
        missing.append(bkg_name)
    if hsig is None:
        missing.append(sig_name)
    if missing:
        f.Close()
        return f"Error: missing histograms: {', '.join(missing)} in {file_path}"

    n_bkg_frac = _fractional_integral(hbkg, center, window)
    n_sig_frac = _fractional_integral(hsig, center, window)
    if hdata:
        n_data_frac = _fractional_integral(hdata, center, window)
        n_obs = max(0, int(round(n_data_frac)))
    else:
        # expected sensitivity: assume observed = S + B
        n_obs = max(0, int(round(n_bkg_frac + n_sig_frac)))

    discovery_p_b = _poisson_tail_ge(n_obs, n_bkg_frac)
    discovery_p_sb = _poisson_tail_ge(n_obs, n_bkg_frac + n_sig_frac)
    exclusion_clb = _poisson_tail_le(n_obs, n_bkg_frac)
    exclusion_clsplusb = _poisson_tail_le(n_obs, n_bkg_frac + n_sig_frac)
    cls = exclusion_clsplusb / exclusion_clb if exclusion_clb > 0.0 else float("inf")
    significance = _significance_from_pvalue(discovery_p_b, n_obs, n_bkg_frac)

    lines = [
        f"DiscoveryPValue: {discovery_p_b:.6g}",
        f"DiscoveryAltPValue: {discovery_p_sb:.6g}",
        f"DiscoverySignificance: {significance:.6g}",
        f"CLs: {cls:.6g}",
        f"CLs+b: {exclusion_clsplusb:.6g}",
        f"CLb: {exclusion_clb:.6g}",
    ]

    f.Close()
    summary = " | ".join(lines)
    return summary


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
    """Compute a CLs upper limit on signal strength from TH1 histograms.

    When no significant excess is observed this tool returns the upper limit
    on the signal strength parameter mu (observed) and the expected limit
    assuming the background-only hypothesis (mu_exp).

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing histograms.
    bkg_name, sig_name : str
        Names of the background and signal histograms.
    data_name : str
        Name of the data histogram. If empty the observed count is taken as the
        nearest integer to B (background-only Asimov).
    center, window : float
        Center and half-width of the counting window (same units as x-axis).
    cl : float
        Confidence level, e.g. 0.95 for 95% CL (default).

    Returns
    -------
    str
        Summary string with observed and expected upper limits.
    """
    if not file_path:
        return "Error: file_path is required."

    f = _open_root_file(file_path)
    if not f:
        return f"Error: cannot open ROOT file '{file_path}'"

    hbkg = _get_hist(f, bkg_name)
    hsig = _get_hist(f, sig_name)
    hdata = _get_hist(f, data_name) if data_name else None

    missing = []
    if hbkg is None:
        missing.append(bkg_name)
    if hsig is None:
        missing.append(sig_name)
    if missing:
        f.Close()
        return f"Error: missing histograms: {', '.join(missing)} in {file_path}"

    n_bkg = _fractional_integral(hbkg, center, window)
    n_sig = _fractional_integral(hsig, center, window)

    if n_sig <= 0.0:
        f.Close()
        return "Error: signal yield in window is zero; cannot compute upper limit."

    if hdata:
        n_data = _fractional_integral(hdata, center, window)
        n_obs = max(0, int(round(n_data)))
    else:
        n_obs = max(0, int(round(n_bkg)))  # background-only Asimov

    n_obs_exp = max(0, int(round(n_bkg)))  # expected (B-only Asimov)

    mu_obs = _upper_limit_bisect(n_obs, n_bkg, n_sig, cl=cl)
    mu_exp = _upper_limit_bisect(n_obs_exp, n_bkg, n_sig, cl=cl)

    f.Close()

    lines = [
        f"CL: {cl:.0%}",
        f"N_obs: {n_obs}",
        f"N_bkg: {n_bkg:.4g}",
        f"N_sig (nominal): {n_sig:.4g}",
        f"ObservedUpperLimit_mu: {mu_obs:.4g}",
        f"ExpectedUpperLimit_mu: {mu_exp:.4g}",
        f"ObservedUpperLimit_yield: {mu_obs * n_sig:.4g}",
        f"ExpectedUpperLimit_yield: {mu_exp * n_sig:.4g}",
    ]
    return " | ".join(lines)
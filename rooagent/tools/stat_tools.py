from langchain_core.tools import tool
from .utils import (
    _open_root_file,
    _get_hist,
    _fractional_integral,
    _upper_limit_bisect,
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
    """Compute exact-Poisson p-values, discovery Z, and CLs from TH1 histograms.

    Always reports expected sensitivity (S+B Asimov, n_obs = round(S+B)).
    If data_name is given, also reports observed quantities.
    Discovery Z is one-sided (Z >= 0), valid for any B (exact Poisson).
    CLs = CLs+b / CLb is reported for exclusion cross-check.
    center, window: counting window [center-window, center+window] in x-axis units.
    """
    if not file_path:
        return "Error: file_path is required."
    f = _open_root_file(file_path)
    if not f:
        return f"Error: cannot open ROOT file '{file_path}'"

    hbkg, hsig = _get_hist(f, bkg_name), _get_hist(f, sig_name)
    hdata = _get_hist(f, data_name) if data_name else None
    f.Close()

    missing = [n for n, h in [(bkg_name, hbkg), (sig_name, hsig)] if h is None]
    if missing:
        return f"Error: missing histograms: {', '.join(missing)} in {file_path}"

    n_bkg = _fractional_integral(hbkg, center, window)
    n_sig = _fractional_integral(hsig, center, window)
    n_obs = max(0, int(round(_fractional_integral(hdata, center, window)))) if hdata is not None else None

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
    """Compute CLs upper limit on signal strength mu from TH1 histograms.

    Always reports expected limit (B-only Asimov, n_obs = round(B)).
    If data_name is given, also reports observed limit.
    center, window: counting window half-width in x-axis units. cl: confidence level.
    """
    if not file_path:
        return "Error: file_path is required."
    f = _open_root_file(file_path)
    if not f:
        return f"Error: cannot open ROOT file '{file_path}'"

    hbkg, hsig = _get_hist(f, bkg_name), _get_hist(f, sig_name)
    hdata = _get_hist(f, data_name) if data_name else None
    f.Close()

    missing = [n for n, h in [(bkg_name, hbkg), (sig_name, hsig)] if h is None]
    if missing:
        return f"Error: missing histograms: {', '.join(missing)} in {file_path}"

    n_bkg = _fractional_integral(hbkg, center, window)
    n_sig = _fractional_integral(hsig, center, window)
    if n_sig <= 0.0:
        return "Error: signal yield in window is zero; cannot compute upper limit."

    n_exp = max(0, int(round(n_bkg)))  # B-only Asimov
    mu_exp = _upper_limit_bisect(n_exp, n_bkg, n_sig, cl=cl)

    lines = [
        f"CL={cl:.0%}  N_bkg={n_bkg:.4g}  N_sig={n_sig:.4g}",
        f"Expected(B Asimov): N={n_exp}  mu_up={mu_exp:.4g}  yield_up={mu_exp * n_sig:.4g}",
    ]
    if hdata is not None:
        n_obs = max(0, int(round(_fractional_integral(hdata, center, window))))
        mu_obs = _upper_limit_bisect(n_obs, n_bkg, n_sig, cl=cl)
        lines.append(f"Observed: N={n_obs}  mu_up={mu_obs:.4g}  yield_up={mu_obs * n_sig:.4g}")
    return " | ".join(lines)
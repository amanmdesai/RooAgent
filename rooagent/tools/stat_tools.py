from typing import Dict, List, Optional

from langchain_core.tools import tool
from .utils import (
    _counting_window_inputs,
    _format_scan_summary,
    _normalize_parallel_arrays,
    _upper_limit,
    _stat_summary,
)


@tool
def histogram_significance_and_limits(
    file_path: str,
    data_name: str = "",
    bkg_name: str = "bkg",
    sig_name: str = "sig",
    center: float = None,
    window: float = None,
) -> str:
    """Compute discovery significance (Z) and CLs exclusion metrics from histograms.

    Args:
        file_path (str): ROOT file containing the histograms used for counting.
        data_name (str, optional): Name of the observed-data histogram (if available).
        bkg_name (str, optional): Name of the background histogram (default 'bkg').
        sig_name (str, optional): Name of the signal histogram (default 'sig').
        center (float, optional): Center of the counting window. If None, uses histogram center.
        window (float, optional): Half-width of the counting window. If None, uses full histogram range.

    Returns:
        str: Textual summary containing expected and observed Z (if data), and CLs, CLs+b, CLb values.
    Notes:
        If center/window are not provided, automatically uses the full available range of the histogram.
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
    used_center = inputs["center"]
    used_window = inputs["window"]

    header = (
        f"Signal={sig_name}  Center={used_center:.4g}  Window=[{used_center - used_window:.4g}, {used_center + used_window:.4g}]"
        f"  N_bkg={n_bkg:.4g}  N_sig={n_sig:.4g}"
    )
    return f"{header} | {_stat_summary(n_bkg, n_sig, n_obs)}"


@tool
def histogram_upper_limit(
    file_path: str,
    bkg_name: str = "bkg",
    sig_name: str = "sig",
    data_name: str = "",
    center: float = None,
    window: float = None,
    cl: float = 0.95,
) -> str:
    """Compute a CLs upper limit on the signal strength (mu) from histogram counts.

    Args:
        file_path (str): ROOT file with signal, background, and optionally observed-data histograms.
        bkg_name (str, optional): Background histogram name (default 'bkg').
        sig_name (str, optional): Signal histogram name used to scale signal strength (default 'sig').
        data_name (str, optional): Observed-data histogram name.
        center (float, optional): Center of the counting window. If None, uses histogram center.
        window (float, optional): Half-width of the counting window. If None, uses full histogram range.
        cl (float, optional): Confidence level to use (default 0.95 for 95% CL).

    Returns:
        str: Expected and observed mu_up and corresponding yield_up; or an explanatory error if the signal yield is zero.
    Notes:
        If center/window are not provided, automatically uses the full available range of the histogram.
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
    used_center = inputs["center"]
    used_window = inputs["window"]
    
    if n_sig <= 0.0:
        return "Error: signal yield in window is zero; cannot compute upper limit."

    n_exp = max(0, int(round(n_bkg)))  # B-only Asimov
    mu_exp = _upper_limit(n_exp, n_bkg, n_sig, cl=cl)

    lines = [
        (
            f"Signal={sig_name}  Center={used_center:.4g}  Window=[{used_center - used_window:.4g}, {used_center + used_window:.4g}]"
            f"  CL={cl:.0%}  N_bkg={n_bkg:.4g}  N_sig={n_sig:.4g}"
        ),
        f"Expected(B Asimov): N={n_exp}  mu_up={mu_exp:.4g}  yield_up={mu_exp * n_sig:.4g}",
    ]
    if n_obs is not None:
        mu_obs = _upper_limit(n_obs, n_bkg, n_sig, cl=cl)
        lines.append(f"Observed: N={n_obs}  mu_up={mu_obs:.4g}  yield_up={mu_obs * n_sig:.4g}")
    return " | ".join(lines)


@tool
def summarize_parameter_scan(
    parameter_values: List[float],
    series: Dict[str, List[float]] = {},
    significance: Optional[List[float]] = None,
    cls: Optional[List[float]] = None,
    upper_limits: Optional[List[float]] = None,
    pvalue: Optional[List[float]] = None,
    observed_significance: Optional[List[float]] = None,
    expected_significance: Optional[List[float]] = None,
    observed_pvalue: Optional[List[float]] = None,
    expected_pvalue: Optional[List[float]] = None,
    parameter_name: str = "parameter",
    parameter_unit: str = "",
    sort_by: str = "",
    top_n: int = 5,
    descending: Optional[bool] = None,
) -> str:
    """Summarize a generic parameter scan from aligned arrays.

    Provide one scan parameter array and a dictionary of same-length result
    arrays such as significance, cls, p-values, or limits. The tool validates
    alignment once and produces a compact ranking automatically.
    """
    resolved_series = dict(series or {})

    legacy_series = {
        "significance": significance,
        "cls": cls,
        "upper_limits": upper_limits,
        "pvalue": pvalue,
        "observed_significance": observed_significance,
        "expected_significance": expected_significance,
        "observed_pvalue": observed_pvalue,
        "expected_pvalue": expected_pvalue,
    }
    for name, values in legacy_series.items():
        if values is not None and name not in resolved_series:
            resolved_series[name] = values

    if not resolved_series:
        return "Error: series must contain at least one named array."

    try:
        parameter_data, series_data = _normalize_parallel_arrays("parameter_values", parameter_values, resolved_series)
        return _format_scan_summary(
            parameter_values=parameter_data,
            series=series_data,
            parameter_name=parameter_name,
            parameter_unit=parameter_unit,
            sort_by=sort_by,
            top_n=top_n,
            descending=descending,
        )
    except ValueError as exc:
        return f"Error: {exc}"
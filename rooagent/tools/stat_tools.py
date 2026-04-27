from typing import Dict, List, Optional

from langchain_core.tools import tool
from .utils import (
    _counting_window_inputs,
    _format_scan_summary,
    _normalize_parallel_arrays,
    _stat_summary,
)


@tool
def histogram_significance_and_cls(
    file_path: str,
    data_name: str = "",
    bkg_name: str = "bkg",
    sig_name: str = "sig",
    center: float = None,
    window: float = None,
    compute_cls: bool = False,
) -> str:
    """Compute discovery Z and/or CLs from signal/background/data histograms in a mass window.

    Use for histogram-based analyses after building histograms with root_tree_to_histogram.
    Name histograms 'sig' and 'bkg' by convention so this tool finds them automatically.

    compute_cls=False (default): discovery only (p0, Z). Use for all discovery analyses.
    compute_cls=True: also computes CLs, CLs+b, CLb. Use only for exclusion studies.
    Note: small CLs means the signal is excluded, NOT discovered.

    Args:
        file_path: ROOT file containing the signal and background (and optionally data) histograms.
        data_name: Histogram name for observed data; enables observed stats alongside Asimov.
        bkg_name: Background histogram name (default 'bkg').
        sig_name: Signal histogram name (default 'sig').
        center: Window centre; defaults to histogram midpoint.
        window: Half-width of counting window; defaults to full histogram half-range.
        compute_cls: Set True to also compute exclusion CLs.

    Returns: Header with yields and stat summary, or error string.
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
    return f"{header} | {_stat_summary(n_bkg, n_sig, n_obs, compute_cls=compute_cls)}"


@tool
def summarize_parameter_scan(
    parameter_values: List[float],
    series: Dict[str, List[float]] = {},
    significance: Optional[List[float]] = None,
    cls: Optional[List[float]] = None,
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
    """Rank and summarize results from a parameter scan. Always call after running a multi-point scan.

    Pass arrays via the series dict (e.g. series={"significance": [...]}) or use the
    legacy convenience kwargs (significance=..., cls=..., pvalue=...). Arrays must be
    index-aligned with parameter_values.

    Args:
        parameter_values: Scan parameter x-axis values.
        series: Dict of named result arrays, e.g. {"significance": [...], "cls": [...]}.
        significance / cls / pvalue / observed_* / expected_*: Aliases for series entries.
        parameter_name: Label for the scan variable.
        parameter_unit: Units string appended to the parameter label.
        sort_by: Series key to rank by; auto-detected from key type if omitted.
        top_n: Number of top candidates to display (default 5).
        descending: Override sort direction (auto-detected from key name if omitted).

    Returns: Ranked table of top scan points, or error string.
    """
    resolved_series = dict(series or {})

    legacy_series = {
        "significance": significance,
        "cls": cls,
        
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
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
    """Compute discovery significance (Z) and/or CLs exclusion metrics from counting histograms.

    Integrates signal, background and (optionally) observed-data histograms over a mass window
    [center − window, center + window] using fractional bin interpolation, then computes
    counting-statistics metrics.

    HEP STATISTICAL CONVENTION — two distinct analyses:

    **Discovery** (default compute_cls=False when only searching for an excess):
      p0 = P(N ≥ n_obs | background-only hypothesis)  — upper-tail Poisson SF.
      Z  = Φ⁻¹(1 − p0)  — Gaussian equivalent significance (one-sided).
      Claim discovery at Z ≥ 5 (p0 ≈ 3×10⁻⁷); evidence at Z ≥ 3.

    **Exclusion** (set compute_cls=True, the default):
      CLs+b = P(N ≤ n_obs | signal+background hypothesis)  — lower-tail Poisson CDF.
      CLb   = P(N ≤ n_obs | background-only hypothesis)    — lower-tail Poisson CDF.
      CLs   = CLs+b / CLb  ∈ [0, 1].
      Signal hypothesis excluded at 95 % CL when CLs < 0.05.
      CLs is NOT a discovery metric; small CLs means the signal is excluded, not discovered.

    When both discovery and exclusion are relevant (e.g. combined scan output), both sets of
    metrics are reported. Set compute_cls=False to suppress CLs for pure discovery scans and
    reduce output noise.

    If `data_name` is provided the tool reports both an **Expected (Asimov)** line (using
    n_exp = round(B+S)) and an **Observed** line (using the actual integrated data counts).
    Otherwise only the expected line is reported.

    Args:
        file_path (str): Path to the ROOT file containing the histograms.
        data_name (str, optional): Name of the observed-data histogram. Omit for expected-only output.
        bkg_name (str, optional): Name of the background histogram (default 'bkg').
        sig_name (str, optional): Name of the signal histogram (default 'sig').
        center (float, optional): Centre of the counting window on the x-axis (e.g. the signal
            mass hypothesis). Defaults to the histogram mid-point when not provided.
        window (float, optional): Half-width of the counting window. Defaults to the full
            histogram half-range when not provided.
        compute_cls (bool, optional): Whether to compute and report CLs exclusion metrics
            (CLs, CLs+b, CLb). Default False. Set to True only for exclusion analyses.
            CLs must not be computed or reported in discovery contexts.

    Returns:
        str: Formatted summary:
            "Signal=<name>  Center=<c>  Window=[lo, hi]  N_bkg=<B>  N_sig=<S> |
             Expected(S+B Asimov): N=<n>  p0=<p>  Z=<Z>sigma [CLs=... CLs+b=... CLb=...]
             | Observed: N=<n>  p0=<p>  Z=<Z>sigma [CLs=... CLs+b=... CLb=...]"
        CLs metrics are only present when compute_cls=True.
        Returns a descriptive error string when the file or histogram cannot be opened.

    Notes:
        - For a mass-scan exclusion analysis: set compute_cls=True (default), collect CLs per
          point, then call summarize_parameter_scan and plot_significance_and_cls.
        - For a mass-scan discovery analysis: set compute_cls=False, collect Z and p0 per
          point, then call summarize_parameter_scan and plot_significance_and_cls.
        - Parse the "Expected" and "Observed" fields separately to build scan arrays.
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
    """Summarise a completed parameter scan from aligned numeric arrays and rank the results.

    Use this tool AFTER collecting all per-point results from a scan (e.g. a mass scan where
    each point was evaluated with histogram_significance_and_cls). Pass the scan-parameter
    array and a dictionary of named result arrays; the tool validates that all arrays have the
    same length, then produces a ranked summary sorted by the most informative series.

    Typical usage for a mass scan::

        summarize_parameter_scan(
            parameter_values=[100, 150, 200, 250],
            series={"z_obs": [...], "z_exp": [...], "cls": [...]},
            parameter_name="mass",
            parameter_unit="GeV",
            sort_by="z_obs",
            top_n=5,
        )

    Args:
        parameter_values (List[float]): Ordered array of scan-parameter values (e.g. mass points).
        series (Dict[str, List[float]], optional): Named arrays of result values, all the same
            length as `parameter_values`. Keys become column labels in the output table.
        significance (List[float], optional): Legacy shorthand — equivalent to series["significance"].
        cls (List[float], optional): Legacy shorthand — equivalent to series["cls"].
        pvalue (List[float], optional): Legacy shorthand — equivalent to series["pvalue"].
        observed_significance (List[float], optional): Legacy shorthand.
        expected_significance (List[float], optional): Legacy shorthand.
        observed_pvalue (List[float], optional): Legacy shorthand.
        expected_pvalue (List[float], optional): Legacy shorthand.
        parameter_name (str, optional): Human-readable name for the scan parameter (default 'parameter').
        parameter_unit (str, optional): Physical unit appended to the parameter in the output table.
        sort_by (str, optional): Series key to sort by. Defaults to the first significance-like key
            found (e.g. 'z_obs'). Significance-like keys are sorted descending; CLs/p-value keys
            are sorted ascending.
        top_n (int, optional): Number of top candidates to list in the summary (default 5).
        descending (bool, optional): Override the automatic sort direction.

    Returns:
        str: A human-readable ranking table showing the best candidate point and the top-N entries.
        Returns an error string if arrays are misaligned or no series is provided.

    Notes:
        - Always call this tool before plotting to get a best-point summary in the response.
        - Do not manually re-rank results from free-text tool outputs; use this tool instead.
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
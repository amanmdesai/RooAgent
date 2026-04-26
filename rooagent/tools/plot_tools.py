from typing import List, Optional

import ROOT
from langchain_core.tools import tool

from .utils import (
    _normalize_parallel_arrays,
    _plot_2d_hist,
    _plot_2d_tree,
    _plot_hist,
    _plot_hist_compare,
    _plot_signal_vs_backgrounds,
    _plot_tree,
    _plot_tree_compare,
)


@tool
def plot(
    mode: str,
    output_pdf: str,
    file_path: str = "",
    hist_name: str = "",
    tree_name: str = "",
    variable: str = "",
    bins: int = 40,
    xmin: float = 0.0,
    xmax: float = 100.0,
    file_paths: Optional[List[str]] = None,
    hist_names: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    legends: Optional[List[str]] = None,
    xlabel: str = "",
    ylabel: str = "Events",
    logy: bool = False,
    normalize: bool = False,
    show_ratio: bool = False,
    rebin: int = 1,
    weight_branch: str = "",
    cuts: Optional[List[str]] = None,
    vector_mode: str = "any",
    apply_cuts_before_plot: bool = True,
    signal_file: str = "",
    signal_files: Optional[List[str]] = None,
    signal_label: str = "Signal",
    signal_labels: Optional[List[str]] = None,
    background_files: Optional[List[str]] = None,
    background_labels: Optional[List[str]] = None,
    data_file: str = "",
    data_label: str = "Data",
    plot_data: bool = False,
):
    """Plot 1D distributions from histograms or TTree branches.

    Mode → required params:
      'hist'             → file_path, hist_name
      'tree'             → file_path, tree_name, variable
      'tree_compare'     → file_paths, tree_name, variables, legends
      'hist_compare'     → file_paths, hist_names, legends
      'signal_background'→ signal_file (or signal_files), background_files, tree_name, variable

    Args:
        mode: Plotting mode (see above).
        output_pdf: Output PDF path.
        file_path: Single ROOT file (hist/tree modes).
        file_paths: Multiple ROOT files (compare/signal_background modes).
        hist_name / hist_names: Histogram name(s) for hist modes.
        tree_name: TTree name for tree modes.
        variable / variables: Branch(es) to plot.
        legends: Labels parallel to file_paths/hist_names (required for compare modes).
        xlabel / ylabel: Axis labels.
        bins / xmin / xmax: Binning for tree→hist conversion.
        logy: Logarithmic y-axis.
        normalize: Normalize before plotting.
        show_ratio: Show ratio subplot (compare modes).
        rebin: Rebin factor.
        weight_branch: Per-event weight branch or expression.
        cuts: C++ selection expressions.
        vector_mode: 'any'(default) or 'all' for vector-branch cuts.
        apply_cuts_before_plot: Set False to skip provided cuts.
        signal_file / signal_files / signal_label / signal_labels: Signal inputs (signal_background mode).
        background_files / background_labels: Background inputs (signal_background mode).
        data_file / data_label / plot_data: Optional data overlay (signal_background mode).

    Returns: Confirmation with saved path, or error string.
    """
    mode_key = (mode or "").strip().lower()

    if mode_key == "hist":
        if not file_path or not hist_name:
            return "Error: file_path and hist_name are required for mode='hist'."
        return _plot_hist(
            file_path=file_path,
            hist_name=hist_name,
            output_pdf=output_pdf,
            xlabel=xlabel,
            ylabel=ylabel,
            logy=logy,
            normalize=normalize,
            rebin=rebin,
        )

    if mode_key == "tree":
        if not file_path or not tree_name or not variable:
            return "Error: file_path, tree_name, and variable are required for mode='tree'."
        return _plot_tree(
            file_path=file_path,
            tree_name=tree_name,
            variable=variable,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            output_pdf=output_pdf,
            normalize=normalize,
            weight_branch=weight_branch,
            cuts=cuts,
            rebin=rebin,
            vector_mode=vector_mode,
            apply_cuts_before_plot=apply_cuts_before_plot,
        )

    if mode_key == "tree_compare":
        if not file_paths or not tree_name or not variables or not legends:
            return "Error: file_paths, tree_name, variables, and legends are required for mode='tree_compare'."
        return _plot_tree_compare(
            file_paths=file_paths,
            tree_name=tree_name,
            variables=variables,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            legends=legends,
            output_pdf=output_pdf,
            normalize=normalize,
            show_ratio=show_ratio,
            rebin=rebin,
            weight_branch=weight_branch,
            cuts=cuts,
            vector_mode=vector_mode,
            apply_cuts_before_plot=apply_cuts_before_plot,
        )

    if mode_key == "hist_compare":
        if not file_paths or not hist_names or not legends:
            return "Error: file_paths, hist_names, and legends are required for mode='hist_compare'."
        return _plot_hist_compare(
            file_paths=file_paths,
            hist_names=hist_names,
            legends=legends,
            output_pdf=output_pdf,
            xlabel=xlabel,
            ylabel=ylabel,
            logy=logy,
            normalize=normalize,
            show_ratio=show_ratio,
            rebin=rebin,
        )

    if mode_key in {"signal_background", "signal_vs_backgrounds"}:
        return _plot_signal_vs_backgrounds(
            signal_file=signal_file,
            signal_files=signal_files,
            signal_label=signal_label,
            signal_labels=signal_labels,
            background_files=background_files,
            background_labels=background_labels,
            data_file=data_file,
            data_label=data_label,
            plot_data=plot_data,
            tree_name=tree_name,
            variable=variable,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            output_pdf=output_pdf,
            normalize=normalize,
            show_ratio=show_ratio,
            rebin=rebin,
            weight_branch=weight_branch,
            cuts=cuts,
            vector_mode=vector_mode,
            apply_cuts_before_plot=apply_cuts_before_plot,
        )

    return "Error: unsupported mode. Use one of: hist, tree, tree_compare, hist_compare, signal_background."


@tool
def plot_2d(
    mode: str,
    output_pdf: str,
    file_path: str = "",
    tree_name: str = "",
    variable_x: str = "",
    variable_y: str = "",
    x_branch: str = "",
    y_branch: str = "",
    hist_name: str = "",
    bins_x: int = 40,
    xmin: float = 0.0,
    xmax: float = 100.0,
    bins_y: int = 40,
    ymin: float = 0.0,
    ymax: float = 100.0,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    logz: bool = False,
    normalize: bool = False,
    rebin_x: int = 1,
    rebin_y: int = 1,
    weight_branch: str = "",
    cuts: Optional[List[str]] = None,
    vector_mode: str = "any",
    apply_cuts_before_plot: bool = True,
):
    """Create a 2D plot from a TH2 histogram or two TTree branches.

    Mode → required params:
      'hist' → file_path, hist_name
      'tree' → file_path, tree_name, variable_x, variable_y

    Args:
        mode: 'hist' or 'tree'.
        output_pdf: Output PDF path.
        file_path: ROOT file path.
        hist_name: TH2 name (mode='hist').
        tree_name: TTree name (mode='tree').
        variable_x / variable_y: Branch names for x/y axes (mode='tree'; preferred over x_branch/y_branch).
        bins_x / xmin / xmax: X-axis binning.
        bins_y / ymin / ymax: Y-axis binning.
        xlabel / ylabel / zlabel: Axis labels.
        logz: Logarithmic color scale.
        normalize: Normalize before plotting.
        rebin_x / rebin_y: Rebin factors per axis.
        weight_branch: Per-event weight branch or expression.
        cuts: C++ selection expressions.
        vector_mode: 'any'(default) or 'all' for vector-branch cuts.
        apply_cuts_before_plot: Set False to skip provided cuts.

    Returns: Confirmation with saved path, or error string.
    """
    mode_key = (mode or "").strip().lower()

    if mode_key == "hist":
        if not file_path or not hist_name:
            return "Error: file_path and hist_name are required for mode='hist'."
        return _plot_2d_hist(
            file_path=file_path,
            hist_name=hist_name,
            output_pdf=output_pdf,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            logz=logz,
            normalize=normalize,
            rebin_x=rebin_x,
            rebin_y=rebin_y,
        )

    if mode_key == "tree":
        vx = variable_x or x_branch
        vy = variable_y or y_branch
        if not file_path or not tree_name or not vx or not vy:
            return "Error: file_path, tree_name, variable_x, and variable_y are required for mode='tree'."
        return _plot_2d_tree(
            file_path=file_path,
            tree_name=tree_name,
            x_branch=vx,
            y_branch=vy,
            output_pdf=output_pdf,
            bins_x=bins_x,
            xmin=xmin,
            xmax=xmax,
            bins_y=bins_y,
            ymin=ymin,
            ymax=ymax,
            xlabel=xlabel,
            ylabel=ylabel,
            color_palette=55,
            normalize=normalize,
            rebin_x=rebin_x,
            rebin_y=rebin_y,
            weight_branch=weight_branch,
            cuts=cuts,
            vector_mode=vector_mode,
            apply_cuts_before_plot=apply_cuts_before_plot,
        )

    return "Error: unsupported mode. Use one of: hist, tree."


@tool
def plot_significance_and_cls(
    parameter_values: Optional[List[float]] = None,
    significance: Optional[List[Optional[float]]] = None,
    cls: Optional[List[Optional[float]]] = None,
    y: Optional[List[Optional[float]]] = None,
    expected: Optional[List[Optional[float]]] = None,
    y_label: str = "Y",
    parameter_label: str = "Parameter",
    observed_label: str = "Observed",
    expected_label: str = "Expected",
    draw_cls_threshold: bool = False,
    cls_threshold: float = 0.05,
    logy: bool = False,
    output_png: str = "",
    output_pdf: str = "",
    n_sig: Optional[float] = None,
    n_bkg: Optional[float] = None,
    n_obs: Optional[int] = None,
):
    """Plot significance/CLs vs a scan parameter (array mode), or return a point stat summary (summary mode).

    Array mode: provide parameter_values + exactly one of significance/cls/y.
      Optional expected overlays a dashed second curve.
      When cls data is passed, the CLs=0.05 threshold line is drawn automatically.
      Saves to output_png/output_pdf (default: significance_cls.png).
    Summary mode: provide n_sig + n_bkg (+ optional n_obs) — returns text, no plot.

    Args:
        parameter_values: Scan x-axis values (array mode).
        significance: Z values per scan point (primary curve).
        cls: CLs values per scan point (primary curve; auto-draws threshold line).
        y: Generic y-values per scan point (primary curve).
        expected: Dashed overlay curve; must match parameter_values length.
        y_label: Y-axis label.
        parameter_label: X-axis label.
        observed_label / expected_label: Legend labels for each curve.
        draw_cls_threshold: Force-draw threshold line regardless of series type.
        cls_threshold: Y-value for threshold line (default 0.05).
        logy: Logarithmic y-axis.
        output_png / output_pdf: Output paths (default significance_cls.png if neither given).
        n_sig / n_bkg / n_obs: Yields for summary mode (no plot produced).

    Returns: Saved file paths (array mode) or stat text summary (summary mode), or error string.
    """
    if n_sig is not None and n_bkg is not None:
        from .utils import _stat_summary
        return _stat_summary(n_bkg=n_bkg, n_sig=n_sig, n_obs=n_obs)

    # Determine y-data from convenience kwargs.
    provided_series = [
        ("significance", significance),
        ("cls", cls),
        ("y", y),
    ]
    provided_series = [(name, values) for name, values in provided_series if values is not None]

    if len(provided_series) > 1:
        return "Error: provide exactly one of significance, cls, or y when plotting arrays."

    series_name = ""
    y_data = None
    if provided_series:
        series_name, y_data = provided_series[0]

    if y_data is None:
        return "Error: no y-data provided (significance/cls/y)."

    if not output_png and not output_pdf:
        output_png = "significance_cls.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return "Error: plotting backend not available"

    if parameter_values is None:
        return "Error: parameter_values must be provided when plotting arrays."

    try:
        parameter_data, series_data = _normalize_parallel_arrays("parameter_values", parameter_values, {"y": y_data})
    except ValueError as exc:
        message = str(exc)
        if "y has length" in message:
            return "Error: parameter_values and y-data must have the same length."
        if message == "values must be finite numeric values":
            return "Error: parameter_values and y-data must be finite numeric values."
        return f"Error: {message}"

    # Validate and parse the optional expected overlay.
    expected_data = None
    if expected is not None:
        try:
            _, exp_series = _normalize_parallel_arrays("parameter_values", parameter_values, {"expected": expected})
            expected_data = exp_series["expected"]
        except ValueError as exc:
            msg = str(exc)
            if "expected has length" in msg:
                return "Error: expected and parameter_values must have the same length."
            if msg == "values must be finite numeric values":
                return "Error: expected values must be finite numeric values."
            return f"Error: {msg}"

    x_arr = np.array(parameter_data, dtype=float)
    y_arr = np.array(series_data["y"], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_arr, y_arr, marker="o", linestyle="-", color="C0", label=observed_label)

    if expected_data is not None:
        exp_arr = np.array(expected_data, dtype=float)
        ax.plot(x_arr, exp_arr, marker="s", linestyle="--", color="C1", label=expected_label)

    if draw_cls_threshold or series_name == "cls":
        ax.axhline(float(cls_threshold), color="black", linestyle="--", linewidth=1.0, label=f"CLs={float(cls_threshold):.3g}")

    ax.set_xlabel(parameter_label)
    ax.set_ylabel(y_label)
    if logy:
        ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    if output_png:
        fig.savefig(output_png, bbox_inches="tight")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    saved = [p for p in [output_png, output_pdf] if p]
    if saved:
        return "Saved " + ", ".join(saved)
    return ""

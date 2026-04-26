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
    """Plot 1D distributions from histograms or TTree branches and produce comparison or signal/background plots.

    Args:
        mode (str): One of 'hist', 'tree', 'tree_compare', 'hist_compare', or 'signal_background'.
        output_pdf (str): Path to save the resulting plot (PDF).
        file_path (str): Path to a single ROOT file (required for single-file modes).
        file_paths (List[str], optional): Multiple ROOT files for compare modes.
        hist_name (str, optional): Histogram name when mode='hist'.
        hist_names (List[str], optional): Histogram names when mode='hist_compare'.
        tree_name (str, optional): TTree name when mode involves trees.
        variable (str, optional): Branch name to plot for single-tree modes.
        variables (List[str], optional): Branch names for compare/tree_compare modes.
        legends (List[str], optional): Legend labels parallel to file_paths/hist_names.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        bins (int, optional): Binning for tree->hist conversion.
        xmin (float, optional): Lower bound for binning.
        xmax (float, optional): Upper bound for binning.
        logy (bool, optional): Use a logarithmic y-axis.
        normalize (bool, optional): Normalize histograms before plotting.
        show_ratio (bool, optional): Show ratio subplot for compares.
        rebin (int, optional): Rebin factor to apply before plotting.
        weight_branch (str, optional): Weight branch or expression for event weights.
        cuts (List[str], optional): Selection expressions (C++ syntax).
        vector_mode (str, optional): How to treat vector branches ('any' or 'all').
        apply_cuts_before_plot (bool, optional): If False, any provided `cuts` will be ignored and not applied before plotting.
        signal_file (str), signal_files (List[str], optional): Signal inputs for signal/background mode.
        background_files (List[str], optional): Background inputs for signal/background mode.
        data_file (str, optional): Observed data file for overlay.
        plot_data (bool, optional): Whether to include a data overlay.

    Returns:
        str: Confirmation message with saved file path on success, or a descriptive error message.
    Notes:
        Request only parameters relevant to the selected `mode`. Provide an explicit example when prompting for missing inputs.
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
    """Create 2D plots from a 2D histogram or two tree branches.

    Args:
        mode (str): 'hist' or 'tree'. 'hist' expects `file_path` + `hist_name`. 'tree' expects `file_path` + `tree_name` + `variable_x` + `variable_y` (or x_branch/y_branch).
        output_pdf (str): Path to save the produced plot (PDF).
        file_path (str): ROOT file path.
        hist_name (str, optional): Histogram name when mode='hist'.
        tree_name (str, optional): TTree name when mode='tree'.
        variable_x (str, optional): X-axis branch name when mode='tree'.
        variable_y (str, optional): Y-axis branch name when mode='tree'.
        bins_x (int, optional), xmin (float, optional), xmax (float, optional): X-axis binning and limits.
        bins_y (int, optional), ymin (float, optional), ymax (float, optional): Y-axis binning and limits.
        xlabel (str, optional), ylabel (str, optional), zlabel (str, optional): Axis labels.
        logz (bool, optional): Use logarithmic color scale.
        normalize (bool, optional): Normalize 2D histogram before plotting.
        rebin_x (int, optional), rebin_y (int, optional): Rebin factors for each axis.
        weight_branch (str, optional): Weight branch or expression for event weights.
        cuts (List[str], optional): Selection expressions (C++ syntax) applied before plotting.
        vector_mode (str, optional): How to treat vector branches ('any' or 'all').
        apply_cuts_before_plot (bool, optional): If False, any provided `cuts` will be ignored and not applied before plotting.

    Returns:
        str: Confirmation message with saved file path or an error message.
    Notes:
        Request only the parameters required for the selected `mode` and provide an explicit example when prompting the user.
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
    """Plot or summarize discovery significance and CLs vs a scan parameter.

    This tool has two operating modes:

    **Array plotting mode** — provide `parameter_values` and one of `significance`, `cls`, or
    a generic `y` array as the primary (observed) curve. An optional `expected` array of the
    same length may be passed to overlay a second dashed curve (typically the Asimov/expected
    result) on the same axes. Both curves share the x-axis defined by `parameter_values`.
    If neither `output_png` nor `output_pdf` is given the plot is saved to `significance_cls.png`.
    Output directories are created automatically if they do not yet exist.

    **Numeric summary mode** — provide `n_sig` (expected signal yield) and `n_bkg` (expected
    background yield). `n_obs` is optional and produces observed quantities alongside the
    expected ones when supplied. No plot is produced; a text summary is returned instead.

    Args:
        parameter_values (List[float], optional): Scan parameter points (x-axis), required in array mode.
        significance (List[float], optional): Significance (Z) values for each scan point (primary curve).
        cls (List[float], optional): CLs values for each scan point (primary curve).
        y (List[float], optional): Generic y-values for each scan point (primary curve).
        expected (List[float], optional): Second curve (e.g. Asimov/expected significance or CLs) drawn
            as a dashed line. Must have the same length as `parameter_values`. If lengths differ an
            error is returned.
        y_label (str, optional): Y-axis label for the plot.
        parameter_label (str, optional): X-axis label for the scan parameter.
        observed_label (str, optional): Legend label for the primary (solid) curve. Default 'Observed'.
        expected_label (str, optional): Legend label for the dashed expected curve. Default 'Expected'.
        draw_cls_threshold (bool, optional): Draw a horizontal dashed threshold line (e.g. CLs = 0.05).
        cls_threshold (float, optional): Y-value for the threshold guide line (default 0.05).
        logy (bool, optional): Use a logarithmic y-axis.
        output_png (str, optional): Path to save the PNG. Parent directory is created if needed.
        output_pdf (str, optional): Path to save the PDF. Parent directory is created if needed.
        n_sig (float, optional): Signal yield for single-point numeric summary mode.
        n_bkg (float, optional): Background yield for single-point numeric summary mode.
        n_obs (int, optional): Observed count for single-point numeric summary mode.

    Returns:
        str: In array mode, a message listing all saved file paths. In numeric mode, a textual
            summary with Z, p-value, and CLs metrics. Returns a descriptive error string on failure.

    Notes:
        - Provide exactly one of `significance`, `cls`, or `y` as the primary curve; supplying
          more than one is an error.
        - For a mass-scan workflow, call this tool once per plot type (significance, p-value, CLs)
          and pass the observed series as the primary curve and the Asimov series as `expected`.
        - To produce separate observed-only and expected-only plots, call twice with different
          `output_png`/`output_pdf` paths.
        - When `n_sig` and `n_bkg` are given, the tool returns text without creating any file.
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
        message = str(exc).replace("parameter_values", "parameter_values")
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

    import os as _os
    if output_png:
        fig.savefig(output_png, bbox_inches="tight")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    saved = [p for p in [output_png, output_pdf] if p]
    if saved:
        return "Saved " + ", ".join(saved)
    return ""

import math
from typing import List, Optional

import ROOT
from langchain_core.tools import tool

from .utils import (
    _plot_2d_hist,
    _plot_2d_tree,
    _plot_hist,
    _plot_hist_compare,
    _plot_signal_vs_backgrounds,
    _plot_tree,
    _build_signal_background_ratio,
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
    ylabel: str = "",
    logy: bool = False,
    normalize: bool = False,
    show_ratio: bool = False,
    rebin: int = 1,
    weight_branch: str = "",
    cuts: Optional[List[str]] = None,
    vector_mode: str = "any",
    signal_file: str = "",
    signal_files: Optional[List[str]] = None,
    signal_label: str = "Signal",
    signal_labels: Optional[List[str]] = None,
    background_files: Optional[List[str]] = None,
    background_labels: Optional[List[str]] = None,
    data_file: str = "",
    data_label: str = "Data",
    plot_data: bool = False,
    apply_cuts_before_plot: bool = False,
):
    """
    General plotting tool for histograms and trees. Supports several modes.
    mode: "hist" (file_path+hist_name) | "tree" (file_path+tree_name+variable+bins/xmin/xmax) |
          "tree_compare" (file_paths+variables+legends) |
          "hist_compare" (file_paths+hist_names+legends) |
          "signal_background" (signal_file/signal_files + background_files).
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
    # legacy aliases accepted by tests / callers
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
):
    """
    Plot 2D histograms or trees.
    mode: "hist" (file_path+hist_name) | "tree" (file_path+tree_name+variable_x+variable_y).
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
        # Accept either `variable_x/variable_y` or legacy `x_branch/y_branch` names.
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
        )

    return "Error: unsupported mode. Use one of: hist, tree."


@tool
def plot_significance_and_cls(
    masses: Optional[List[float]] = None,
    significance: Optional[List[Optional[float]]] = None,
    cls: Optional[List[Optional[float]]] = None,
    upper_limits: Optional[List[Optional[float]]] = None,
    y: Optional[List[Optional[float]]] = None,
    y_label: str = "Y",
    output_png: str = "",
    output_pdf: str = "",
    # Legacy number-counting signature (kept for backwards compatibility)
    n_sig: Optional[float] = None,
    n_bkg: Optional[float] = None,
    n_obs: Optional[int] = None,
    method: str = "roostats",
    null_is_sb: bool = False,
    conf_level: float = 0.95,
):
    """Flexible plotting or numeric summary for significances / CLs.

    - If `n_sig` and `n_bkg` are provided, returns the numeric `_stat_summary`.
    - Otherwise expects `masses` + one of (`significance` | `cls` | `upper_limits` | `y`) and
      will create a PNG/PDF plot saved to `output_png`/`output_pdf`.
    """
    # Backwards-compatible numeric mode
    if n_sig is not None and n_bkg is not None:
        from .utils import _stat_summary

        return _stat_summary(n_bkg=n_bkg, n_sig=n_sig, n_obs=n_obs)

    # Determine y-data from convenience kwargs
    y_data = None
    for cand in (significance, cls, upper_limits, y):
        if cand is not None:
            y_data = cand
            break

    if y_data is None:
        return "Error: no y-data provided (significance/cls/upper_limits/y)."

    if not output_png and not output_pdf:
        return "Error: must provide output_png or output_pdf to save plot."

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return "Error: plotting backend not available"

    if masses is None:
        return "Error: masses must be provided when plotting arrays."

    # Convert to numpy arrays and handle None -> nan
    masses_arr = np.array([float(x) for x in masses], dtype=float)
    y_list = [np.nan if v is None else float(v) for v in y_data]
    y_arr = np.array(y_list, dtype=float)

    # Truncate to shortest length with warning if lengths mismatch
    ln = min(len(masses_arr), len(y_arr))
    warn = ""
    if len(masses_arr) != len(y_arr):
        warn = "WARNING: masses/y length mismatch — truncating to shortest length."
        masses_arr = masses_arr[:ln]
        y_arr = y_arr[:ln]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(masses_arr, y_arr, marker="o", linestyle="-", color="C0")
    ax.set_xlabel("Mass")
    ax.set_ylabel(y_label)
    ax.grid(True)

    if output_png:
        fig.savefig(output_png, bbox_inches="tight")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    msg = ""
    if output_png:
        msg = f"Saved {output_png}"
    elif output_pdf:
        msg = f"Saved {output_pdf}"
    if warn:
        msg = f"{warn} {msg}"
    return msg

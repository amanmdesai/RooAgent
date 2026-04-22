from typing import List, Dict, Optional
import ROOT
import os
import math
from langchain_core.tools import tool
from .utils import (
    _parse_background_inputs,
    _unique_canvas_name,
    _get_vector_branches,
    _rewrite_vector_cut,
    _build_dataframe,
    _has_column,
    _filtered_yield,
    _total_yield,
    _asymptotic_significance,
)


@tool
def apply_cut_and_count(file_path: str, tree_name: str, cut: str,
                        vector_mode: str = "all",
                        weight: Optional[str] = None,
                        file_paths: Optional[List[str]] = None) -> str:
    """Apply a selection cut to one or more TTrees and return the passing yield.

    Parameters
    ----------
    file_path : str
        Primary ROOT file path (or the only file when analyzing one sample).
    tree_name : str
        Name of the TTree to read.
    cut : str
        Selection expressed in C++ syntax (&&, ||, true, false). This will be
        automatically rewritten for vector branches when appropriate.
    vector_mode : str
        Reducer mode for vector branch comparisons: 'any' or 'all'.
    weight : Optional[str]
        Name of the MC weight branch to sum; if missing or not present, raw
        event counts are returned.
    file_paths : Optional[List[str]]
        Additional ROOT files to include (merged) when counting across samples.

    Returns
    -------
    str
        Formatted yield string (e.g. "yield=... files=N") or an error message.
    """

    paths = _parse_background_inputs(file_path, file_paths)
    if not paths:
        return "Error: no file(s) provided."

    vector_vars = _get_vector_branches(paths[0], tree_name)
    cut = _rewrite_vector_cut(cut, vector_vars, vector_mode)

    df = _build_dataframe(tree_name, paths)

    value = _filtered_yield(df, cut, weight)
    w_tag = "(w)" if weight else ""
    return f"yield{w_tag}={value} cut='{cut}' files={len(paths)}"


@tool
def compute_significance(signal_file: str,
                         tree_name: str,
                         cut: str,
                         background_file: str = "",
                         vector_mode: str = "all",
                         weight: Optional[str] = None,
                         background_files: Optional[List[str]] = None) -> str:
    """Compute the Poisson discovery significance for a selection cut.

    Uses the asymptotic Cowan formula: Z = sqrt(2*[(S+B)*ln(1+S/B) - S]),
    which reduces to S/sqrt(B) for S << B. Falls back to S/sqrt(B) when S or B
    are zero or degenerate.

    This tool computes significance once with all backgrounds summed. Always pass
    all background files together via `background_files` so that B is the sum of
    all backgrounds (do not call this tool separately per background file).

    Parameters
    ----------
    signal_file : str
        Path to the signal ROOT file.
    background_file : str
        Comma-separated string of background files or a single background file.
    tree_name : str
        Name of the TTree to read.
    cut : str
        Selection in C++ syntax (&&, ||, true, false). Vector comparisons are
        rewritten automatically when needed.
    vector_mode : str
        Reducer for vector-branch cuts: 'any' or 'all'.
    weight : Optional[str]
        Name of the MC weight branch to apply to both signal and backgrounds.
    background_files : Optional[List[str]]
        Explicit list of background files; preferred over passing a CSV string.

    Returns
    -------
    str
        Formatted string containing S, B and the computed Z value or an error.
    """

    vector_vars = _get_vector_branches(signal_file, tree_name)
    cut = _rewrite_vector_cut(cut, vector_vars, vector_mode)

    bkg_paths = _parse_background_inputs(background_file, background_files)
    if not bkg_paths:
        return "Error: no background file(s) provided."

    sig_df = ROOT.RDataFrame(tree_name, signal_file)
    bkg_df = _build_dataframe(tree_name, bkg_paths)

    S = _filtered_yield(sig_df, cut, weight)
    B = _filtered_yield(bkg_df, cut, weight)

    if B <= 0 and S <= 0:
        return "No events after cuts; significance undefined."
    if B <= 0:
        return f"S={S} B={B} Z=inf (no background)"

    significance = _asymptotic_significance(S, B)
    return f"S={S} B={B} Z={significance:.3f}"

@tool
def define_variable(
    file_path: str,
    tree_name: str,
    new_var_name: str,
    expression: str,
    save_file: Optional[str] = None
) -> str:
    """Define a new variable in a TTree using RDataFrame and optionally save.

    Parameters
    ----------
    file_path : str
        Path to the input ROOT file.
    tree_name : str
        Name of the TTree to operate on.
    new_var_name : str
        Name of the new variable to define.
    expression : str
        RDataFrame expression that computes the new variable.
    save_file : Optional[str]
        If provided, the output ROOT file path where the updated tree is saved.

    Returns
    -------
    str
        Message indicating where the new variable was saved.
    """

    rdf = ROOT.RDataFrame(tree_name, file_path)

    rdf = rdf.Define(new_var_name, expression)

    if save_file:
        output_file = save_file
    else:
        output_file = file_path.replace(".root", "_updated.root")

    rdf.Snapshot(tree_name, output_file)

    return f"New variable '{new_var_name}' defined and saved to '{output_file}'"


@tool
def define_variable_and_plot(file_path: str, tree_name: str,
                             new_variables: Dict[str, str],
                             variable_to_plot: str,
                             bins: int, xmin: float, xmax: float,
                             cuts: List[str],
                             output_file: str,
                             vector_mode: str = "all",
                             weight: Optional[str] = None) -> str:
    """Define new variables in a TTree, apply sequential cuts, and save a 1D histogram.

    Parameters
    ----------
    file_path : str
        Input ROOT file path.
    tree_name : str
        Name of the TTree to process.
    new_variables : dict
        Mapping of new variable names to RDataFrame expressions to define.
    variable_to_plot : str
        Name of the variable (one of the newly-defined or existing branches) to plot.
    bins, xmin, xmax : int/float
        Binning for the histogram.
    cuts : list[str]
        Sequential selection cuts to apply; each must use C++ syntax.
    output_file : str
        Output PDF path for the plot.
    vector_mode : str
        Reducer mode for vector comparisons: 'any' or 'all'.
    weight : Optional[str]
        Name of MC weight branch to use for weighted histogramming.

    Returns
    -------
    str
        Message indicating the saved plot path or an error.
    """

    df = ROOT.RDataFrame(tree_name, file_path)
    vector_vars = _get_vector_branches(file_path, tree_name)

    for name, expr in new_variables.items():
        df = df.Define(name, expr)

    for cut in cuts:
        safe_cut = _rewrite_vector_cut(cut, vector_vars, vector_mode)
        df = df.Filter(safe_cut)

    if weight and _has_column(df, weight):
        wname = weight
    else:
        wname = "__rooagent_unit_weight"
        df = df.Define(wname, "1.0")

    hist_ptr = df.Histo1D(
        (variable_to_plot, variable_to_plot, bins, xmin, xmax),
        variable_to_plot,
        wname
    )

    h = hist_ptr.GetValue()
    ROOT.SetOwnership(h, False)

    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)

    h.GetXaxis().SetTitle(variable_to_plot)
    h.GetYaxis().SetTitle("Events")

    canvas = ROOT.TCanvas(_unique_canvas_name("c1"), "", 900, 700)
    h.Draw("HIST")

    canvas.Update()
    canvas.SaveAs(output_file)

    return f"Plot saved to {output_file}"


@tool
def find_optimal_cut(signal_file: str,
                     tree_name: str,
                     variable: str,
                     min_cut: float,
                     max_cut: float,
                     step: float,
                     background_file: str = "",
                     base_cut: str = "",
                     vector_mode: str = "all",
                     weight: Optional[str] = None,
                     background_files: Optional[List[str]] = None) -> str:
    """Scan a variable cut range and return the cut value that maximizes significance.

    Parameters
    ----------
    signal_file, background_file/background_files, tree_name : see compute_significance
    variable : str
        Variable to scan (cut applied as `variable > cut_val`).
    min_cut, max_cut, step : float
        Range and step size for the scan.
    base_cut : str
        Additional base selection combined with the scanned cut.
    vector_mode, weight : see compute_significance

    Returns
    -------
    str
        Human-readable summary containing the optimal cut and significance.
    """

    bkg_paths = _parse_background_inputs(background_file, background_files)
    if not bkg_paths:
        return "Error: no background file(s) provided."

    sig_df = ROOT.RDataFrame(tree_name, signal_file)
    bkg_df = _build_dataframe(tree_name, bkg_paths)

    vector_vars = _get_vector_branches(signal_file, tree_name)

    best_cut = None
    best_sig = -1
    best_S = 0
    best_B = 0

    n_steps = int(round((max_cut - min_cut) / step)) + 1
    for i in range(n_steps):
        cut_val = round(min_cut + i * step, 10)
        if cut_val > max_cut:
            break

        scan_cut = f"{variable} > {cut_val}"

        if base_cut:
            full_cut = f"({base_cut}) && ({scan_cut})"
        else:
            full_cut = scan_cut

        full_cut = _rewrite_vector_cut(full_cut, vector_vars, vector_mode)

        S = _filtered_yield(sig_df, full_cut, weight)
        B = _filtered_yield(bkg_df, full_cut, weight)

        significance = _asymptotic_significance(S, B) if B > 0 else (S / math.sqrt(S) if S > 0 else 0.0)

        if significance > best_sig:
            best_sig = significance
            best_cut = cut_val
            best_S = S
            best_B = B

    return (
        f"Optimal cut found:\n"
        f"{variable} > {best_cut}\n"
        f"Signal file: {signal_file}\n"
        f"Background files ({len(bkg_paths)}): {', '.join(bkg_paths)}\n"
        f"S = {best_S}, B = {best_B}\n"
        f"Significance = {best_sig:.3f}"
    )


@tool
def generate_cutflow(
    file_path: str,
    tree_name: str,
    cuts: List[str],
    vector_mode: str = "any",
    weight: Optional[str] = None,
    file_paths: Optional[List[str]] = None
) -> str:
    """Generate a sequential cutflow by applying provided cuts to a TTree.

    Parameters
    ----------
    file_path : str
        Primary ROOT file path (or first file when multiple are supplied).
    tree_name : str
        Name of the TTree to process.
    cuts : list[str]
        Ordered list of selection cuts (C++ syntax) applied sequentially.
    vector_mode : str
        Reducer mode for vector-branch comparisons: 'any' or 'all'.
    weight : Optional[str]
        Name of the MC weight branch to use for weighted yields; if missing,
        raw event counts are reported.
    file_paths : Optional[List[str]]
        Additional files to include when computing the cutflow across samples.

    Returns
    -------
    str
        A multi-line cutflow summary showing yields after each cut.
    """
    paths = _parse_background_inputs(file_path, file_paths)
    if not paths:
        return "Error: no file(s) provided."

    # Per-file cutflows (helpful to inspect each background individually)
    per_file_sections = []
    for p in paths:
        df_p = ROOT.RDataFrame(tree_name, p)
        vector_vars_p = _get_vector_branches(p, tree_name)

        lines = []
        initial_p = _total_yield(df_p, weight)
        lines.append(f"Initial events: {initial_p}")

        current_df_p = df_p
        for cut in cuts:
            safe_cut_p = _rewrite_vector_cut(cut, vector_vars_p, vector_mode)
            current_df_p = current_df_p.Filter(safe_cut_p)
            val_p = _total_yield(current_df_p, weight)
            lines.append(f"{cut}: {val_p}")

        fname = os.path.basename(p)
        per_file_sections.append(f"Cutflow for file: {fname} ({p}):\n" + "\n".join(["- " + l for l in lines]))

    # Combined (merged) cutflow across all provided files
    df_all = _build_dataframe(tree_name, paths)
    vector_vars_all = _get_vector_branches(paths[0], tree_name)

    combined_lines = []
    initial_all = _total_yield(df_all, weight)
    combined_lines.append(f"Initial events (files={len(paths)}): {initial_all}")

    current_df_all = df_all
    for cut in cuts:
        safe_cut_all = _rewrite_vector_cut(cut, vector_vars_all, vector_mode)
        current_df_all = current_df_all.Filter(safe_cut_all)
        val_all = _total_yield(current_df_all, weight)
        combined_lines.append(f"{cut}: {val_all}")

    combined_section = "Combined cutflow (merged):\n" + "\n".join(["- " + l for l in combined_lines])

    # Concatenate per-file sections then combined summary
    output_sections = ["Cutflow (per-file):"] + per_file_sections + ["", combined_section]
    return "\n\n".join(output_sections)
from typing import List, Dict, Optional
import ROOT
import os
from langchain_core.tools import tool
from .utils import (
    _parse_paths,
    _unique_canvas_name,
    _get_vector_branches,
    _rewrite_vector_cut,
    _build_dataframe,
    _build_tree_hist,
    _has_column,
    _filtered_yield,
    _total_yield,
    _compute_significance_from_yields,
    _optimal_cut_significance,
)


@tool
def root_tree_to_histogram(
    file_path: str,
    tree_name: str,
    variable: str,
    bins: int,
    xmin: float,
    xmax: float,
    cuts: Optional[List[str]] = None,
    vector_mode: str = "any",
    weight: Optional[str] = None,
    output_root: Optional[str] = None,
    hist_name: Optional[str] = None,
) -> str:
    """Build a 1D histogram from a TTree branch and save it to a ROOT file.

    Args:
        file_path (str): Input ROOT file containing the TTree.
        tree_name (str): Name of the TTree to read.
        variable (str): Branch name to histogram.
        bins (int): Number of bins for the histogram.
        xmin (float): Lower x-axis bound.
        xmax (float): Upper x-axis bound.
        cuts (List[str], optional): C++-style selection expressions to filter events.
        vector_mode (str, optional): How to handle vector branches ('any' or 'all').
        weight (str, optional): Branch/expression to use as event weight.
        output_root (str, optional): Output ROOT filename to save the histogram; default is derived from the input file.
        hist_name (str, optional): Name for the created histogram.

    Returns:
        str: Confirmation message with saved histogram name and output file path, or an error message.
    """
    # Determine default histogram name if not provided
    stem = os.path.splitext(os.path.basename(file_path))[0]
    default_hist_name = f"h_{stem}_{variable}"
    chosen_hist_name = hist_name if hist_name else default_hist_name

    h = _build_tree_hist(
        file_path=file_path,
        tree_name=tree_name,
        variable=variable,
        bins=bins,
        xmin=xmin,
        xmax=xmax,
        hist_name=chosen_hist_name,
        weight_branch=weight or "",
        cuts=cuts,
        vector_mode=vector_mode,
        rebin=1,
    )

    if output_root is None:
        output_root = str(file_path).replace(".root", f"_{variable}_hist.root")

    f = ROOT.TFile.Open(output_root, "RECREATE")
    hcopy = h.Clone(chosen_hist_name)
    hcopy.SetDirectory(f)
    f.Write()
    f.Close()

    return f"Saved histogram '{chosen_hist_name}' to {output_root}"


@tool
def apply_cut_and_count(file_path: str, tree_name: str, cut: str,
                        vector_mode: str = "any",
                        weight: Optional[str] = None,
                        file_paths: Optional[List[str]] = None) -> str:
    """Count events that pass a selection cut in one or more ROOT files.

    Args:
        file_path (str): Primary ROOT file path (or empty if using `file_paths`).
        tree_name (str): TTree name to evaluate the cut on.
        cut (str): C++ boolean expression used to select events.
        vector_mode (str, optional): How to rewrite vector-branch cuts ('any' or 'all').
        weight (str, optional): Branch/expression name for event weights; if omitted, counts are unweighted.
        file_paths (List[str], optional): List of ROOT files to process instead of a single file.

    Returns:
        str: A string in the form "yield=<value> cut='<expr>' files=<count>" or an error message.
    """

    paths = _parse_paths(file_path, file_paths)
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
                         vector_mode: str = "any",
                         weight: Optional[str] = None,
                         background_files: Optional[List[str]] = None) -> str:
    """Compute discovery significance Z from signal and background yields after selection.

    Args:
        signal_file (str): ROOT file containing the signal TTree.
        tree_name (str): TTree to evaluate selection on.
        cut (str): Selection expression (C++ syntax).
        background_file (str, optional): Single background ROOT file.
        background_files (List[str], optional): Multiple background ROOT files.
        vector_mode (str, optional): How to handle vector-branch conditions ('any' or 'all').
        weight (str, optional): Branch/expression to apply per-event weights.

    Returns:
        str: Textual result like "S=<N> B=<N> Z=<value>" or an error message.
    """

    vector_vars = _get_vector_branches(signal_file, tree_name)
    cut = _rewrite_vector_cut(cut, vector_vars, vector_mode)

    bkg_paths = _parse_paths(background_file, background_files)
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

    significance = _compute_significance_from_yields(S, B)
    return f"S={S} B={B} Z={significance:.3f}"

@tool
def define_variable(
    file_path: str,
    tree_name: str,
    new_var_name: str,
    expression: str,
    save_file: Optional[str] = None
) -> str:
    """Add a derived branch to a TTree using a C++ expression and optionally save the updated tree.

    Args:
        file_path (str): ROOT file containing the TTree.
        tree_name (str): Name of the TTree to modify.
        new_var_name (str): Name of the new derived branch to create.
        expression (str): C++ expression describing how to compute the new branch.
        save_file (str, optional): Path to save the updated TTree; if omitted a new file name is derived.

    Returns:
        str: Confirmation message including the new variable name and the output file path.
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
                             output_file: str,
                             cuts: Optional[List[str]] = None,
                             vector_mode: str = "any",
                             weight: Optional[str] = None) -> str:
    """Define derived variables, apply optional cuts, and plot a single variable from the updated dataframe.

    Args:
        file_path (str): ROOT file containing the TTree.
        tree_name (str): Name of the TTree to read and modify.
        new_variables (Dict[str, str]): Mapping of new variable names to C++ expressions.
        variable_to_plot (str): Which derived or existing variable to histogram and plot.
        bins (int): Number of bins for the histogram.
        xmin (float): Lower x bound for the histogram.
        xmax (float): Upper x bound for the histogram.
        output_file (str): Path to save the plot (e.g., PDF).
        cuts (List[str], optional): Selection expressions applied after definitions.
        vector_mode (str, optional): How to handle vector branch expressions.
        weight (str, optional): Branch/expression used for event weighting.

    Returns:
        str: Confirmation message with saved plot path or an error message.
    """

    df = ROOT.RDataFrame(tree_name, file_path)
    vector_vars = _get_vector_branches(file_path, tree_name)

    for name, expr in new_variables.items():
        df = df.Define(name, expr)

    for cut in (cuts or []):
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
                     vector_mode: str = "any",
                     weight: Optional[str] = None,
                     background_files: Optional[List[str]] = None) -> str:
    """Scan threshold values on a variable to find the cut that maximizes an optimization metric.

    Args:
        signal_file (str): ROOT file with signal events.
        tree_name (str): TTree name to evaluate selection on.
        variable (str): Variable to scan (for example, mass or a classification score).
        min_cut (float): Minimum threshold for the scan.
        max_cut (float): Maximum threshold for the scan.
        step (float): Step size for the scan (must be > 0).
        background_file (str, optional): Single background ROOT file.
        background_files (List[str], optional): Multiple background ROOT files.
        base_cut (str, optional): Additional selection expression applied to all scans.
        vector_mode (str, optional): How vector branches are interpreted ('any' or 'all').
        weight (str, optional): Event weight branch/expression.

    Returns:
        str: A multi-line report describing the optimal cut, signal/background yields, and the computed significance.
    Notes:
        Validate that `step` > 0 and `max_cut >= min_cut` before starting the scan.
    """

    bkg_paths = _parse_paths(background_file, background_files)
    if not bkg_paths:
        return "Error: no background file(s) provided."
    if step <= 0:
        return "Error: step must be > 0."
    if max_cut < min_cut:
        return "Error: max_cut must be >= min_cut."

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

        significance = _optimal_cut_significance(S, B)

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
    tree_name: str,
    cuts: List[str],
    vector_mode: str = "any",
    weight: Optional[str] = None,
    file_path: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
) -> str:
    """Produce a cutflow table reporting yields after each sequential cut, both per-file and combined.

    Args:
        tree_name (str): TTree to evaluate cuts on.
        cuts (List[str]): Ordered list of C++ selection expressions to apply.
        vector_mode (str, optional): How to rewrite vector-branch expressions ('any' or 'all').
        weight (str, optional): Branch/expression to use as event weight.
        file_path (str, optional): Primary ROOT file path (or omitted if using `file_paths`).
        file_paths (List[str], optional): List of ROOT files to process instead of a single file.

    Returns:
        str: A human-readable cutflow report with per-file sections and a merged summary.
    Notes:
        If a machine-readable output is desired (CSV or JSON), consider serializing the returned information to a file instead of the default human-readable format.
    """
    paths = _parse_paths(file_path, file_paths)
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
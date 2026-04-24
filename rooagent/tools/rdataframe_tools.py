from typing import List, Dict, Optional
import ROOT
import os
import math
from langchain_core.tools import tool
from .utils import (
    _parse_paths,
    _unique_canvas_name,
    _get_vector_branches,
    _rewrite_vector_cut,
    _build_dataframe,
    _tree_variable_to_histogram,
    _has_column,
    _filtered_yield,
    _total_yield,
    _compute_significance_from_yields,
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
    vector_mode: str = "all",
    weight: Optional[str] = None,
    output_root: Optional[str] = None,
    hist_name: Optional[str] = None,
) -> str:
    """Convert a numeric tree variable into a 1D TH1 and save to a ROOT file.
    Output defaults to `<input>_<variable>_hist.root` if not specified. Supports optional cuts and weights.
    """
    h = _tree_variable_to_histogram(file_path, tree_name, variable, bins, xmin, xmax, cuts, vector_mode, weight)

    chosen_name = hist_name if hist_name else h.GetName()
    if output_root is None:
        output_root = str(file_path).replace(".root", f"_{variable}_hist.root")

    f = ROOT.TFile.Open(output_root, "RECREATE")
    hcopy = h.Clone(chosen_name)
    hcopy.SetDirectory(f)
    f.Write()
    f.Close()

    return f"Saved histogram '{chosen_name}' to {output_root}"


@tool
def apply_cut_and_count(file_path: str, tree_name: str, cut: str,
                        vector_mode: str = "all",
                        weight: Optional[str] = None,
                        file_paths: Optional[List[str]] = None) -> str:
    """Apply a selection cut to one or more TTrees and return the passing yield.

    cut: C++ syntax; vector branches rewritten automatically.
    weight: MC weight branch to sum; absent -> raw event count.
    file_paths: extra files merged with file_path.
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
                         vector_mode: str = "all",
                         weight: Optional[str] = None,
                         background_files: Optional[List[str]] = None,
                         method: str = "simple") -> str:
    """Compute expected discovery significance Z for a tree variable after selection cut.
    Supports "simple" (RooStats number-counting) or "asymptotic" (Asimov) methods. Sum all backgrounds first.
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

    significance = _compute_significance_from_yields(S, B, method)
    return f"S={S} B={B} Z={significance:.3f} method={method}"

@tool
def define_variable(
    file_path: str,
    tree_name: str,
    new_var_name: str,
    expression: str,
    save_file: Optional[str] = None
) -> str:
    """Define a new branch in a TTree via an RDataFrame expression and save.

    save_file: output path; defaults to <input>_updated.root.
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
    """Define new variables in a TTree, apply cuts, and save a 1D histogram.

    new_variables: dict of name->RDataFrame expression.
    cuts: sequential C++ selections applied in order.
    weight: MC weight branch for weighted fill.
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
                     background_files: Optional[List[str]] = None,
                     method: str = "simple") -> str:
    """Scan variable > cut_val and return the cut value that maximises significance.

    method: "simple" (RooStats number-counting, default) or "asymptotic" (Cowan 2010, only when B>>10).
    base_cut: extra selection AND-ed with each scanned cut.
    """

    bkg_paths = _parse_paths(background_file, background_files)
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

        significance = _compute_significance_from_yields(S, B, method) if B > 0 else 0.0

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
    """Apply sequential cuts to a TTree and report yield after each step.

    cuts: ordered list of C++ selections.
    weight: MC weight branch; absent -> raw counts.
    file_paths: additional files merged with file_path.
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
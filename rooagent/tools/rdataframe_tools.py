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
    """Project a TTree branch into a 1D histogram and save it to a ROOT file.

    This is the standard way to convert raw event-level ROOT data into a histogram that can
    then be passed to histogram_significance_and_cls or plot tools. The histogram is saved
    with the chosen `hist_name` so it can be retrieved by name in subsequent tool calls.

    Args:
        file_path (str): Input ROOT file containing the TTree.
        tree_name (str): Name of the TTree to read.
        variable (str): Branch name to histogram. Must be a numeric (scalar or vector) branch.
        bins (int): Number of histogram bins.
        xmin (float): Lower x-axis bound (left edge of first bin).
        xmax (float): Upper x-axis bound (right edge of last bin).
        cuts (List[str], optional): Ordered list of C++ boolean expressions applied before
            filling the histogram. Events failing any cut are excluded.
        vector_mode (str, optional): How to evaluate cuts on vector branches.
            'any' (default): condition is satisfied if ANY element passes.
            'all': condition is satisfied only if ALL elements pass.
        weight (str, optional): Branch name or C++ expression used as per-event weight.
            If the branch does not exist, falls back to unweighted counting.
        output_root (str, optional): Path for the output ROOT file. Defaults to
            '<input_stem>_<variable>_hist.root' in the same directory as the input.
        hist_name (str, optional): Name to give the stored histogram (used to retrieve it later).
            Defaults to 'h_<file_stem>_<variable>'.

    Returns:
        str: Confirmation message "Saved histogram '<name>' to <path>" or an error string.

    Notes:
        - Use this tool to prepare signal and background histograms for a mass-window counting
          analysis before calling histogram_significance_and_cls.
        - Set hist_name='sig' / hist_name='bkg' so the stat tool can find them by convention.
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
    """Count events passing a selection cut in one or more ROOT files.

    Evaluates a C++ boolean expression on a TTree and returns the (optionally weighted) event
    yield. Vector-branch conditions are automatically rewritten using ROOT::VecOps::Any or
    ROOT::VecOps::All depending on `vector_mode`.

    Args:
        file_path (str): Primary ROOT file path. Pass an empty string if using `file_paths` only.
            May also be a comma-separated list of file paths.
        tree_name (str): Name of the TTree to evaluate the cut on.
        cut (str): C++ boolean expression, e.g. "pt > 25 && abs(eta) < 2.4".
        vector_mode (str, optional): How vector-branch conditions are handled.
            'any' (default): satisfied if any element passes. 'all': all elements must pass.
        weight (str, optional): Branch or expression used as per-event weight. If the column
            does not exist in the dataframe, the tool silently falls back to unweighted counting.
        file_paths (List[str], optional): Additional ROOT files merged with `file_path`.

    Returns:
        str: "yield=<N> cut='<expr>' files=<K>" where N is the (weighted) count and K is the
            number of files processed. Returns an error string on failure.
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
    """Compute discovery significance Z from signal and background yields after a cut selection.

    Counts signal (S) and background (B) events passing the `cut` expression, then computes
    the Poisson-based discovery significance Z using the Asimov approximation
    (n_obs = round(S + B), background-only p-value = P(N ≥ n_obs | B)).

    Args:
        signal_file (str): ROOT file containing the signal TTree.
        tree_name (str): TTree name shared by signal and background files.
        cut (str): C++ boolean selection expression applied to both signal and background.
        background_file (str, optional): Single background ROOT file path. May be a
            comma-separated list of paths.
        background_files (List[str], optional): Additional background ROOT files (merged with
            `background_file` counts).
        vector_mode (str, optional): How vector-branch conditions are evaluated ('any'/'all').
        weight (str, optional): Per-event weight branch or expression.

    Returns:
        str: "S=<N> B=<N> Z=<value>" on success, or an error string. Z is the Gaussian
            equivalent of the Poisson discovery p-value. Returns "Z=inf" when B = 0.

    Notes:
        - For histogram-based significance (after binning), use histogram_significance_and_cls
          which also provides CLs exclusion metrics.
        - This tool uses the Asimov expected count; pass observed data histograms to
          histogram_significance_and_cls for observed Z.
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
    """Add a derived branch to a TTree via a C++ expression and save the updated tree.

    Uses ROOT RDataFrame's Define() to compute a new column from an arbitrary C++ expression,
    then snapshots the resulting tree to a new ROOT file. Useful for computing physics variables
    (e.g. invariant mass, transverse momentum) from raw branches before analysis.

    Args:
        file_path (str): Input ROOT file containing the TTree.
        tree_name (str): Name of the TTree to extend.
        new_var_name (str): Name for the new derived branch (must be a valid C++ identifier).
        expression (str): C++ expression evaluated per-event, e.g. "sqrt(px*px + py*py)".
            All existing branches in the tree are in scope.
        save_file (str, optional): Output ROOT file path. If omitted, defaults to
            '<input_stem>_updated.root' in the same directory.

    Returns:
        str: "New variable '<name>' defined and saved to '<path>'" or an error string.

    Notes:
        - After saving, use inspect_root_data(mode='branches') on the output file to confirm
          the new branch is present before running downstream tools.
        - To define multiple variables or immediately plot the result, use
          define_variable_and_plot instead.
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
    """Define derived branches, apply cuts, and immediately plot one variable.

    Combines variable definition (RDataFrame.Define), event selection (RDataFrame.Filter)
    and histogram plotting into a single step. Use this when you want to visualise a newly
    computed physics quantity without saving an intermediate ROOT file.

    Args:
        file_path (str): Input ROOT file containing the TTree.
        tree_name (str): TTree name to read.
        new_variables (Dict[str, str]): Mapping from new branch name to C++ expression, e.g.
            {"m_inv": "sqrt((e1+e2)*(e1+e2) - (px1+px2)*(px1+px2))"}.
            Multiple variables may be defined; later definitions can reference earlier ones.
        variable_to_plot (str): Name of the branch (existing or newly defined) to histogram.
        bins (int): Number of histogram bins.
        xmin (float): Lower x-axis bound.
        xmax (float): Upper x-axis bound.
        output_file (str): Path to save the plot (PDF or PNG).
        cuts (List[str], optional): C++ selection expressions applied after all Define() calls.
        vector_mode (str, optional): Vector-branch cut evaluation mode ('any'/'all').
        weight (str, optional): Event weight branch or expression. Falls back to unweighted
            counting if the branch is not found.

    Returns:
        str: "Plot saved to <path>" on success, or an error string on failure.
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
    """Scan a variable threshold and find the cut that maximises discovery significance.

    Iterates over threshold values from `min_cut` to `max_cut` in steps of `step`, evaluating
    signal (S) and background (B) yields at each threshold using `variable > threshold`. The
    significance at each point is computed as S / sqrt(S + B) (Asimov approximation). Returns
    the threshold that maximises this metric.

    Args:
        signal_file (str): ROOT file containing signal events.
        tree_name (str): TTree name shared by signal and background files.
        variable (str): Branch name whose threshold is scanned (e.g. a BDT score or invariant mass).
        min_cut (float): Starting threshold value.
        max_cut (float): Ending threshold value (inclusive). Must be ≥ `min_cut`.
        step (float): Increment between threshold values. Must be > 0.
        background_file (str, optional): Single background ROOT file path.
        background_files (List[str], optional): Additional background ROOT files.
        base_cut (str, optional): Fixed C++ selection applied in addition to the scanned threshold.
        vector_mode (str, optional): How vector-branch conditions are evaluated ('any'/'all').
        weight (str, optional): Per-event weight branch or expression.

    Returns:
        str: Multi-line report with the optimal threshold, S, B and significance at that point,
            plus a summary of all scanned points. Returns an error string if step <= 0 or no
            background files are provided.

    Notes:
        - Use this for quick cut optimisation on a single variable. For a full parameter scan
          producing significance vs mass plots, use histogram_significance_and_cls in a loop
          followed by summarize_parameter_scan and plot_significance_and_cls.
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
    """Build a sequential cutflow table showing event yields after each ordered cut.

    Applies the `cuts` list sequentially (each cut is cumulative — events must pass ALL
    preceding cuts to reach the next) and reports the yield at each stage. The output
    includes a per-file breakdown (to compare individual samples) and a merged total
    across all provided files.

    Args:
        tree_name (str): TTree name present in all provided files.
        cuts (List[str]): Ordered list of C++ boolean expressions applied sequentially.
            Example: ["pt > 20", "abs(eta) < 2.5", "m > 120 && m < 130"].
        vector_mode (str, optional): How vector-branch conditions are evaluated ('any'/'all').
        weight (str, optional): Per-event weight branch or expression. Falls back to
            unweighted counting if the branch does not exist.
        file_path (str, optional): Primary ROOT file. May be a comma-separated list.
        file_paths (List[str], optional): Additional ROOT files merged with `file_path`.

    Returns:
        str: Human-readable report with sections:
            - "Cutflow (per-file):" — one block per file showing initial count + yield after each cut.
            - "Combined cutflow (merged):" — aggregate across all files.
        Returns an error string if no files are provided.

    Notes:
        - Use this as the first tool after file discovery to understand selection efficiency
          before choosing working points for apply_cut_and_count or compute_significance.
        - The initial counts should match apply_cut_and_count with cut="1==1" for each file.
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
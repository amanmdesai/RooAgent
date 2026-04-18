from typing import List, Dict, Optional
import ROOT
from langchain_core.tools import tool
from .utils import *


# Module-level counter ensures unique ROOT canvas names
_canvas_counter = [0]


def _unique_canvas_name(base: str) -> str:
    _canvas_counter[0] += 1
    return f"{base}_{_canvas_counter[0]}"


def _parse_background_inputs(
    background_file: Optional[str] = None,
    background_files: Optional[List[str]] = None
) -> List[str]:
    """
    Normalize background inputs into a clean list of ROOT file paths.

    Supports either:
    - background_file="bkg.root"
    - background_files=["bkg1.root", "bkg2.root"]
    - background_file="bkg1.root, bkg2.root" (comma-separated)
    """
    parsed: List[str] = []

    if background_file:
        parsed.extend([p.strip() for p in background_file.split(",") if p.strip()])

    if background_files:
        parsed.extend([p.strip() for p in background_files if p and p.strip()])

    # De-duplicate while preserving order.
    unique = list(dict.fromkeys(parsed))
    return unique


def _build_dataframe(tree_name: str, files: List[str]):
    """Build an RDataFrame from one or more ROOT files."""
    if len(files) == 1:
        return ROOT.RDataFrame(tree_name, files[0])
    return ROOT.RDataFrame(tree_name, files)


def _has_column(df, column_name: str) -> bool:
    """Return True if the RDataFrame contains a column with the given name."""
    if not column_name:
        return False

    cols = [str(c) for c in df.GetColumnNames()]
    return column_name in cols


def _filtered_yield(df, cut: str, weight: Optional[str] = None):
    """Return weighted or unweighted yield after applying a cut."""
    filtered = df.Filter(cut)
    if weight and _has_column(filtered, weight):
        return filtered.Sum(weight).GetValue()
    return filtered.Count().GetValue()


def _total_yield(df, weight: Optional[str] = None):
    """Return weighted total yield, or Count if weight is missing."""
    if weight and _has_column(df, weight):
        return df.Sum(weight).GetValue()
    return df.Count().GetValue()


@tool
def apply_cut_and_count(file_path: str, tree_name: str, cut: str,
                        vector_mode: str = "all",
                        weight: Optional[str] = None,
                        file_paths: Optional[List[str]] = None) -> str:
    """Apply a selection cut to a TTree and count passing events. vector_mode 'any'=any object passes, 'all'=all objects pass."""

    paths = _parse_background_inputs(file_path, file_paths)
    if not paths:
        return "Error: no file(s) provided."

    vector_vars = get_vector_branches(paths[0], tree_name)
    cut = rewrite_vector_cut(cut, vector_vars, vector_mode)

    df = _build_dataframe(tree_name, paths)

    value = _filtered_yield(df, cut, weight)
    if weight:
        return f"Weighted events passing cut '{cut}': {value} (files={len(paths)})"
    return f"Events passing cut '{cut}': {value} (files={len(paths)})"


@tool
def compute_significance(signal_file: str,
                         background_file: str,
                         tree_name: str,
                         cut: str,
                         vector_mode: str = "all",
                         weight: Optional[str] = None,
                         background_files: Optional[List[str]] = None) -> str:
    """Compute S/sqrt(S+B) significance for a selection cut. vector_mode 'any'/'all' for vector branches."""

    vector_vars = get_vector_branches(signal_file, tree_name)
    cut = rewrite_vector_cut(cut, vector_vars, vector_mode)

    bkg_paths = _parse_background_inputs(background_file, background_files)
    if not bkg_paths:
        return "Error: no background file(s) provided."

    sig_df = ROOT.RDataFrame(tree_name, signal_file)
    bkg_df = _build_dataframe(tree_name, bkg_paths)

    S = _filtered_yield(sig_df, cut, weight)
    B = _filtered_yield(bkg_df, cut, weight)

    if (S + B) <= 0:
        return "No events after cuts; significance undefined."

    significance = S / ((S + B) ** 0.5)

    return (
        f"Signal file: {signal_file}\n"
        f"Background files ({len(bkg_paths)}): {', '.join(bkg_paths)}\n"
        f"S={S}, B={B}, Significance={significance:.3f}"
    )

@tool
def define_variable(
    file_path: str,
    tree_name: str,
    new_var_name: str,
    expression: str,
    save_file: Optional[str] = None
) -> str:
    """Define a new variable in a TTree via RDataFrame and save to a ROOT file."""

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
    """Define new TTree variables, apply sequential cuts, and save a 1D histogram plot."""

    df = ROOT.RDataFrame(tree_name, file_path)
    vector_vars = get_vector_branches(file_path, tree_name)

    for name, expr in new_variables.items():
        df = df.Define(name, expr)

    for cut in cuts:
        safe_cut = rewrite_vector_cut(cut, vector_vars, vector_mode)
        df = df.Filter(safe_cut)

    if weight and _has_column(df, weight):
        hist_ptr = df.Histo1D(
            (variable_to_plot, variable_to_plot, bins, xmin, xmax),
            variable_to_plot,
            weight
        )
    else:
        hist_ptr = df.Histo1D(
            (variable_to_plot, variable_to_plot, bins, xmin, xmax),
            variable_to_plot
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
                     background_file: str,
                     tree_name: str,
                     variable: str,
                     min_cut: float,
                     max_cut: float,
                     step: float,
                     base_cut: str = "",
                     vector_mode: str = "all",
                     weight: Optional[str] = None,
                     background_files: Optional[List[str]] = None) -> str:
    """Scan a variable cut range to find the value maximizing S/sqrt(S+B) significance."""

    bkg_paths = _parse_background_inputs(background_file, background_files)
    if not bkg_paths:
        return "Error: no background file(s) provided."

    sig_df = ROOT.RDataFrame(tree_name, signal_file)
    bkg_df = _build_dataframe(tree_name, bkg_paths)

    vector_vars = get_vector_branches(signal_file, tree_name)

    best_cut = None
    best_sig = -1
    best_S = 0
    best_B = 0

    import math
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

        full_cut = rewrite_vector_cut(full_cut, vector_vars, vector_mode)

        S = _filtered_yield(sig_df, full_cut, weight)
        B = _filtered_yield(bkg_df, full_cut, weight)

        significance = S / ((S + B) ** 0.5) if (S + B) > 0 else 0

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
    """Generate a cutflow table by sequentially applying cuts to a TTree."""

    paths = _parse_background_inputs(file_path, file_paths)
    if not paths:
        return "Error: no file(s) provided."

    df = _build_dataframe(tree_name, paths)

    vector_vars = get_vector_branches(paths[0], tree_name)

    results = []

    # initial count
    initial = _total_yield(df, weight)

    results.append(f"Initial events (files={len(paths)}): {initial}")

    current_df = df

    for cut in cuts:

        safe_cut = rewrite_vector_cut(cut, vector_vars, vector_mode)

        current_df = current_df.Filter(safe_cut)

        value = _total_yield(current_df, weight)

        results.append(f"{cut}: {value}")

    return "Cutflow:\n" + "\n".join(results)
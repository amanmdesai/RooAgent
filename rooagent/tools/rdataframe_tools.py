from typing import List, Dict, Optional
import ROOT
from langchain_core.tools import tool
import re
from .utils import *


@tool
def apply_cut_and_count(file_path: str, tree_name: str, cut: str,
                        vector_mode: str = "any",
                        weight: Optional[str] = None) -> str:
    """
    Apply a selection cut to a ROOT TTree and count the number of events passing the cut.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree inside the ROOT file.
    cut : str
        Selection string (e.g., "pt > 20 && abs(eta) < 2.5") to filter events.
    vector_mode options:
        "any" -> event passes if any object satisfies the cut
        "all" -> event passes if all objects satisfy the cut
    weight : str, optional
        Name of the branch containing event weights. If provided, the tool
        returns the sum of weights for events passing the cut instead of
        the raw event count.

    Returns
    -------
    str
        A message reporting the number of events (or weighted events)
        passing the cut.
    """

    vector_vars = get_vector_branches(file_path, tree_name)
    cut = rewrite_vector_cut(cut, vector_vars, vector_mode)

    df = ROOT.RDataFrame(tree_name, file_path)
    filtered = df.Filter(cut)

    if weight:
        value = filtered.Sum(weight).GetValue()
        return f"Weighted events passing cut '{cut}': {value}"
    else:
        value = filtered.Count().GetValue()
        return f"Events passing cut '{cut}': {value}"


@tool
def compute_significance(signal_file: str,
                         background_file: str,
                         tree_name: str,
                         cut: str,
                         vector_mode: str = "any",
                         weight: Optional[str] = None) -> str:
    """
    Compute the statistical significance S / sqrt(S + B) for a given selection cut.

    Parameters
    ----------
    signal_file : str
        Path to the ROOT file containing the signal TTree.
    background_file : str
        Path to the ROOT file containing the background TTree.
    tree_name : str
        Name of the TTree in both files.
    cut : str
        Selection string to filter events for both signal and background.
    vector_mode options:
        "any" -> event passes if any object satisfies the cut
        "all" -> event passes if all objects satisfy the cut
    weight : str, optional
        Name of the branch containing event weights. If provided, the
        significance will be computed using weighted yields.

    Returns
    -------
    str
        A message reporting the number of signal (S) and background (B)
        events passing the cut and the computed significance.
    """

    vector_vars = get_vector_branches(signal_file, tree_name)
    cut = rewrite_vector_cut(cut, vector_vars, vector_mode)

    sig_df = ROOT.RDataFrame(tree_name, signal_file).Filter(cut)
    bkg_df = ROOT.RDataFrame(tree_name, background_file).Filter(cut)

    if weight:
        S = sig_df.Sum(weight).GetValue()
        B = bkg_df.Sum(weight).GetValue()
    else:
        S = sig_df.Count().GetValue()
        B = bkg_df.Count().GetValue()

    if (S + B) <= 0:
        return "No events after cuts; significance undefined."

    significance = S / ((S + B) ** 0.5)

    return f"S={S}, B={B}, Significance={significance:.3f}"

@tool
def define_variable(
    file_path: str,
    tree_name: str,
    new_var_name: str,
    expression: str,
    save_file: Optional[str] = None
) -> str:
    """
    Define a new variable in a ROOT TTree using RDataFrame and save it.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree to modify.
    new_var_name : str
        Name of the new variable to define.
    expression : str
        Expression used to compute the new variable.
    save_file : str, optional
        Output ROOT file path. If None, a new file with suffix '_updated.root' is created.

    Returns
    -------
    str
        Confirmation message with the new variable added.
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
                             vector_mode: str = "any",
                             weight: Optional[str] = None) -> str:
    """
    Define new variables in a ROOT TTree, apply selection cuts, and plot a 1D histogram.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree inside the ROOT file.
    new_variables : dict
        Dictionary of variable definitions to add to the TTree.
    variable_to_plot : str
        Name of the variable to histogram after cuts.
    bins : int
        Number of bins in the histogram.
    xmin : float
        Lower edge of the histogram.
    xmax : float
        Upper edge of the histogram.
    cuts : list of str
        List of selection cuts to apply sequentially.
    output_file : str
        Path to save the resulting histogram.
    vector_mode options:
        "any" -> event passes if any object satisfies the cut
        "all" -> event passes if all objects satisfy the cut
    weight : str, optional
        Name of the branch containing event weights. If provided,
        the histogram will be filled using weighted events.

    Returns
    -------
    str
        Confirmation message indicating where the histogram was saved.
    """

    df = ROOT.RDataFrame(tree_name, file_path)
    vector_vars = get_vector_branches(file_path, tree_name)

    for name, expr in new_variables.items():
        df = df.Define(name, expr)

    for cut in cuts:
        safe_cut = rewrite_vector_cut(cut, vector_vars, vector_mode)
        df = df.Filter(safe_cut)

    if weight:
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

    canvas = ROOT.TCanvas("c1", "", 900, 700)
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
                     vector_mode: str = "any",
                     weight: Optional[str] = None) -> str:
    """
    Scan a cut on a variable and find the value that maximizes S/sqrt(S+B).

    Parameters
    ----------
    signal_file : str
        ROOT file containing the signal TTree.
    background_file : str
        ROOT file containing the background TTree.
    tree_name : str
        Name of the TTree in both files.
    variable : str
        Variable to scan.
    min_cut : float
        Minimum cut value to scan.
    max_cut : float
        Maximum cut value to scan.
    step : float
        Step size for the scan.
    base_cut : str, optional
        Additional fixed selection applied before scanning.
    vector_mode options:
        "any" -> event passes if any object satisfies the cut
        "all" -> event passes if all objects satisfy the cut
    weight : str, optional
        Name of the branch containing event weights. If provided,
        the scan uses weighted event yields for S and B.

    Returns
    -------
    str
        Report of the optimal cut value with corresponding S, B,
        and significance.
    """

    sig_df = ROOT.RDataFrame(tree_name, signal_file)
    bkg_df = ROOT.RDataFrame(tree_name, background_file)

    vector_vars = get_vector_branches(signal_file, tree_name)

    best_cut = None
    best_sig = -1
    best_S = 0
    best_B = 0

    cut_val = min_cut

    while cut_val <= max_cut:

        scan_cut = f"{variable} > {cut_val}"

        if base_cut:
            full_cut = f"({base_cut}) && ({scan_cut})"
        else:
            full_cut = scan_cut

        full_cut = rewrite_vector_cut(full_cut, vector_vars, vector_mode)

        sig_filtered = sig_df.Filter(full_cut)
        bkg_filtered = bkg_df.Filter(full_cut)

        if weight:
            S = sig_filtered.Sum(weight).GetValue()
            B = bkg_filtered.Sum(weight).GetValue()
        else:
            S = sig_filtered.Count().GetValue()
            B = bkg_filtered.Count().GetValue()

        significance = S / ((S + B) ** 0.5) if (S + B) > 0 else 0

        if significance > best_sig:
            best_sig = significance
            best_cut = cut_val
            best_S = S
            best_B = B

        cut_val += step

    return (
        f"Optimal cut found:\n"
        f"{variable} > {best_cut}\n"
        f"S = {best_S}, B = {best_B}\n"
        f"Significance = {best_sig:.3f}"
    )


@tool
def generate_cutflow(
    file_path: str,
    tree_name: str,
    cuts: List[str],
    vector_mode: str = "any",
    weight: Optional[str] = None
) -> str:
    """
    Generate a cutflow table by sequentially applying cuts to a ROOT TTree.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree.
    cuts : list of str
        List of selection cuts applied sequentially.
    vector_mode : str
        How vector branches are handled:
        "any" -> event passes if any object satisfies the cut
        "all" -> event passes if all objects satisfy the cut
    weight : str, optional
        Name of the branch containing event weights. If provided,
        weighted yields will be reported.

    Returns
    -------
    str
        A formatted cutflow table showing the number of events
        remaining after each selection.
    """

    df = ROOT.RDataFrame(tree_name, file_path)

    vector_vars = get_vector_branches(file_path, tree_name)

    results = []

    # initial count
    if weight:
        initial = df.Sum(weight).GetValue()
    else:
        initial = df.Count().GetValue()

    results.append(f"Initial events: {initial}")

    current_df = df

    for cut in cuts:

        safe_cut = rewrite_vector_cut(cut, vector_vars, vector_mode)

        current_df = current_df.Filter(safe_cut)

        if weight:
            value = current_df.Sum(weight).GetValue()
        else:
            value = current_df.Count().GetValue()

        results.append(f"{cut}: {value}")

    return "Cutflow:\n" + "\n".join(results)
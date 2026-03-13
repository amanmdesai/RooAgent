from typing import List, Dict, Optional
import ROOT
from langchain_core.tools import tool
import re

def get_vector_branches(file_path: str, tree_name: str) -> List[str]:
    """
    Detect vector branches in a TTree.
    """
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return []

    tree = f.Get(tree_name)
    if not tree:
        f.Close()
        return []

    vector_branches = []

    for branch in tree.GetListOfBranches():
        classname = branch.GetClassName()
        name = branch.GetName()

        if "vector" in classname:
            vector_branches.append(name)

    f.Close()
    return vector_branches

def rewrite_vector_cut(cut: str, vector_vars: List[str], mode: str = "any") -> str:
    """
    Rewrite selection cuts so comparisons on vector branches are wrapped
    with ROOT::VecOps reductions.

    mode:
        "any" -> ROOT::VecOps::Any()
        "all" -> ROOT::VecOps::All()

    Example:
        jet_pt > 30 -> ROOT::VecOps::Any(jet_pt > 30)
    """

    if mode not in ["any", "all"]:
        return cut

    reducer = "Any" if mode == "any" else "All"

    for var in vector_vars:

        pattern = rf"({var}\s*[<>!=]=?\s*[-+]?\d*\.?\d+)"

        matches = re.findall(pattern, cut)

        for match in matches:
            wrapped = f"ROOT::VecOps::{reducer}({match})"
            cut = cut.replace(match, wrapped)

    return cut


@tool
def apply_cut_and_count(file_path: str, tree_name: str, cut: str,
                        vector_mode: str = "any") -> str:
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
        
    Returns
    -------
    str
        A message reporting the number of events passing the cut, e.g.,
        "Events passing cut 'pt > 20': 12345".

    """


    vector_vars = get_vector_branches(file_path, tree_name)
    cut = rewrite_vector_cut(cut, vector_vars, vector_mode)

    df = ROOT.RDataFrame(tree_name, file_path)
    filtered = df.Filter(cut)

    count = filtered.Count().GetValue()

    return f"Events passing cut '{cut}': {count}"

@tool
def compute_significance(signal_file: str,
                         background_file: str,
                         tree_name: str,
                         cut: str,
                         vector_mode: str = "any") -> str:
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

    Returns
    -------
    str
        A message reporting the number of signal (S) and background (B) events
        passing the cut and the computed significance. Example:
        "S=100, B=400, Significance=4.472".

    """
    vector_vars = get_vector_branches(signal_file, tree_name)
    cut = rewrite_vector_cut(cut, vector_vars, vector_mode)

    sig_df = ROOT.RDataFrame(tree_name, signal_file)
    bkg_df = ROOT.RDataFrame(tree_name, background_file)

    S = sig_df.Filter(cut).Count().GetValue()
    B = bkg_df.Filter(cut).Count().GetValue()

    if B <= 0:
        return "Background is zero; significance undefined."

    significance = S / ((S + B) ** 0.5)

    return f"S={S}, B={B}, Significance={significance:.3f}"



# ================= DEFINE NEW TTree VARIABLE USING RDataFrame =================
@tool
def save_tree_variable(
    file_path: str,
    tree_name: str,
    new_var_name: str,
    expression: str,
    save_file: Optional[str] = None
) -> str:
    """
    Define a new variable in a ROOT TTree using RDataFrame and add it to the input files.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree to modify.
    new_var_name : str
        Name of the new variable to define.
    expression : str
        Expression to compute the new variable (ROOT C++/RDataFrame syntax).
        Example: "sqrt(px*px + py*py)" or "pt[0] + pt[1]".
    save_file : str, optional
        Path to save the updated ROOT file. If None, updates the original file in memory.
    vector_mode options:
        "any" -> event passes if any object satisfies the cut
        "all" -> event passes if all objects satisfy the cut

    Returns
    -------
    str
        Confirmation message with the new variable added.
    """
    # Open ROOT file
    f = ROOT.TFile.Open(file_path, "READ")
    if not f or f.IsZombie():
        return f"Error: Could not open file {file_path}"

    # Create RDataFrame
    rdf = ROOT.RDataFrame(tree_name, file_path)

    # Define new variable
    rdf = rdf.Define(new_var_name, expression)

    # Determine output file
    output_file = save_file if save_file else file_path

    # Save new tree (Snapshot)
    rdf.Snapshot(tree_name, output_file)

    f.Close()
    return f"New variable '{new_var_name}' defined in TTree '{tree_name}' and saved to '{output_file}'"

@tool
def define_variable_and_plot(file_path: str, tree_name: str,
                             new_variables: Dict[str, str],
                             variable_to_plot: str,
                             bins: int, xmin: float, xmax: float,
                             cuts: List[str],
                             output_file: str,
                             vector_mode: str = "any") -> str:
    """
    Define new variables in a ROOT TTree, apply selection cuts, and plot a 1D histogram.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree inside the ROOT file.
    new_variables : dict
        Dictionary of variable definitions to add to the TTree. Keys are
        the new variable names, values are ROOT expressions (e.g., "pt*0.001").
    variable_to_plot : str
        Name of the variable to histogram after cuts.
    bins : int
        Number of bins in the histogram.
    xmin : float
        Lower edge of the histogram.
    xmax : float
        Upper edge of the histogram.
    cuts : list of str
        List of selection cuts to apply sequentially. Each cut is a string.
    output_file : str
        Path to save the resulting histogram (e.g., "output.pdf").
    vector_mode options:
        "any" -> event passes if any object satisfies the cut
        "all" -> event passes if all objects satisfy the cut

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
                     vector_mode: str = "any") -> str:
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
        Variable to scan (e.g., "pt", "mass").
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

    Returns
    -------
    str
        Report of the optimal cut value with corresponding S, B, and significance.
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

        S = sig_df.Filter(full_cut).Count().GetValue()
        B = bkg_df.Filter(full_cut).Count().GetValue()

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
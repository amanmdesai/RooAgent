from typing import List
import ROOT
from langchain_core.tools import tool

@tool
def apply_cut_and_count(file_path: str, tree_name: str, cut: str) -> str:
    """
    Apply a selection cut to a ROOT TTree and count the remaining events.
    """
    df = ROOT.RDataFrame(tree_name, file_path)
    filtered = df.Filter(cut)
    count = filtered.Count().GetValue()
    return f"Events passing cut '{cut}': {count}"

@tool
def compute_significance(signal_file: str, background_file: str, tree_name: str, cut: str) -> str:
    """
    Compute statistical significance S / sqrt(S + B) for a cut.
    """
    sig_df = ROOT.RDataFrame(tree_name, signal_file)
    bkg_df = ROOT.RDataFrame(tree_name, background_file)
    S = sig_df.Filter(cut).Count().GetValue()
    B = bkg_df.Filter(cut).Count().GetValue()
    if B <= 0:
        return "Background is zero; significance undefined."
    significance = S / ((S + B) ** 0.5)
    return f"S={S}, B={B}, Significance={significance:.3f}"

@tool
def define_variable_and_plot(file_path: str, tree_name: str, new_variables: dict,
                             variable_to_plot: str, bins: int, xmin: float, xmax: float,
                             cuts: List[str], output_file: str) -> str:
    """
    Define new variables, apply cuts, and plot a histogram.
    """
    df = ROOT.RDataFrame(tree_name, file_path)
    for name, expr in new_variables.items():
        df = df.Define(name, expr)
    for cut in cuts:
        df = df.Filter(cut)
    hist_ptr = df.Histo1D((variable_to_plot, variable_to_plot, bins, xmin, xmax), variable_to_plot)
    h = hist_ptr.GetValue()
    ROOT.SetOwnership(h, False)
    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)
    h.GetXaxis().SetTitle(variable_to_plot)
    h.GetYaxis().SetTitle("Events")
    canvas = ROOT.TCanvas("c1", "", 900, 700)
    canvas.SetGrid()
    h.Draw("HIST")
    canvas.Update()
    canvas.SaveAs(output_file)
    return f"Plot saved to {output_file}"
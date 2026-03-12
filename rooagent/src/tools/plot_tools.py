from typing import List
import ROOT
from langchain_core.tools import tool


ROOT.gStyle.SetOptStat(0)

ROOT.gStyle.SetTitleFont(42, "XYZ")
ROOT.gStyle.SetLabelFont(42, "XYZ")

ROOT.gStyle.SetTitleSize(0.05, "XYZ")
ROOT.gStyle.SetLabelSize(0.04, "XYZ")

ROOT.gStyle.SetPadLeftMargin(0.12)
ROOT.gStyle.SetPadBottomMargin(0.12)

ROOT.gStyle.SetLegendBorderSize(0)
ROOT.gStyle.SetLegendFillColor(0)

ROOT.gStyle.SetFrameLineWidth(2)
ROOT.gStyle.SetLineWidth(2)

ROOT.gStyle.SetPadGridX(True)
ROOT.gStyle.SetPadGridY(True)


@tool
def plot_tree_variable(file_path: str, tree_name: str, variable: str,
                       bins: int, xmin: float, xmax: float,
                       output_pdf: str) -> str:
    """
    Plot a variable from a ROOT TTree and save it as an image.
    """
    df = ROOT.RDataFrame(tree_name, file_path)
    hist_ptr = df.Histo1D((variable, variable, bins, xmin, xmax), variable)
    h = hist_ptr.GetValue()
    ROOT.SetOwnership(h, False)
    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)
    h.GetXaxis().SetTitle(variable)
    h.GetYaxis().SetTitle("Events")
    canv = ROOT.TCanvas("c1", "", 900, 700)
    h.Draw("HIST")
    canv.SetGrid()
    canv.Update()
    canv.SaveAs(output_pdf)
    return f"Saved plot to {output_pdf}"

@tool
def compare_tree_variables(file_paths: List[str], tree_names: List[str],
                           variables: List[str], bins: int, xmin: float, xmax: float,
                           legends: List[str], output_pdf: str) -> str:
    """
    Compare variables from multiple TTrees on the same histogram.
    """
    canv = ROOT.TCanvas("c1", "Comparison", 900, 700)
    canv.SetGrid()
    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    colors = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]
    hist_list = []
    for i, (fpath, tname, var, label) in enumerate(zip(file_paths, tree_names, variables, legends)):
        df = ROOT.RDataFrame(tname, fpath)
        hist_ptr = df.Histo1D((f"h{i}", var, bins, xmin, xmax), var)
        h = hist_ptr.GetValue()
        ROOT.SetOwnership(h, False)
        h.SetDirectory(0)
        h.SetLineColor(colors[i % len(colors)])
        h.SetLineWidth(3)
        h.SetTitle("")
        h.GetXaxis().SetTitle(var)
        h.GetYaxis().SetTitle("Events")
        hist_list.append(h)
        legend.AddEntry(h, label, "l")
    if not hist_list:
        return "No histograms created."
    max_val = max(h.GetMaximum() for h in hist_list)
    for i, h in enumerate(hist_list):
        h.SetMaximum(max_val * 1.3)
        draw_opt = "HIST" if i == 0 else "HIST SAME"
        h.Draw(draw_opt)
    legend.Draw()
    canv.Update()
    canv.SaveAs(output_pdf)
    return f"Saved comparison histogram to {output_pdf}"

@tool
def draw_histograms_same_canvas(
    file_paths: List[str],
    hist_names: List[str],
    legends: List[str],
    output_pdf: str,
    xlabel: str = "",
    ylabel: str = "Events",
    logy: bool = False
) -> str:
    """
    Draw multiple ROOT histograms from different ROOT files on the same canvas
    and save the resulting plot to a PDF file with professional styling.

    Parameters
    ----------
    file_paths : List[str]
        List of ROOT file paths.
    hist_names : List[str]
        List of histogram names in the files.
    legends : List[str]
        Legend labels for each histogram.
    output_pdf : str
        Output PDF file path.
    xlabel : str, optional
        Label for the X-axis (default "").
    ylabel : str, optional
        Label for the Y-axis (default "Events").
    logy : bool, optional
        Whether to use a logarithmic Y-axis (default False).

    Returns
    -------
    str
        Confirmation message with saved file.
    """

    # Create canvas
    canv = ROOT.TCanvas("c1", "Combined Histograms", 1000, 700)
    canv.SetGridx()
    canv.SetGridy()
    if logy:
        canv.SetLogy(True)

    # Legend
    legend = ROOT.TLegend(0.65, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)

    # Color and line style palette
    colors = [ROOT.kBlue+1, ROOT.kRed+1, ROOT.kGreen+2, ROOT.kMagenta+1, ROOT.kOrange+7, ROOT.kCyan+2]

    hist_list = []

    # Load histograms
    for i, (fpath, hname, label) in enumerate(zip(file_paths, hist_names, legends)):
        f = ROOT.TFile.Open(fpath)
        if not f or f.IsZombie():
            continue

        h = f.Get(hname)
        if not h:
            f.Close()
            continue

        h.SetDirectory(0)
        h.SetLineColor(colors[i % len(colors)])
        h.SetLineWidth(3)
        h.SetLineStyle(1 + i % 4)  # Different line styles
        hist_list.append(h)
        legend.AddEntry(h, label, "l")
        f.Close()

    if not hist_list:
        return "Error: No histograms were found."

    # Determine max for scaling
    max_val = max(h.GetMaximum() for h in hist_list)
    for h in hist_list:
        h.SetMaximum(max_val * 1.3)
        h.GetXaxis().SetTitle(xlabel)
        h.GetYaxis().SetTitle(ylabel)
        h.GetXaxis().SetTitleFont(42)
        h.GetYaxis().SetTitleFont(42)
        h.GetXaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleSize(0.045)
        h.GetXaxis().SetLabelSize(0.04)
        h.GetYaxis().SetLabelSize(0.04)

    # Draw histograms
    for i, h in enumerate(hist_list):
        draw_opt = "HIST" if i == 0 else "HIST SAME"
        h.Draw(draw_opt)

    # Draw legend
    legend.Draw()

    canv.Update()
    canv.SaveAs(output_pdf)

    return f"Saved combined histogram plot to {output_pdf}"
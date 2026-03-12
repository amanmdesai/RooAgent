from typing import List
import ROOT
from langchain_core.tools import tool

# ================= ROOT STYLE =================
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

ROOT.gStyle.SetPadGridX(False)
ROOT.gStyle.SetPadGridY(False)

# ================= SINGLE VARIABLE =================
@tool
def plot_tree_variable(file_path: str, tree_name: str, variable: str,
                       bins: int, xmin: float, xmax: float,
                       output_pdf: str,
                       normalize: bool = False) -> str:
    """
    Plot a single variable from a ROOT TTree and save it as a histogram PDF.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree to read from the ROOT file.
    variable : str
        Name of the variable to histogram.
    bins : int
        Number of bins in the histogram.
    xmin : float
        Lower edge of the histogram range.
    xmax : float
        Upper edge of the histogram range.
    output_pdf : str
        Path to save the resulting histogram as a PDF file.
    normalize : bool, optional
        Whether to normalize the histogram to unit area (default False).

    Returns
    -------
    str
        Confirmation message indicating the file path of the saved PDF.
    """
    df = ROOT.RDataFrame(tree_name, file_path)
    hist_ptr = df.Histo1D((variable, variable, bins, xmin, xmax), variable)
    h = hist_ptr.GetValue()
    ROOT.SetOwnership(h, False)
    
    if normalize and h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)
    h.GetXaxis().SetTitle(variable)
    h.GetYaxis().SetTitle("Normalized Events" if normalize else "Events")

    canv = ROOT.TCanvas("c1", "", 900, 700)
    h.Draw("HIST")
    canv.Update()
    canv.SaveAs(output_pdf)
    return f"Saved plot to {output_pdf}"


# ================= COMPARE TREE VARIABLES =================
@tool
def compare_tree_variables(file_paths: List[str], tree_names: List[str],
                           variables: List[str], bins: int, xmin: float, xmax: float,
                           legends: List[str], output_pdf: str,
                           normalize: bool = False) -> str:
    """
    Compare the same or different variables from multiple ROOT TTrees on the same histogram.

    Parameters
    ----------
    file_paths : List[str]
        List of paths to ROOT files containing the TTrees.
    tree_names : List[str]
        List of TTree names corresponding to each ROOT file.
    variables : List[str]
        List of variable names to histogram from each TTree.
    bins : int
        Number of bins for all histograms.
    xmin : float
        Lower edge of the histogram range.
    xmax : float
        Upper edge of the histogram range.
    legends : List[str]
        List of labels for the legend corresponding to each histogram.
    output_pdf : str
        Path to save the resulting comparison histogram as a PDF file.
    normalize : bool, optional
        Whether to normalize histograms to unit area (default False).

    Returns
    -------
    str
        Confirmation message indicating the file path of the saved PDF.
    """
    canv = ROOT.TCanvas("c1", "Comparison", 900, 700)
    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    colors = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]
    hist_list = []

    for i, (fpath, tname, var, label) in enumerate(zip(file_paths, tree_names, variables, legends)):
        df = ROOT.RDataFrame(tname, fpath)
        hist_ptr = df.Histo1D((f"h{i}", var, bins, xmin, xmax), var)
        h = hist_ptr.GetValue()
        ROOT.SetOwnership(h, False)
        h.SetDirectory(0)

        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        h.SetLineColor(colors[i % len(colors)])
        h.SetLineWidth(3)
        h.SetTitle("")
        h.GetXaxis().SetTitle(var)
        h.GetYaxis().SetTitle("Normalized Events" if normalize else "Events")
        hist_list.append(h)
        legend.AddEntry(h, label, "l")

    if not hist_list:
        return "No histograms created."

    max_val = max(h.GetMaximum() for h in hist_list)
    for i, h in enumerate(hist_list):
        h.SetMaximum(max_val * 1.3)
        draw_opt = "HIST" if i == 0 else "HIST SAME"
        h.Draw(draw_opt)
    legend.SetFillStyle(0)
    legend.Draw()
    canv.Update()
    canv.SaveAs(output_pdf)
    return f"Saved comparison histogram to {output_pdf}"


# ================= DRAW HISTOGRAMS FROM FILES =================
@tool
def draw_histograms_same_canvas(
    file_paths: List[str],
    hist_names: List[str],
    legends: List[str],
    output_pdf: str,
    xlabel: str = "",
    ylabel: str = "Events",
    logy: bool = False,
    normalize: bool = False
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
    normalize : bool, optional
        Whether to normalize histograms to unit area (default False).

    Returns
    -------
    str
        Confirmation message with saved file.
    """
    canv = ROOT.TCanvas("c1", "Combined Histograms", 1000, 700)
    if logy:
        canv.SetLogy(True)

    legend = ROOT.TLegend(0.65, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)

    colors = [ROOT.kBlue+1, ROOT.kRed+1, ROOT.kGreen+2, ROOT.kMagenta+1, ROOT.kOrange+7, ROOT.kCyan+2]
    hist_list = []

    for i, (fpath, hname, label) in enumerate(zip(file_paths, hist_names, legends)):
        f = ROOT.TFile.Open(fpath)
        if not f or f.IsZombie():
            continue

        h = f.Get(hname)
        if not h:
            f.Close()
            continue

        h.SetDirectory(0)
        ROOT.SetOwnership(h, False)

        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        h.SetLineColor(colors[i % len(colors)])
        h.SetLineWidth(3)
        h.SetLineStyle(1 + i % 4)
        hist_list.append(h)
        legend.AddEntry(h, label, "l")
        f.Close()

    if not hist_list:
        return "Error: No histograms were found."

    max_val = max(h.GetMaximum() for h in hist_list)
    for h in hist_list:
        h.SetMaximum(max_val * 1.3)
        h.GetXaxis().SetTitle(xlabel)
        h.GetYaxis().SetTitle("Normalized Events" if normalize else ylabel)
        h.GetXaxis().SetTitleFont(42)
        h.GetYaxis().SetTitleFont(42)
        h.GetXaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleSize(0.045)
        h.GetXaxis().SetLabelSize(0.04)
        h.GetYaxis().SetLabelSize(0.04)

    for i, h in enumerate(hist_list):
        draw_opt = "HIST" if i == 0 else "HIST SAME"
        h.Draw(draw_opt)

    legend.SetFillStyle(0)
    legend.Draw()
    canv.Update()
    canv.SaveAs(output_pdf)

    return f"Saved combined histogram plot to {output_pdf}"


# ================= 2D HISTOGRAM =================
@tool
def draw_2d_histogram(
    file_path: str,
    hist_name: str,
    output_pdf: str,
    xlabel: str = "",
    ylabel: str = "",
    color_palette: int = 55,
    normalize: bool = False
) -> str:
    """
    Draw a 2D histogram from a ROOT file with professional styling and save as a PDF.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the histogram.
    hist_name : str
        Name of the 2D histogram inside the ROOT file.
    output_pdf : str
        Output PDF file path.
    xlabel : str, optional
        X-axis label (default "").
    ylabel : str, optional
        Y-axis label (default "").
    color_palette : int, optional
        ROOT color palette index (default 55).
    normalize : bool, optional
        Whether to normalize the histogram to unit area (default False).

    Returns
    -------
    str
        Confirmation message with the saved PDF path.
    """
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return f"Error: Could not open file {file_path}"

    h = f.Get(hist_name)
    if not h:
        f.Close()
        return f"Error: Histogram {hist_name} not found in file {file_path}"

    h.SetDirectory(0)
    ROOT.SetOwnership(h, False)

    if normalize and h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

    canv = ROOT.TCanvas("c1", "2D Histogram", 900, 700)

    h.GetXaxis().SetTitle(xlabel)
    h.GetYaxis().SetTitle(ylabel)
    h.GetXaxis().SetTitleFont(42)
    h.GetYaxis().SetTitleFont(42)
    h.GetXaxis().SetTitleSize(0.045)
    h.GetYaxis().SetTitleSize(0.045)
    h.GetXaxis().SetLabelSize(0.04)
    h.GetYaxis().SetLabelSize(0.04)

    ROOT.gStyle.SetPalette(color_palette)
    h.Draw("COLZ")

    canv.Update()
    canv.SaveAs(output_pdf)
    f.Close()

    return f"Saved 2D histogram to {output_pdf}"
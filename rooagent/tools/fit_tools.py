import ROOT
from langchain_core.tools import tool

@tool
def fit_tree_variable(
    file_path: str,
    tree_name: str,
    variable: str,
    bins: int,
    xmin: float,
    xmax: float,
    fit_function: str,
    output_plot: str
) -> str:
    """
    Create a histogram from a ROOT TTree variable, perform a statistical fit,
    and save the resulting plot with legend.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree to read.
    variable : str
        Name of the variable to histogram and fit.
    bins : int
        Number of bins in the histogram.
    xmin : float
        Minimum x-axis value.
    xmax : float
        Maximum x-axis value.
    fit_function : str
        ROOT fit function (examples: 'gaus', 'expo', 'pol1', 'pol2').
    output_plot : str
        File name where the canvas with histogram and fit will be saved.

    Returns
    -------
    str
        Summary of fit parameters and chi-square/NDF.
    """
    df = ROOT.RDataFrame(tree_name, file_path)
    hist_ptr = df.Histo1D((variable, variable, bins, xmin, xmax), variable)
    h = hist_ptr.GetValue()
    ROOT.SetOwnership(h, False)

    canvas = ROOT.TCanvas("c_fit", "", 900, 700)
    #canvas.SetGrid()

    # Perform fit
    h.Fit(fit_function, "S")

    # Style histogram
    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)
    h.SetMarkerStyle(20)
    h.SetMarkerSize(1.2)
    h.GetXaxis().SetTitle(variable)
    h.GetYaxis().SetTitle("Events")
    h.Draw("E")

    # Style fit function
    func = h.GetFunction(fit_function)
    if func:
        func.SetLineColor(ROOT.kRed)
        func.SetLineWidth(2)

    # Add legend
    legend = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(h, "Histogram", "lep")
    if func:
        legend.AddEntry(func, f"Fit: {fit_function}", "l")
    legend.Draw()

    canvas.Update()
    canvas.SaveAs(output_plot)

    # Extract fit parameters
    params = [f"p{i} = {func.GetParameter(i):.4f}" for i in range(func.GetNpar())]
    chi2 = func.GetChisquare()
    ndf = func.GetNDF()

    return (
        f"Fit function: {fit_function}\n"
        f"Parameters: {', '.join(params)}\n"
        f"Chi2/NDF: {chi2:.3f}/{ndf}"
    )


@tool
def fit_histogram(
    file_path: str,
    hist_name: str,
    fit_function: str,
    output_plot: str
) -> str:
    """
    Fit an existing histogram in a ROOT file, overlay the fit, and save with legend.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the histogram.
    hist_name : str
        Name of the histogram to fit.
    fit_function : str
        ROOT fit function (examples: 'gaus', 'expo', 'pol1', 'pol2').
    output_plot : str
        File name where the canvas with histogram and fit will be saved.

    Returns
    -------
    str
        Summary of fit parameters and chi-square/NDF.
    """
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return "Error: could not open ROOT file."

    h = f.Get(hist_name)
    if not h:
        f.Close()
        return f"Histogram '{hist_name}' not found."

    ROOT.SetOwnership(h, False)

    canvas = ROOT.TCanvas("c_fit_hist", "", 900, 700)
    #canvas.SetGrid()

    # Perform fit
    h.Fit(fit_function, "S")

    # Style histogram
    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)
    h.SetMarkerStyle(20)
    h.SetMarkerSize(1.2)
    h.Draw("E")

    # Style fit function
    func = h.GetFunction(fit_function)
    if func:
        func.SetLineColor(ROOT.kRed)
        func.SetLineWidth(2)

    # Add legend
    legend = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(h, "Histogram", "lep")
    if func:
        legend.AddEntry(func, f"Fit: {fit_function}", "l")
    legend.Draw()

    canvas.Update()
    canvas.SaveAs(output_plot)

    # Extract fit parameters
    params = [f"p{i}={func.GetParameter(i):.4f}" for i in range(func.GetNpar())]
    chi2 = func.GetChisquare()
    ndf = func.GetNDF()

    f.Close()

    return (
        f"Histogram fit completed.\n"
        f"Function: {fit_function}\n"
        f"Parameters: {', '.join(params)}\n"
        f"Chi2/NDF: {chi2:.3f}/{ndf}"
    )
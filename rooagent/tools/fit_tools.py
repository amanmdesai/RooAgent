import ROOT
from langchain_core.tools import tool


# Module-level counter for unique canvas names
_canvas_counter = [0]


def _unique_canvas_name(base: str) -> str:
    _canvas_counter[0] += 1
    return f"{base}_{_canvas_counter[0]}"

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
    """USE WHEN: histogram a TTree variable and fit it with a ROOT function (e.g. 'gaus','expo'). Saves plot and returns fit parameters and Chi2/NDF."""
    df = ROOT.RDataFrame(tree_name, file_path)
    hist_ptr = df.Histo1D((variable, variable, bins, xmin, xmax), variable)
    h = hist_ptr.GetValue()
    # Detach histogram from file ownership so it survives after file close
    h.SetDirectory(0)
    ROOT.SetOwnership(h, False)

    canvas = ROOT.TCanvas(_unique_canvas_name("c_fit"), "", 900, 700)
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
    del hist_ptr  # safe to release after saving

    if func is None:
        return f"Plot saved to {output_plot}, but fit function '{fit_function}' not found after fitting."

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
    """USE WHEN: fit a pre-existing TH1 in a ROOT file with a ROOT function (e.g. 'gaus'). Saves overlay plot and returns fit parameters and Chi2/NDF."""
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return "Error: could not open ROOT file."

    h = f.Get(hist_name)
    if not h:
        f.Close()
        return f"Histogram '{hist_name}' not found."

    # Detach from file so the histogram is safe after closing the file
    h.SetDirectory(0)
    ROOT.SetOwnership(h, False)

    canvas = ROOT.TCanvas(_unique_canvas_name("c_fit_hist"), "", 900, 700)
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

    if func is None:
        f.Close()
        return f"Plot saved to {output_plot}, but fit function '{fit_function}' not found after fitting."

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
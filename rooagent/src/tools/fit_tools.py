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
    Create a histogram from a ROOT TTree variable and perform a statistical fit.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str
        Name of the TTree.
    variable : str
        Variable to histogram and fit.
    bins : int
        Number of histogram bins.
    xmin : float
        Minimum x-axis value.
    xmax : float
        Maximum x-axis value.
    fit_function : str
        ROOT fit function (examples: 'gaus', 'expo', 'pol1', 'pol2').
    output_plot : str
        File name where the plot with fit will be saved.

    Returns
    -------
    str
        Summary of fit parameters and goodness-of-fit.
    """

    # Create dataframe
    df = ROOT.RDataFrame(tree_name, file_path)

    # Create histogram
    hist_ptr = df.Histo1D((variable, variable, bins, xmin, xmax), variable)
    h = hist_ptr.GetValue()
    ROOT.SetOwnership(h, False)

    # Canvas
    canvas = ROOT.TCanvas("c_fit", "", 900, 700)
    canvas.SetGrid()

    # Fit
    fit_result = h.Fit(fit_function, "S")

    # Draw histogram
    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)
    h.GetXaxis().SetTitle(variable)
    h.GetYaxis().SetTitle("Events")

    h.Draw("E")

    canvas.Update()
    canvas.SaveAs(output_plot)

    # Extract fit info
    func = h.GetFunction(fit_function)

    params = []
    for i in range(func.GetNpar()):
        params.append(f"p{i} = {func.GetParameter(i):.4f}")

    chi2 = func.GetChisquare()
    ndf = func.GetNDF()

    result = (
        f"Fit function: {fit_function}\n"
        f"Parameters: {', '.join(params)}\n"
        f"Chi2/NDF: {chi2:.3f}/{ndf}"
    )

    return result


@tool
def fit_histogram(
    file_path: str,
    hist_name: str,
    fit_function: str,
    output_plot: str
) -> str:
    """
    Fit an existing histogram stored in a ROOT file.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the histogram.
    hist_name : str
        Name of the histogram to fit.
    fit_function : str
        ROOT fitting function (e.g. 'gaus', 'expo', 'pol1', 'pol2').
    output_plot : str
        File name where the plot with the fit result will be saved.

    Returns
    -------
    str
        Summary of fit parameters and chi-square.
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
    canvas.SetGrid()

    # Perform fit
    fit_result = h.Fit(fit_function, "S")

    # Draw histogram
    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)
    h.Draw("E")

    canvas.Update()
    canvas.SaveAs(output_plot)

    func = h.GetFunction(fit_function)

    params = []
    for i in range(func.GetNpar()):
        params.append(f"p{i}={func.GetParameter(i):.4f}")

    chi2 = func.GetChisquare()
    ndf = func.GetNDF()

    f.Close()

    return (
        f"Histogram fit completed.\n"
        f"Function: {fit_function}\n"
        f"Parameters: {', '.join(params)}\n"
        f"Chi2/NDF: {chi2:.3f}/{ndf}"
    )
import ROOT
from langchain_core.tools import tool
from .utils import _unique_canvas_name


@tool
def fit_distribution(
    source: str,
    fit_function: str,
    output_plot: str,
    file_path: str,
    tree_name: str = "",
    variable: str = "",
    bins: int = 50,
    xmin: float = 0.0,
    xmax: float = 1.0,
    hist_name: str = "",
) -> str:
    """Fit a ROOT function to a 1D distribution obtained from a TTree branch or a stored histogram.

    Supports two data sources:
    - **'tree'**: builds a histogram from a TTree branch on-the-fly, then fits it.
    - **'hist'**: reads an existing TH1 from a ROOT file and fits it directly.

    The fit is performed using ROOT's TH1::Fit and the result is plotted to a canvas saved
    as `output_plot`. Both fitted parameter values and goodness-of-fit (chi²/ndf) are returned.

    Args:
        source (str): Data source — 'tree' (branch-to-histogram) or 'hist' (stored TH1).
        fit_function (str): ROOT TF1 function name or expression. Built-in names include
            'gaus', 'landau', 'expo', 'pol0' … 'pol9'. Custom expressions use standard
            C++ syntax, e.g. "[0]*exp(-[1]*x)".
        output_plot (str): Path to save the fitted plot canvas (PDF or PNG).
        file_path (str): ROOT file containing the source data.
        tree_name (str, optional): TTree name — required when source='tree'.
        variable (str, optional): Branch name to histogram — required when source='tree'.
        bins (int, optional): Number of bins for the histogram when source='tree' (default 50).
        xmin (float, optional): Lower x bound for the histogram when source='tree' (default 0.0).
        xmax (float, optional): Upper x bound for the histogram when source='tree' (default 1.0).
        hist_name (str, optional): Stored histogram name — required when source='hist'.

    Returns:
        str: "Fit <function> [<label>]: p0=<v>, p1=<v>, ... chi2/ndf=<chi2>/<ndf>" on success,
            or a descriptive error string when required arguments are missing or the fit fails.

    Notes:
        - Choose `xmin`/`xmax` to cover the distribution of interest; fitting outside the
          data range produces unreliable parameter estimates.
        - For narrow resonance fits (e.g. a Gaussian peak), set bins to 30-100 and
          xmin/xmax to a few widths around the expected peak.
    """
    source_key = (source or "").strip().lower()

    if source_key == "tree":
        if not tree_name or not variable:
            return "Error: tree_name and variable are required when source='tree'."

        df = ROOT.RDataFrame(tree_name, file_path)
        hist_ptr = df.Histo1D((variable, variable, bins, xmin, xmax), variable)
        h = hist_ptr.GetValue()
        h.SetDirectory(0)
        ROOT.SetOwnership(h, False)
        label = variable
    elif source_key == "hist":
        if not hist_name:
            return "Error: hist_name is required when source='hist'."

        f = ROOT.TFile.Open(file_path)
        if not f or f.IsZombie():
            return "Error: could not open ROOT file."

        h = f.Get(hist_name)
        if not h:
            f.Close()
            return f"Histogram '{hist_name}' not found."

        h.SetDirectory(0)
        ROOT.SetOwnership(h, False)
        f.Close()
        label = hist_name
    else:
        return "Error: source must be 'tree' or 'hist'."

    canvas = ROOT.TCanvas(_unique_canvas_name("c_fit"), "", 900, 700)

    h.Fit(fit_function, "S")

    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)
    h.SetMarkerStyle(20)
    h.SetMarkerSize(1.2)
    if source_key == "tree":
        h.GetXaxis().SetTitle(variable)
        h.GetYaxis().SetTitle("Events")
    h.Draw("E")

    func = h.GetFunction(fit_function)
    if func:
        func.SetLineColor(ROOT.kRed)
        func.SetLineWidth(2)

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
        return f"Fit '{fit_function}' not found after fitting -> {output_plot}"

    params = [f"p{i}={func.GetParameter(i):.4f}" for i in range(func.GetNpar())]
    chi2 = func.GetChisquare()
    ndf = func.GetNDF()
    return f"Fit {fit_function} [{label}]: {', '.join(params)} chi2/ndf={chi2:.3f}/{ndf}"

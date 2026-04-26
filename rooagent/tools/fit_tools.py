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
    """Fit a ROOT function to a TTree branch or stored TH1 and save a plot.

    Args:
        source: 'tree' — builds histogram from tree_name+variable, then fits.
                'hist' — reads hist_name from file_path and fits directly.
        fit_function: ROOT built-in ('gaus', 'landau', 'expo', 'pol0'..'pol9') or C++ expression.
        output_plot: Path to save the output plot.
        file_path: ROOT file path.
        tree_name: TTree name (required when source='tree').
        variable: Branch to histogram (required when source='tree').
        bins / xmin / xmax: Histogram binning for source='tree'.
        hist_name: Histogram name (required when source='hist').

    Returns: \"Fit <func> [<label>]: p0=<v>, ... chi2/ndf=<v>/<v>\" or error string.
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

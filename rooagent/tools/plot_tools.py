from typing import List, Optional
import ROOT
from langchain_core.tools import tool


def _create_plot_pads(canvas_name: str, canvas_title: str, show_ratio: bool):
    canvas = ROOT.TCanvas(canvas_name, canvas_title, 900, 800 if show_ratio else 700)
    if not show_ratio:
        return canvas, None, None

    upper_pad = ROOT.TPad(f"{canvas_name}_upper", "", 0.0, 0.30, 1.0, 1.0)
    lower_pad = ROOT.TPad(f"{canvas_name}_lower", "", 0.0, 0.00, 1.0, 0.30)

    upper_pad.SetBottomMargin(0.02)
    lower_pad.SetTopMargin(0.04)
    lower_pad.SetBottomMargin(0.32)
    lower_pad.SetGridy(True)

    upper_pad.Draw()
    lower_pad.Draw()
    return canvas, upper_pad, lower_pad


def _style_ratio_hist(ratio_hist, x_title: str):
    ratio_hist.SetTitle("")
    ratio_hist.GetXaxis().SetTitle(x_title)
    ratio_hist.GetYaxis().SetTitle("Ratio")
    ratio_hist.GetYaxis().SetNdivisions(505)
    ratio_hist.GetXaxis().SetTitleSize(0.12)
    ratio_hist.GetYaxis().SetTitleSize(0.10)
    ratio_hist.GetXaxis().SetLabelSize(0.10)
    ratio_hist.GetYaxis().SetLabelSize(0.08)
    ratio_hist.GetYaxis().SetTitleOffset(0.45)
    ratio_hist.GetXaxis().SetTitleOffset(1.05)
    ratio_hist.SetMinimum(0.0)
    ratio_hist.SetMaximum(2.0)


def _build_reference_ratio_hists(hist_list: List):
    if len(hist_list) < 2:
        return []

    reference = hist_list[0]
    ratio_hists = []
    for i, hist in enumerate(hist_list[1:], start=1):
        ratio = hist.Clone(f"{hist.GetName()}_ratio_{i}")
        ratio.SetDirectory(0)
        ROOT.SetOwnership(ratio, False)
        ratio.Divide(reference)
        ratio_hists.append(ratio)

    return ratio_hists


def _build_signal_background_ratio(signal_hist, background_hists: List):
    if signal_hist is None or not background_hists:
        return None

    summed_background = background_hists[0].Clone(f"{signal_hist.GetName()}_bkg_sum")
    summed_background.SetDirectory(0)
    ROOT.SetOwnership(summed_background, False)

    for hist in background_hists[1:]:
        summed_background.Add(hist)

    ratio = signal_hist.Clone(f"{signal_hist.GetName()}_over_bkg_ratio")
    ratio.SetDirectory(0)
    ROOT.SetOwnership(ratio, False)
    ratio.Divide(summed_background)
    return ratio


def _parse_file_inputs(
    primary_file: Optional[str] = None,
    additional_files: Optional[List[str]] = None,
) -> List[str]:
    """Normalize one-or-many file inputs into a clean list of paths."""
    parsed: List[str] = []

    if primary_file:
        parsed.extend([p.strip() for p in primary_file.split(",") if p.strip()])

    if additional_files:
        parsed.extend([p.strip() for p in additional_files if p and p.strip()])

    return list(dict.fromkeys(parsed))


def _draw_ratio_panel(hist_list: List, lower_pad, x_title: str, ratio_hists: List = None):
    if lower_pad is None:
        return

    if ratio_hists is None:
        ratio_hists = _build_reference_ratio_hists(hist_list)

    if not ratio_hists:
        return

    lower_pad.cd()

    for i, ratio in enumerate(ratio_hists):
        _style_ratio_hist(ratio, x_title)
        ratio.Draw("HIST" if i == 0 else "HIST SAME")

    x_axis = ratio_hists[0].GetXaxis()
    unity = ROOT.TLine(x_axis.GetXmin(), 1.0, x_axis.GetXmax(), 1.0)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw()


def _draw_overlay_plot(hist_list: List, legend, output_pdf: str, x_title: str,
                       y_title: str, canvas_name: str, canvas_title: str,
                       show_ratio: bool = False, logy: bool = False,
                       ratio_hists: List = None,
                       draw_options: List[str] = None) -> str:
    if not hist_list:
        return "No histograms created."

    canvas, upper_pad, lower_pad = _create_plot_pads(canvas_name, canvas_title, show_ratio)
    draw_pad = upper_pad if upper_pad else canvas
    draw_pad.cd()

    if logy:
        draw_pad.SetLogy(True)

    max_val = max(h.GetMaximum() for h in hist_list)
    for i, hist in enumerate(hist_list):
        hist.SetMaximum(max_val * (15.0 if logy else 1.35))
        hist.GetXaxis().SetTitle(x_title)
        hist.GetYaxis().SetTitle(y_title)
        if show_ratio:
            hist.GetXaxis().SetLabelSize(0)
            hist.GetXaxis().SetTitleSize(0)
            hist.GetYaxis().SetTitleSize(0.055)
            hist.GetYaxis().SetLabelSize(0.045)
        draw_option = draw_options[i] if draw_options and i < len(draw_options) else ("HIST" if i == 0 else "HIST SAME")
        hist.Draw(draw_option)

    legend.SetFillStyle(0)
    legend.Draw()

    _draw_ratio_panel(hist_list, lower_pad, x_title, ratio_hists=ratio_hists)

    canvas.Update()
    canvas.SaveAs(output_pdf)
    return output_pdf


# ================= DRAW SINGLE 1D HISTOGRAM =================
@tool
def draw_1d_histogram(
    file_path: str,
    hist_name: str,
    output_pdf: str,
    xlabel: str = "",
    ylabel: str = "Events",
    logy: bool = False,
    normalize: bool = False,
    line_color: int = ROOT.kBlue+1
) -> str:
    """
    Draw a 1D histogram from a ROOT file and save it to PDF.

    Supports axis labels, optional log-y scaling, and optional normalization
    to unit area before drawing.
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

    h.SetLineColor(line_color)
    h.SetLineWidth(3)
    h.GetXaxis().SetTitle(xlabel)
    h.GetYaxis().SetTitle("Normalized Events" if normalize else ylabel)
    h.GetXaxis().SetTitleFont(42)
    h.GetYaxis().SetTitleFont(42)
    h.GetXaxis().SetTitleSize(0.045)
    h.GetYaxis().SetTitleSize(0.045)
    h.GetXaxis().SetLabelSize(0.04)
    h.GetYaxis().SetLabelSize(0.04)

    canv = ROOT.TCanvas("c1", "1D Histogram", 900, 700)
    if logy:
        canv.SetLogy(True)

    h.Draw("HIST")
    canv.Update()
    canv.SaveAs(output_pdf)
    f.Close()

    return f"Saved 1D histogram {hist_name} to {output_pdf}"

@tool
def plot_tree_variable(file_path: str, tree_name: str, variable: str,
                       bins: int, xmin: float, xmax: float,
                       output_pdf: str,
                       normalize: bool = False) -> str:
    """
    Plot one TTree variable as a 1D histogram and save to PDF.

    The histogram range is controlled by bins/xmin/xmax and can be normalized
    to compare shape differences across samples.
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
def compare_tree_variables(file_paths: List[str], tree_name: str,
                           variables: List[str], bins: int, xmin: float, xmax: float,
                           legends: List[str], output_pdf: str,
                           normalize: bool = False,
                           show_ratio: bool = False) -> str:
    """
    Overlay variables from multiple ROOT files/trees on one canvas.

    Inputs are paired by index: file_paths[i], variables[i], legends[i].
    Optionally normalizes each histogram and draws a ratio panel.
    """
    if not (len(file_paths) == len(variables) == len(legends)):
        return "Error: file_paths, variables, and legends must have the same length."

    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    colors = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]
    hist_list = []

    for i, (fpath, var, label) in enumerate(zip(file_paths, variables, legends)):
        df = ROOT.RDataFrame(tree_name, fpath)
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

    result = _draw_overlay_plot(
        hist_list=hist_list,
        legend=legend,
        output_pdf=output_pdf,
        x_title=variables[0],
        y_title="Normalized Events" if normalize else "Events",
        canvas_name="c_compare",
        canvas_title="Comparison",
        show_ratio=show_ratio,
    )
    if result == "No histograms created.":
        return result
    return f"Saved comparison histogram to {output_pdf}"


@tool
def plot_signal_vs_backgrounds(
    signal_file: str,
    background_files: List[str],
    tree_name: str,
    variable: str,
    bins: int,
    xmin: float,
    xmax: float,
    output_pdf: str,
    signal_label: str = "Signal",
    signal_files: List[str] = None,
    signal_labels: List[str] = None,
    background_labels: List[str] = None,
    data_file: str = "",
    data_label: str = "Data",
    plot_data: bool = False,
    normalize: bool = False,
    show_ratio: bool = False
) -> str:
    """
    Plot HEP-style overlays for signal, backgrounds, and optional data.

    Visual convention: backgrounds are filled, signals are line-only, and data
    is drawn with markers+errors. Supports multiple signal/background files.
    """
    signal_paths = _parse_file_inputs(signal_file, signal_files)
    if not signal_paths:
        return "Error: at least one signal file must be provided via signal_file or signal_files."

    background_paths = _parse_file_inputs(additional_files=background_files)
    if not background_paths:
        return "Error: background_files cannot be empty."

    if signal_labels is None:
        if len(signal_paths) == 1:
            signal_labels = [signal_label]
        else:
            signal_labels = [f"{signal_label} {i + 1}" for i in range(len(signal_paths))]

    if len(signal_labels) != len(signal_paths):
        return "Error: number of signal_labels must match number of signal files."

    if background_labels is None:
        background_labels = [f"Background {i + 1}" for i in range(len(background_paths))]

    if len(background_labels) != len(background_paths):
        return "Error: number of background_labels must match number of background_files."

    wants_data = plot_data or bool(data_file.strip())
    if wants_data and not data_file.strip():
        return "Error: data_file must be provided when plot_data is True."

    legend = ROOT.TLegend(0.62, 0.68, 0.88, 0.88)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)

    signal_line_colors = [ROOT.kBlack, ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kMagenta + 1, ROOT.kGreen + 2, ROOT.kOrange + 7]
    background_fill_colors = [ROOT.kAzure - 9, ROOT.kOrange - 2, ROOT.kSpring - 6, ROOT.kPink - 8, ROOT.kViolet - 6, ROOT.kCyan - 6]

    hist_list = []
    draw_options: List[str] = []
    signal_hists = []
    background_hists = []

    for i, (fpath, label) in enumerate(zip(background_paths, background_labels)):
        df = ROOT.RDataFrame(tree_name, fpath)
        hist_ptr = df.Histo1D((f"h_sigbkg_bkg_{i}", variable, bins, xmin, xmax), variable)
        h = hist_ptr.GetValue()
        ROOT.SetOwnership(h, False)
        h.SetDirectory(0)

        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        fill_color = background_fill_colors[i % len(background_fill_colors)]
        h.SetFillColor(fill_color)
        h.SetLineColor(fill_color)
        h.SetLineWidth(2)
        h.SetLineStyle(1)
        h.SetTitle("")
        h.GetXaxis().SetTitle(variable)
        h.GetYaxis().SetTitle("Normalized Events" if normalize else "Events")
        background_hists.append(h)
        hist_list.append(h)
        draw_options.append("HIST" if len(draw_options) == 0 else "HIST SAME")
        legend.AddEntry(h, label, "f")

    for i, (fpath, label) in enumerate(zip(signal_paths, signal_labels)):
        df = ROOT.RDataFrame(tree_name, fpath)
        hist_ptr = df.Histo1D((f"h_sigbkg_sig_{i}", variable, bins, xmin, xmax), variable)
        h = hist_ptr.GetValue()
        ROOT.SetOwnership(h, False)
        h.SetDirectory(0)

        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        line_color = signal_line_colors[i % len(signal_line_colors)]
        h.SetFillStyle(0)
        h.SetLineColor(line_color)
        h.SetLineWidth(4)
        h.SetLineStyle(1 + (i % 4))
        h.SetTitle("")
        h.GetXaxis().SetTitle(variable)
        h.GetYaxis().SetTitle("Normalized Events" if normalize else "Events")
        signal_hists.append(h)
        hist_list.append(h)
        draw_options.append("HIST" if len(draw_options) == 0 else "HIST SAME")
        legend.AddEntry(h, label, "l")

    if wants_data:
        df = ROOT.RDataFrame(tree_name, data_file)
        hist_ptr = df.Histo1D(("h_sigbkg_data", variable, bins, xmin, xmax), variable)
        data_hist = hist_ptr.GetValue()
        ROOT.SetOwnership(data_hist, False)
        data_hist.SetDirectory(0)

        if normalize and data_hist.Integral() > 0:
            data_hist.Scale(1.0 / data_hist.Integral())

        data_hist.SetFillStyle(0)
        data_hist.SetLineColor(ROOT.kBlack)
        data_hist.SetMarkerColor(ROOT.kBlack)
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerSize(1.0)
        data_hist.SetTitle("")
        data_hist.GetXaxis().SetTitle(variable)
        data_hist.GetYaxis().SetTitle("Normalized Events" if normalize else "Events")
        hist_list.append(data_hist)
        draw_options.append("E1" if len(draw_options) == 0 else "E1 SAME")
        legend.AddEntry(data_hist, data_label, "lep")

    ratio_hist = _build_signal_background_ratio(signal_hists[0] if signal_hists else None, background_hists)

    result = _draw_overlay_plot(
        hist_list=hist_list,
        legend=legend,
        output_pdf=output_pdf,
        x_title=variable,
        y_title="Normalized Events" if normalize else "Events",
        canvas_name="c_sig_bkg",
        canvas_title="Signal vs Backgrounds",
        show_ratio=show_ratio,
        ratio_hists=[ratio_hist] if show_ratio and ratio_hist is not None else None,
        draw_options=draw_options,
    )
    if result == "No histograms created.":
        return result

    return f"Saved signal-vs-background comparison to {output_pdf}"


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
    normalize: bool = False,
    show_ratio: bool = False
) -> str:
    """
    Draw existing histograms from multiple ROOT files on a shared canvas.

    Histogram names and legend labels are matched by index. Supports optional
    normalization, log-y rendering, and a ratio panel.
    """
    if not (len(file_paths) == len(hist_names) == len(legends)):
        return "Error: file_paths, hist_names, and legends must have the same length."

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

    x_axis_title = xlabel if xlabel else hist_list[0].GetXaxis().GetTitle()
    result = _draw_overlay_plot(
        hist_list=hist_list,
        legend=legend,
        output_pdf=output_pdf,
        x_title=x_axis_title,
        y_title="Normalized Events" if normalize else ylabel,
        canvas_name="c_same_canvas",
        canvas_title="Combined Histograms",
        show_ratio=show_ratio,
        logy=logy,
    )
    if result == "No histograms created.":
        return "Error: No histograms were found."

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
    Draw a 2D histogram from a ROOT file and save to PDF.

    Allows custom axis labels, ROOT color palette selection, and optional
    normalization to unit integral.
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


@tool
def draw_2d_histogram_from_tree(
    file_path: str,
    tree_name: str,
    x_branch: str,
    y_branch: str,
    output_pdf: str,
    bins_x: int = 50,
    xmin: float = 0,
    xmax: float = 100,
    bins_y: int = 50,
    ymin: float = 0,
    ymax: float = 100,
    xlabel: str = "",
    ylabel: str = "",
    color_palette: int = 55
) -> str:
    """
    Build and draw a 2D histogram directly from two TTree branches.

    Uses explicit x/y binning and ranges, then saves a COLZ-style plot to PDF
    with optional custom axis labels and palette.
    """

    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return f"Error: Could not open file {file_path}"

    tree = f.Get(tree_name)
    if not tree:
        f.Close()
        return f"Error: TTree {tree_name} not found in {file_path}"

    h2 = ROOT.TH2F(
        "h2",
        "",
        bins_x, xmin, xmax,
        bins_y, ymin, ymax
    )

    tree.Draw(
        f"{y_branch}:{x_branch} >> h2",
        "",
        "goff"
    )

    canv = ROOT.TCanvas("c1", "2D Histogram", 900, 700)

    h2.GetXaxis().SetTitle(xlabel if xlabel else x_branch)
    h2.GetYaxis().SetTitle(ylabel if ylabel else y_branch)

    ROOT.gStyle.SetPalette(color_palette)

    h2.Draw("COLZ")

    canv.Update()
    canv.SaveAs(output_pdf)

    f.Close()

    return f"Saved 2D histogram ({y_branch} vs {x_branch}) to {output_pdf}"
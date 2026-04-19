from pathlib import Path
from typing import List, Optional
import ROOT
from langchain_core.tools import tool
import re


# Module-level counter ensures every TCanvas gets a unique name, preventing
# ROOT from silently deleting an existing canvas (and its objects) when a new
# one with the same name is created.
_canvas_counter = [0]


def _unique_canvas_name(base: str) -> str:
    _canvas_counter[0] += 1
    return f"{base}_{_canvas_counter[0]}"


def _safe_get_vector_branches(file_path: str, tree_name: str) -> List[str]:
    """Safely obtain vector branch names; return empty list if ROOT/imports unavailable."""
    try:
        from .utils import get_vector_branches as _get_vector_branches

        return _get_vector_branches(file_path, tree_name)
    except Exception:
        return []


def _rewrite_vector_cut_local(cut: str, vector_vars: List[str], mode: str = "any") -> str:
    """Rewrite comparisons on vector branches using ROOT::VecOps reducers.

    This is kept local to avoid importing the whole `utils` module (which
    imports ROOT at top-level) during package installation.
    """
    if mode not in ["any", "all"]:
        return cut

    # Normalize Python boolean literals → C++ and logical operators → C++
    cut = re.sub(r"\bTrue\b", "true", cut)
    cut = re.sub(r"\bFalse\b", "false", cut)
    cut = re.sub(r"\band\b", "&&", cut)
    cut = re.sub(r"\bor\b", "||", cut)

    reducer = "Any" if mode == "any" else "All"

    for var in vector_vars:
        pattern = rf"({var}\s*[<>!=]=?\s*[-+]?\d*\.?\d+)"
        matches = re.findall(pattern, cut)

        for match in matches:
            wrapped = f"ROOT::VecOps::{reducer}({match})"
            cut = cut.replace(match, wrapped)

    return cut


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


def _resolve_weight_branch(df, weight_branch: str) -> str:
    """Return a valid weight branch name for a dataframe, or empty if unavailable."""
    if not weight_branch:
        return ""

    columns = [str(c) for c in df.GetColumnNames()]
    return weight_branch if weight_branch in columns else ""
def _maybe_rebin_hist(hist, rebin: int):
    """Return a rebinned histogram when `rebin` > 1, otherwise return original.

    Uses TH1.Rebin to create a new histogram with grouped bins. If `rebin`
    is 1 or invalid, the original histogram is returned unchanged.
    """
    try:
        r = int(rebin) if rebin is not None else 1
    except Exception:
        r = 1
    if r <= 1 or hist is None:
        return hist
    newname = f"{hist.GetName()}_rebin{r}"
    try:
        hreb = hist.Rebin(r, newname)
        ROOT.SetOwnership(hreb, False)
        hreb.SetDirectory(0)
        return hreb
    except Exception:
        return hist


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
    ROOT.SetOwnership(unity, False)
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

    canvas, upper_pad, lower_pad = _create_plot_pads(_unique_canvas_name(canvas_name), canvas_title, show_ratio)
    draw_pad = upper_pad if upper_pad else canvas
    draw_pad.cd()

    if logy:
        draw_pad.SetLogy(True)

    max_val = max(h.GetMaximum() for h in hist_list)
    prepared_draws = []
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
        prepared_draws.append((hist, draw_option))

    # Draw filled/line histograms first, then marker-style overlays (e.g. data) on top.
    non_overlay_draws = []
    marker_overlay_draws = []
    for hist, draw_option in prepared_draws:
        opt = draw_option.upper()
        is_marker_overlay = ("HIST" not in opt) and (("E" in opt) or ("P" in opt))
        if is_marker_overlay:
            marker_overlay_draws.append((hist, draw_option))
        else:
            non_overlay_draws.append((hist, draw_option))

    draw_sequence = non_overlay_draws + marker_overlay_draws
    for i, (hist, draw_option) in enumerate(draw_sequence):
        option = draw_option
        if i > 0 and "SAME" not in option.upper():
            option = f"{option} SAME"
        hist.Draw(option)

    legend.SetFillStyle(0)
    legend.Draw()

    _draw_ratio_panel(hist_list, lower_pad, x_title, ratio_hists=ratio_hists)

    canvas.Update()
    canvas.SaveAs(output_pdf)
    return output_pdf


@tool
def plot_1d(
    mode: str,
    output_pdf: str,
    file_path: str = "",
    file_paths: Optional[List[str]] = None,
    tree_name: str = "",
    variable: str = "",
    variables: Optional[List[str]] = None,
    hist_name: str = "",
    hist_names: Optional[List[str]] = None,
    legends: Optional[List[str]] = None,
    bins: int = 50,
    xmin: float = 0.0,
    xmax: float = 1.0,
    xlabel: str = "",
    ylabel: str = "Events",
    normalize: bool = False,
    logy: bool = False,
    show_ratio: bool = False,
    weight_branch: str = "",
    cuts: Optional[List[str]] = None,
    rebin: int = 1,
    vector_mode: str = "all",
) -> str:
    """Unified 1D plotting tool.

    Modes
    -----
    - `hist`: draw one TH1 from one file.
    - `tree`: draw one branch from one tree/file.
    - `tree_compare`: compare multiple tree variables/files.
    - `hist_compare`: compare multiple existing TH1 histograms.

    Returns
    -------
    str
        Saved PDF path message or a clear error string.
    """
    mode_key = (mode or "").strip().lower()

    if mode_key == "hist":
        if not file_path or not hist_name:
            return "Error: file_path and hist_name are required for mode='hist'."
        return draw_1d_histogram.invoke(
            {
                "file_path": file_path,
                "hist_name": hist_name,
                "output_pdf": output_pdf,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "logy": logy,
                "rebin": rebin,
                "normalize": normalize,
            }
        )

    if mode_key == "tree":
        if not file_path or not tree_name or not variable:
            return "Error: file_path, tree_name, and variable are required for mode='tree'."
        return plot_tree_variable.invoke(
            {
                "file_path": file_path,
                "tree_name": tree_name,
                "variable": variable,
                "bins": bins,
                "xmin": xmin,
                "xmax": xmax,
                "output_pdf": output_pdf,
                "normalize": normalize,
                "weight_branch": weight_branch,
                "rebin": rebin,
                "cuts": cuts,
                "vector_mode": vector_mode,
            }
        )

    if mode_key == "tree_compare":
        if not file_paths or not tree_name or not variables or not legends:
            return "Error: file_paths, tree_name, variables, and legends are required for mode='tree_compare'."
        return compare_tree_variables.invoke(
            {
                "file_paths": file_paths,
                "tree_name": tree_name,
                "variables": variables,
                "bins": bins,
                "xmin": xmin,
                "xmax": xmax,
                "legends": legends,
                "output_pdf": output_pdf,
                "normalize": normalize,
                "show_ratio": show_ratio,
                "rebin": rebin,
                "weight_branch": weight_branch,
                "cuts": cuts,
                "vector_mode": vector_mode,
            }
        )

    if mode_key == "hist_compare":
        if not file_paths or not hist_names or not legends:
            return "Error: file_paths, hist_names, and legends are required for mode='hist_compare'."
        return draw_histograms_same_canvas.invoke(
            {
                "file_paths": file_paths,
                "hist_names": hist_names,
                "legends": legends,
                "output_pdf": output_pdf,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "logy": logy,
                "normalize": normalize,
                "rebin": rebin,
                "show_ratio": show_ratio,
            }
        )

    return "Error: unsupported mode. Use one of: hist, tree, tree_compare, hist_compare."


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
    rebin: int = 1,
    line_color: int = ROOT.kBlue+1
) -> str:
    """Draw a 1D histogram stored in a ROOT file and save it as a PDF.

    Parameters
    ----------
    file_path : str
        Path to the input ROOT file.
    hist_name : str
        Name of the TH1 object inside the ROOT file.
    output_pdf : str
        Output path for the saved PDF.
    xlabel : str
        X-axis label for the plot.
    ylabel : str
        Y-axis label for the plot.
    logy : bool
        If True, use a logarithmic y-axis.
    normalize : bool
        If True, scale the histogram to unit integral before drawing.
    line_color : int
        ROOT color constant for the line.

    Returns
    -------
    str
        Success message or an error string when the file or histogram
        cannot be read.
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
    # Optionally rebin the histogram when requested
    h = _maybe_rebin_hist(h, rebin)

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

    canv = ROOT.TCanvas(_unique_canvas_name("c1"), "1D Histogram", 900, 700)
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
                       rebin: int = 1,
                       normalize: bool = False,
                       weight_branch: str = "",
                       cuts: Optional[List[str]] = None,
                       vector_mode: str = "all") -> str:
    """Build and draw a 1D histogram from a TTree branch and save as a PDF.

    Parameters
    ----------
    file_path : str
        Path to the input ROOT file.
    tree_name : str
        Name of the TTree to read.
    variable : str
        Branch name to histogram.
    bins, xmin, xmax : int/float
        Binning parameters for the histogram.
    output_pdf : str
        Path where the PDF will be written.
    normalize : bool
        If True, normalize the histogram to unit integral.
    weight_branch : str
        Name of the MC weight branch to use for MC histograms. If empty,
        unit weights are used (and data is never weighted).
    cuts : list of str, optional
        Selection cuts applied BEFORE filling the histogram. Use C++ syntax
        (&&, ||, true, false). All event-selection cuts should be forwarded here.
    vector_mode : str
        Reducer mode for vector branches when used in cuts: 'any' or 'all'.

    Returns
    -------
    str
        Success message or an error string.
    """
    df = ROOT.RDataFrame(tree_name, file_path)

    if cuts:
        vector_vars = _safe_get_vector_branches(file_path, tree_name)
        for cut in cuts:
            df = df.Filter(_rewrite_vector_cut_local(cut, vector_vars, vector_mode))

    resolved_weight = _resolve_weight_branch(df, weight_branch)
    if resolved_weight:
        wname = resolved_weight
    else:
        wname = "__rooagent_unit_weight"
        df = df.Define(wname, "1.0")

    hist_ptr = df.Histo1D((variable, variable, bins, xmin, xmax), variable, wname)
    h = hist_ptr.GetValue()
    ROOT.SetOwnership(h, False)
    # Optionally rebin
    h = _maybe_rebin_hist(h, rebin)
    
    if normalize and h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

    h.SetLineWidth(3)
    h.SetLineColor(ROOT.kBlue + 1)
    h.GetXaxis().SetTitle(variable)
    h.GetYaxis().SetTitle("Normalized Events" if normalize else "Events")

    canv = ROOT.TCanvas(_unique_canvas_name("c1"), "", 900, 700)
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
                           show_ratio: bool = False,
                           rebin: int = 1,
                           weight_branch: str = "",
                           cuts: Optional[List[str]] = None,
                           vector_mode: str = "all") -> str:
    """Compare multiple TTree variable distributions overlaid on a single canvas
    and save as a PDF.

    Parameters
    ----------
    file_paths : list[str]
        List of input ROOT file paths to histogram (one per variable entry).
    tree_name : str
        Name of the TTree in each file.
    variables : list[str]
        List of branch names to histogram (one per file).
    bins, xmin, xmax : int/float
        Binning specification for the histograms.
    legends : list[str]
        Labels for the legend, one per file.
    output_pdf : str
        Path to write the PDF.
    normalize : bool
        If True, scale each histogram to unit integral before drawing.
    show_ratio : bool
        If True, adds a ratio panel below the main plot (each histogram divided by the first).
    weight_branch : str
        Name of MC weight branch to apply for MC inputs. Leave empty for data.
    cuts : list[str], optional
        Selection cuts to apply before histogramming; use C++ syntax.
    vector_mode : str
        Reducer used for vector-branch cuts: 'any' or 'all'.

    Returns
    -------
    str
        Success message or an error string.
    """
    if not (len(file_paths) == len(variables) == len(legends)):
        return "Error: file_paths, variables, and legends must have the same length."

    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    colors = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]
    hist_list = []

    for i, (fpath, var, label) in enumerate(zip(file_paths, variables, legends)):
        df = ROOT.RDataFrame(tree_name, fpath)
        if cuts:
            vector_vars = _safe_get_vector_branches(fpath, tree_name)
            for cut in cuts:
                df = df.Filter(_rewrite_vector_cut_local(cut, vector_vars, vector_mode))
        resolved_weight = _resolve_weight_branch(df, weight_branch)
        if resolved_weight:
            wname = resolved_weight
        else:
            wname = "__rooagent_unit_weight"
            df = df.Define(wname, "1.0")

        hist_ptr = df.Histo1D((f"h{i}", var, bins, xmin, xmax), var, wname)
        h = hist_ptr.GetValue()
        ROOT.SetOwnership(h, False)
        # Optionally rebin each histogram
        h = _maybe_rebin_hist(h, rebin)
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
    show_ratio: bool = False,
    rebin: int = 1,
    weight_branch: str = "",
    cuts: Optional[List[str]] = None,
    vector_mode: str = "all",
    apply_cuts_before_plot: bool = False
) -> str:
    """Plot signal, background, and optional data histograms in HEP style and save as a PDF.

    This tool produces publication-style plots with optional stacking of backgrounds,
    hatched signal fills, optional data overlay, and an optional ratio panel. When
    used in analysis flows, always forward the same event-selection cuts used in the
    cutflow/significance calculations via the `cuts` parameter.

    Parameters
    ----------
    signal_file, signal_files : str or list[str]
        One or more signal ROOT files. At least one signal source is required.
    background_files : list[str]
        One or more background ROOT files. Backgrounds are summed/stacked as requested.
    tree_name : str
        Name of the TTree to read in each file.
    variable : str
        Branch name to histogram.
    bins, xmin, xmax : int/float
        Binning for the histogram.
    output_pdf : str
        Output path for the resulting PDF.
    weight_branch : str
        Name of the MC weight branch applied to all MC inputs. Leave empty for data.
    cuts : list[str], optional
        Selection cuts applied to each process before histogramming; use C++ syntax
        (&&, ||, true, false).
    show_ratio : bool
        If True, add a ratio panel (Data/MC or Sig/Bkg) below the main plot.
    normalize : bool
        If True, draw normalized line-overlays instead of stacked counts.

    Returns
    -------
    str
        Success message or an error string.
    """

    if apply_cuts_before_plot and not cuts:
        return "Error: apply_cuts_before_plot=True but no 'cuts' provided. Please provide a non-empty cuts list."
    signal_paths = _parse_file_inputs(signal_file, signal_files)
    if not signal_paths:
        return "Error: at least one signal file must be provided via signal_file or signal_files."

    background_paths = _parse_file_inputs(additional_files=background_files)
    if not background_paths:
        return "Error: background_files cannot be empty."

    if signal_labels is None:
        if len(signal_paths) == 1:
            signal_labels = [Path(signal_paths[0]).stem if signal_label == "Signal" else signal_label]
        else:
            signal_labels = [Path(p).stem for p in signal_paths]

    if len(signal_labels) != len(signal_paths):
        return "Error: number of signal_labels must match number of signal files."

    if background_labels is None:
        background_labels = [Path(p).stem for p in background_paths]

    if len(background_labels) != len(background_paths):
        return "Error: number of background_labels must match number of background_files."

    wants_data = plot_data or bool(data_file.strip())
    if wants_data and not data_file.strip():
        return "Error: data_file must be provided when plot_data is True."

    signal_line_colors = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]
    # Hatched fill styles for signal in the stack (distinguishable from solid background fills)
    signal_hatch_styles = [3345, 3354, 3395, 3444, 3490]
    background_fill_colors = [ROOT.kAzure - 9, ROOT.kOrange - 2, ROOT.kSpring - 6, ROOT.kPink - 8, ROOT.kViolet - 6, ROOT.kCyan - 6]

    background_hists = []
    signal_hists = []
    data_hist = None

    # --- Build background histograms ---
    for i, (fpath, _label) in enumerate(zip(background_paths, background_labels)):
        df = ROOT.RDataFrame(tree_name, fpath)
        if cuts and apply_cuts_before_plot:
            vector_vars = _safe_get_vector_branches(fpath, tree_name)
            for cut in cuts:
                safe_cut = _rewrite_vector_cut_local(cut, vector_vars, vector_mode)
                df = df.Filter(safe_cut)

        resolved_weight = _resolve_weight_branch(df, weight_branch)
        if resolved_weight:
            wname = resolved_weight
        else:
            wname = "__rooagent_unit_weight"
            df = df.Define(wname, "1.0")

        hist_ptr = df.Histo1D((f"h_sigbkg_bkg_{i}", variable, bins, xmin, xmax), variable, wname)
        h = hist_ptr.GetValue()
        ROOT.SetOwnership(h, False)
        # Optionally rebin background histograms
        h = _maybe_rebin_hist(h, rebin)
        h.SetDirectory(0)
        fill_color = background_fill_colors[i % len(background_fill_colors)]
        h.SetFillColor(fill_color)
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(1)
        h.SetTitle("")
        background_hists.append(h)

    # --- Build signal histograms ---
    for i, (fpath, _label) in enumerate(zip(signal_paths, signal_labels)):
        df = ROOT.RDataFrame(tree_name, fpath)
        if cuts and apply_cuts_before_plot:
            vector_vars = _safe_get_vector_branches(fpath, tree_name)
            for cut in cuts:
                safe_cut = _rewrite_vector_cut_local(cut, vector_vars, vector_mode)
                df = df.Filter(safe_cut)

        resolved_weight = _resolve_weight_branch(df, weight_branch)
        if resolved_weight:
            wname = resolved_weight
        else:
            wname = "__rooagent_unit_weight"
            df = df.Define(wname, "1.0")

        hist_ptr = df.Histo1D((f"h_sigbkg_sig_{i}", variable, bins, xmin, xmax), variable, wname)
        h = hist_ptr.GetValue()
        ROOT.SetOwnership(h, False)
        # Optionally rebin signal histograms
        h = _maybe_rebin_hist(h, rebin)
        h.SetDirectory(0)
        line_color = signal_line_colors[i % len(signal_line_colors)]
        # Default style: line only (used when no data / normalize mode)
        h.SetFillStyle(0)
        h.SetLineColor(line_color)
        h.SetLineWidth(3)
        h.SetLineStyle(1 + (i % 4))
        h.SetTitle("")
        signal_hists.append(h)

    # --- Build data histogram (pure overlay – never part of the stack) ---
    if wants_data:
        df = ROOT.RDataFrame(tree_name, data_file)
        if cuts and apply_cuts_before_plot:
            vector_vars = _safe_get_vector_branches(data_file, tree_name)
            for cut in cuts:
                safe_cut = _rewrite_vector_cut_local(cut, vector_vars, vector_mode)
                df = df.Filter(safe_cut)

        # Data should not be weighted by MC weights; always use unit weight for data overlay
        wname = "__rooagent_unit_weight"
        df = df.Define(wname, "1.0")
        hist_ptr = df.Histo1D(("h_sigbkg_data", variable, bins, xmin, xmax), variable, wname)
        data_hist = hist_ptr.GetValue()
        ROOT.SetOwnership(data_hist, False)
        # Optionally rebin data histogram
        data_hist = _maybe_rebin_hist(data_hist, rebin)
        data_hist.SetDirectory(0)
        data_hist.SetFillStyle(0)
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerSize(1.0)
        data_hist.SetMarkerColor(ROOT.kBlack)
        data_hist.SetLineColor(ROOT.kBlack)
        data_hist.SetLineWidth(1)
        data_hist.SetTitle("")

    # --- Normalization ---
    # When normalizing, stacking is not meaningful — switch all processes to line style.
    y_title = "Normalized Events" if normalize else "Events"
    if normalize:
        for i, h in enumerate(background_hists):
            if h.Integral() > 0:
                h.Scale(1.0 / h.Integral())
            h.SetFillStyle(0)
            h.SetLineColor(background_fill_colors[i % len(background_fill_colors)])
            h.SetLineWidth(2)
        for h in signal_hists:
            if h.Integral() > 0:
                h.Scale(1.0 / h.Integral())
        if data_hist and data_hist.Integral() > 0:
            data_hist.Scale(1.0 / data_hist.Integral())

    # Decide whether to use a stack: only when data is present and not normalizing.
    use_stack = wants_data and not normalize

    # When stacking, re-style signal histograms to use hatched fills (distinguishable from backgrounds).
    if use_stack:
        for i, h in enumerate(signal_hists):
            line_color = signal_line_colors[i % len(signal_line_colors)]
            h.SetFillStyle(signal_hatch_styles[i % len(signal_hatch_styles)])
            h.SetFillColor(line_color)
            h.SetLineColor(line_color)
            h.SetLineWidth(2)
            h.SetLineStyle(1)

    # --- Legend: data first, then signal in reverse stack order, then backgrounds in reverse stack order ---
    legend = ROOT.TLegend(0.62, 0.68, 0.88, 0.88)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)
    if data_hist:
        legend.AddEntry(data_hist, data_label, "lep")
    sig_legend_opt = "f" if use_stack else "l"
    for h, label in reversed(list(zip(signal_hists, signal_labels))):
        legend.AddEntry(h, label, sig_legend_opt)
    bkg_legend_opt = "l" if normalize else "f"
    for h, label in reversed(list(zip(background_hists, background_labels))):
        legend.AddEntry(h, label, bkg_legend_opt)

    # --- Canvas and pads ---
    canvas, upper_pad, lower_pad = _create_plot_pads(_unique_canvas_name("c_sig_bkg"), "Signal vs Backgrounds", show_ratio)
    draw_pad = upper_pad if upper_pad else canvas
    draw_pad.cd()

    if not use_stack:
        # Pure line overlay: no data present, or normalize=True
        all_hists = background_hists + signal_hists + ([data_hist] if data_hist else [])
        max_val = max(h.GetMaximum() for h in all_hists)
        drawn = False
        for h in background_hists:
            h.GetXaxis().SetTitle(variable)
            h.GetYaxis().SetTitle(y_title)
            if show_ratio:
                h.GetXaxis().SetLabelSize(0)
                h.GetXaxis().SetTitleSize(0)
                h.GetYaxis().SetTitleSize(0.055)
                h.GetYaxis().SetLabelSize(0.045)
            h.SetMaximum(max_val * 1.35)
            h.Draw("HIST" if not drawn else "HIST SAME")
            drawn = True
        for h in signal_hists:
            h.GetXaxis().SetTitle(variable)
            h.GetYaxis().SetTitle(y_title)
            if show_ratio:
                h.GetXaxis().SetLabelSize(0)
                h.GetXaxis().SetTitleSize(0)
            h.Draw("HIST SAME" if drawn else "HIST")
            drawn = True
        if data_hist:
            data_hist.GetXaxis().SetTitle(variable)
            data_hist.GetYaxis().SetTitle(y_title)
            data_hist.Draw("PE1 SAME" if drawn else "PE1")
    else:
        # Stacked MC (backgrounds first, then signal on top) + data overlay
        stack = ROOT.THStack("hs_sig_bkg", "")
        ROOT.SetOwnership(stack, False)
        for h in background_hists:
            stack.Add(h)
        for h in signal_hists:
            stack.Add(h)

        stack.Draw("HIST")
        # Axis titles must be set after the first Draw() for THStack
        stack.GetXaxis().SetTitle(variable)
        stack.GetYaxis().SetTitle(y_title)
        if show_ratio:
            stack.GetXaxis().SetLabelSize(0)
            stack.GetXaxis().SetTitleSize(0)
            stack.GetYaxis().SetTitleSize(0.055)
            stack.GetYaxis().SetLabelSize(0.045)

        stack_max = stack.GetMaximum()
        data_max = data_hist.GetMaximum() if data_hist else 0.0
        stack.SetMaximum(max(stack_max, data_max) * 1.35)

        # --- MC stat uncertainty hatched band (drawn on top of stack, below data) ---
        all_mc_hists = background_hists + signal_hists
        mc_stat_band = all_mc_hists[0].Clone("h_mc_stat_band")
        mc_stat_band.SetDirectory(0)
        ROOT.SetOwnership(mc_stat_band, False)
        for h in all_mc_hists[1:]:
            mc_stat_band.Add(h)
        mc_stat_band.SetFillStyle(3354)   # hatched pattern
        mc_stat_band.SetFillColor(ROOT.kGray + 2)
        mc_stat_band.SetLineColor(ROOT.kGray + 2)
        mc_stat_band.SetMarkerSize(0)
        mc_stat_band.Draw("E2 SAME")

        # Data is a pure overlay — drawn last, on top of everything
        if data_hist:
            data_hist.Draw("PE1 SAME")

    legend.SetFillStyle(0)
    legend.Draw()

    # --- Ratio panel ---
    if show_ratio and lower_pad is not None:
        # Build total MC denominator (bkg + signal when stacking, bkg-only otherwise)
        all_mc = background_hists + (signal_hists if use_stack else [])
        total_mc = all_mc[0].Clone("h_total_mc_for_ratio")
        total_mc.SetDirectory(0)
        ROOT.SetOwnership(total_mc, False)
        for h in all_mc[1:]:
            total_mc.Add(h)

        if data_hist:
            # Data/MC ratio: standard HEP publication convention
            # Use TH1.Divide(num, denom) with both hists to propagate stat errors correctly
            ratio = data_hist.Clone("h_data_mc_ratio")
            ratio.SetDirectory(0)
            ROOT.SetOwnership(ratio, False)
            ratio.Divide(data_hist, total_mc, 1.0, 1.0, "B")
            ratio_draw_opt = "PE1"
            ratio_y_title = "Data/MC"

            # MC stat uncertainty band in ratio panel (total_mc / total_mc = 1 ± rel err)
            ratio_mc_band = total_mc.Clone("h_ratio_mc_band")
            ratio_mc_band.SetDirectory(0)
            ROOT.SetOwnership(ratio_mc_band, False)
            for b in range(1, ratio_mc_band.GetNbinsX() + 1):
                c = ratio_mc_band.GetBinContent(b)
                e = ratio_mc_band.GetBinError(b)
                ratio_mc_band.SetBinContent(b, 1.0)
                ratio_mc_band.SetBinError(b, e / c if c > 0 else 0.0)
            ratio_mc_band.SetFillStyle(3354)
            ratio_mc_band.SetFillColor(ROOT.kGray + 2)
            ratio_mc_band.SetLineColor(ROOT.kGray + 2)
            ratio_mc_band.SetMarkerSize(0)
        else:
            # No data: Signal/Bkg
            ratio = signal_hists[0].Clone("h_sig_bkg_ratio")
            ratio.SetDirectory(0)
            ROOT.SetOwnership(ratio, False)
            ratio.Divide(total_mc)
            ratio_draw_opt = "HIST"
            ratio_y_title = "Sig/Bkg"
            ratio_mc_band = None

        lower_pad.cd()
        _style_ratio_hist(ratio, variable)
        ratio.GetYaxis().SetTitle(ratio_y_title)
        if ratio_mc_band:
            ratio_mc_band.GetXaxis().SetTitle(variable)
            ratio_mc_band.GetYaxis().SetTitle(ratio_y_title)
            ratio_mc_band.SetMinimum(0.0)
            ratio_mc_band.SetMaximum(2.0)
            ratio_mc_band.Draw("E2")
            ratio.Draw(f"{ratio_draw_opt} SAME")
        else:
            ratio.Draw(ratio_draw_opt)

        x_axis = ratio.GetXaxis()
        unity = ROOT.TLine(x_axis.GetXmin(), 1.0, x_axis.GetXmax(), 1.0)
        ROOT.SetOwnership(unity, False)
        unity.SetLineStyle(2)
        unity.SetLineColor(ROOT.kGray + 2)
        unity.Draw()

    canvas.Update()
    canvas.SaveAs(output_pdf)
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
    show_ratio: bool = False,
    rebin: int = 1
) -> str:
    """Overlay TH1 histograms from multiple ROOT files on a single canvas and save as PDF.

    Parameters
    ----------
    file_paths : list[str]
        List of ROOT file paths containing the histograms.
    hist_names : list[str]
        Names of the TH1 objects to draw from each file.
    legends : list[str]
        Labels for the legend, one per histogram.
    output_pdf : str
        Path to write the combined PDF.
    xlabel, ylabel, logy, normalize : options
        Drawing options controlling axis labels, log scale and normalization.
    show_ratio : bool
        If True, add a ratio panel below the main plot (each histogram divided by the first).

    Returns
    -------
    str
        Success message or an error string.
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
    skipped = []

    for i, (fpath, hname, label) in enumerate(zip(file_paths, hist_names, legends)):
        f = ROOT.TFile.Open(fpath)
        if not f or f.IsZombie():
            skipped.append(f"could not open {fpath}")
            continue

        h = f.Get(hname)
        if not h:
            f.Close()
            skipped.append(f"histogram '{hname}' not found in {fpath}")
            continue

        h.SetDirectory(0)
        ROOT.SetOwnership(h, False)
        # Optionally rebin histograms loaded from files
        h = _maybe_rebin_hist(h, rebin)

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

    msg = f"Saved combined histogram plot to {output_pdf}"
    if skipped:
        msg += "\nWarnings:\n" + "\n".join(f"  - {s}" for s in skipped)
    return msg


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
    """Draw a 2D TH2 histogram from a ROOT file and save it as a PDF.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TH2.
    hist_name : str
        Name of the TH2 object inside the file.
    output_pdf : str
        Output path for the saved PDF.
    xlabel, ylabel : str
        Axis labels for the plot.
    color_palette : int
        ROOT palette index to use when drawing the COLZ plot.
    normalize : bool
        If True, scale the histogram to unit integral before drawing.

    Returns
    -------
    str
        Success message or an error string.
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

    canv = ROOT.TCanvas(_unique_canvas_name("c1"), "2D Histogram", 900, 700)

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
    """Build a 2D histogram from two TTree branches and save a COLZ plot as PDF.

    Parameters
    ----------
    file_path : str
        Path to the input ROOT file containing the TTree.
    tree_name : str
        Name of the TTree inside the file.
    x_branch, y_branch : str
        Branch names to use for the X and Y axes respectively.
    output_pdf : str
        Output path for the saved PDF.
    bins_x, xmin, xmax, bins_y, ymin, ymax : int/float
        Binning parameters for the histogram.
    xlabel, ylabel : str
        Axis labels for the plot.
    color_palette : int
        ROOT palette index for COLZ drawing.

    Returns
    -------
    str
        Success message or an error string.
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

    canv = ROOT.TCanvas(_unique_canvas_name("c1"), "2D Histogram", 900, 700)

    h2.GetXaxis().SetTitle(xlabel if xlabel else x_branch)
    h2.GetYaxis().SetTitle(ylabel if ylabel else y_branch)

    ROOT.gStyle.SetPalette(color_palette)

    h2.Draw("COLZ")

    canv.Update()
    canv.SaveAs(output_pdf)

    f.Close()

    return f"Saved 2D histogram ({y_branch} vs {x_branch}) to {output_pdf}"
from typing import List, Optional
import ROOT
from langchain_core.tools import tool
import re


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

    canvas, upper_pad, lower_pad = _create_plot_pads(canvas_name, canvas_title, show_ratio)
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
    """Draw a 1D histogram stored in a ROOT file and save as PDF."""
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
                       normalize: bool = False,
                       weight_branch: str = "") -> str:
    """Build and draw a 1D histogram from a TTree branch, with optional event weights, and save as PDF."""
    df = ROOT.RDataFrame(tree_name, file_path)
    resolved_weight = _resolve_weight_branch(df, weight_branch)
    if resolved_weight:
        hist_ptr = df.Histo1D((variable, variable, bins, xmin, xmax), variable, resolved_weight)
    else:
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
                           show_ratio: bool = False,
                           weight_branch: str = "") -> str:
    """Compare multiple TTree variable distributions overlaid on one canvas, with optional event weights."""
    if not (len(file_paths) == len(variables) == len(legends)):
        return "Error: file_paths, variables, and legends must have the same length."

    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    colors = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]
    hist_list = []

    for i, (fpath, var, label) in enumerate(zip(file_paths, variables, legends)):
        df = ROOT.RDataFrame(tree_name, fpath)
        resolved_weight = _resolve_weight_branch(df, weight_branch)
        if resolved_weight:
            hist_ptr = df.Histo1D((f"h{i}", var, bins, xmin, xmax), var, resolved_weight)
        else:
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
    show_ratio: bool = False,
    weight_branch: str = "",
    cuts: Optional[List[str]] = None,
    vector_mode: str = "all",
    apply_cuts_before_plot: bool = True
) -> str:
    """Plot signal, backgrounds, and optional data in HEP style with optional event weights.

    Stacking behaviour:
      - With data (and normalize=False): backgrounds are stacked first, then signals stacked on
        top (hatched fill to distinguish from backgrounds). Data is drawn last as black
        filled-circle markers (PE1) — it is never part of the stack.
        Ratio panel shows Data / total-MC (backgrounds + signal).
      - Without data (or normalize=True): all processes are drawn as overlaid line histograms
        with no stacking. Ratio panel shows Signal / Background.

    Legend order: data -> signal (reverse stack) -> backgrounds (reverse stack).
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
            hist_ptr = df.Histo1D((f"h_sigbkg_bkg_{i}", variable, bins, xmin, xmax), variable, resolved_weight)
        else:
            hist_ptr = df.Histo1D((f"h_sigbkg_bkg_{i}", variable, bins, xmin, xmax), variable)
        h = hist_ptr.GetValue()
        ROOT.SetOwnership(h, False)
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
            hist_ptr = df.Histo1D((f"h_sigbkg_sig_{i}", variable, bins, xmin, xmax), variable, resolved_weight)
        else:
            hist_ptr = df.Histo1D((f"h_sigbkg_sig_{i}", variable, bins, xmin, xmax), variable)
        h = hist_ptr.GetValue()
        ROOT.SetOwnership(h, False)
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

        resolved_weight = _resolve_weight_branch(df, weight_branch)
        if resolved_weight:
            hist_ptr = df.Histo1D(("h_sigbkg_data", variable, bins, xmin, xmax), variable, resolved_weight)
        else:
            hist_ptr = df.Histo1D(("h_sigbkg_data", variable, bins, xmin, xmax), variable)
        data_hist = hist_ptr.GetValue()
        ROOT.SetOwnership(data_hist, False)
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
    canvas, upper_pad, lower_pad = _create_plot_pads("c_sig_bkg", "Signal vs Backgrounds", show_ratio)
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

        # Data is a pure overlay — drawn last, on top of everything
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
            ratio = data_hist.Clone("h_data_mc_ratio")
            ratio.SetDirectory(0)
            ROOT.SetOwnership(ratio, False)
            ratio.Divide(total_mc)
            ratio_draw_opt = "PE1"
            ratio_y_title = "Data/MC"
        else:
            # No data: Signal/Bkg
            ratio = signal_hists[0].Clone("h_sig_bkg_ratio")
            ratio.SetDirectory(0)
            ROOT.SetOwnership(ratio, False)
            ratio.Divide(total_mc)
            ratio_draw_opt = "HIST"
            ratio_y_title = "Sig/Bkg"

        lower_pad.cd()
        _style_ratio_hist(ratio, variable)
        ratio.GetYaxis().SetTitle(ratio_y_title)
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
    show_ratio: bool = False
) -> str:
    """Overlay existing histograms from multiple ROOT files on one canvas."""
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
    """Draw a 2D histogram from a ROOT file and save as PDF."""
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
    """Build a 2D histogram from two TTree branches and save a COLZ plot."""

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
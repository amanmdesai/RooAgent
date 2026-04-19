from pathlib import Path
from typing import List, Optional

import ROOT
from langchain_core.tools import tool

from .utils import (
    _unique_canvas_name,
    _maybe_rebin_hist,
    _parse_file_inputs,
    get_vector_branches,
    rewrite_vector_cut,
)


def _safe_get_vector_branches(file_path: str, tree_name: str) -> List[str]:
    try:
        return get_vector_branches(file_path, tree_name)
    except Exception:
        return []


def _apply_cuts(df, file_path: str, tree_name: str, cuts: Optional[List[str]], vector_mode: str):
    if not cuts:
        return df

    vector_vars = _safe_get_vector_branches(file_path, tree_name)
    for cut in cuts:
        df = df.Filter(rewrite_vector_cut(cut, vector_vars, vector_mode))
    return df


def _resolve_weight_branch(df, weight_branch: str) -> str:
    if not weight_branch:
        return ""

    columns = [str(c) for c in df.GetColumnNames()]
    return weight_branch if weight_branch in columns else ""


def _load_hist(file_path: str, hist_name: str, rebin: int):
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return None, f"Error: Could not open file {file_path}"

    h = f.Get(hist_name)
    if not h:
        f.Close()
        return None, f"Error: Histogram {hist_name} not found in file {file_path}"

    h.SetDirectory(0)
    ROOT.SetOwnership(h, False)
    f.Close()

    h = _maybe_rebin_hist(h, rebin)
    return h, None


def _build_tree_hist(
    file_path: str,
    tree_name: str,
    variable: str,
    bins: int,
    xmin: float,
    xmax: float,
    hist_name: str,
    weight_branch: str = "",
    cuts: Optional[List[str]] = None,
    vector_mode: str = "all",
    rebin: int = 1,
):
    df = ROOT.RDataFrame(tree_name, file_path)
    df = _apply_cuts(df, file_path, tree_name, cuts, vector_mode)

    resolved_weight = _resolve_weight_branch(df, weight_branch)
    if resolved_weight:
        wname = resolved_weight
    else:
        wname = "__rooagent_unit_weight"
        df = df.Define(wname, "1.0")

    hist_ptr = df.Histo1D((hist_name, variable, bins, xmin, xmax), variable, wname)
    h = hist_ptr.GetValue()
    ROOT.SetOwnership(h, False)
    h.SetDirectory(0)
    h = _maybe_rebin_hist(h, rebin)
    return h


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


def _style_ratio_hist(ratio_hist, x_title: str, y_title: str = "Ratio"):
    ratio_hist.SetTitle("")
    ratio_hist.GetXaxis().SetTitle(x_title)
    ratio_hist.GetYaxis().SetTitle(y_title)
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


def _draw_ratio_panel(hist_list: List, lower_pad, x_title: str):
    if lower_pad is None:
        return

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


def _draw_overlay_plot(
    hist_list: List,
    legend,
    output_pdf: str,
    x_title: str,
    y_title: str,
    canvas_name: str,
    canvas_title: str,
    show_ratio: bool = False,
    logy: bool = False,
    draw_options: Optional[List[str]] = None,
):
    if not hist_list:
        return "No histograms created."

    canvas, upper_pad, lower_pad = _create_plot_pads(
        _unique_canvas_name(canvas_name), canvas_title, show_ratio
    )
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

        if draw_options and i < len(draw_options):
            draw_option = draw_options[i]
        else:
            draw_option = "HIST" if i == 0 else "HIST SAME"
        prepared_draws.append((hist, draw_option))

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

    _draw_ratio_panel(hist_list, lower_pad, x_title)

    canvas.Update()
    canvas.SaveAs(output_pdf)
    return output_pdf


def _plot_hist(
    file_path: str,
    hist_name: str,
    output_pdf: str,
    xlabel: str,
    ylabel: str,
    logy: bool,
    normalize: bool,
    rebin: int,
):
    h, err = _load_hist(file_path, hist_name, rebin)
    if err:
        return err

    if normalize and h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

    h.SetLineColor(ROOT.kBlue + 1)
    h.SetLineWidth(3)

    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    legend.AddEntry(h, hist_name, "l")

    result = _draw_overlay_plot(
        hist_list=[h],
        legend=legend,
        output_pdf=output_pdf,
        x_title=xlabel,
        y_title="Normalized Events" if normalize else ylabel,
        canvas_name="c_hist",
        canvas_title="Histogram",
        show_ratio=False,
        logy=logy,
    )
    if result == "No histograms created.":
        return "Error: No histograms were found."

    return f"Saved 1D histogram {hist_name} to {output_pdf}"


def _plot_tree(
    file_path: str,
    tree_name: str,
    variable: str,
    bins: int,
    xmin: float,
    xmax: float,
    output_pdf: str,
    normalize: bool,
    weight_branch: str,
    cuts: Optional[List[str]],
    rebin: int,
    vector_mode: str,
):
    h = _build_tree_hist(
        file_path=file_path,
        tree_name=tree_name,
        variable=variable,
        bins=bins,
        xmin=xmin,
        xmax=xmax,
        hist_name=variable,
        weight_branch=weight_branch,
        cuts=cuts,
        vector_mode=vector_mode,
        rebin=rebin,
    )

    if normalize and h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

    h.SetLineColor(ROOT.kBlue + 1)
    h.SetLineWidth(3)

    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    legend.AddEntry(h, variable, "l")

    result = _draw_overlay_plot(
        hist_list=[h],
        legend=legend,
        output_pdf=output_pdf,
        x_title=variable,
        y_title="Normalized Events" if normalize else "Events",
        canvas_name="c_tree",
        canvas_title="Tree Variable",
        show_ratio=False,
    )
    if result == "No histograms created.":
        return "Error: No histograms were created."

    return f"Saved plot to {output_pdf}"


def _plot_tree_compare(
    file_paths: List[str],
    tree_name: str,
    variables: List[str],
    bins: int,
    xmin: float,
    xmax: float,
    legends: List[str],
    output_pdf: str,
    normalize: bool,
    show_ratio: bool,
    rebin: int,
    weight_branch: str,
    cuts: Optional[List[str]],
    vector_mode: str,
):
    if not (len(file_paths) == len(variables) == len(legends)):
        return "Error: file_paths, variables, and legends must have the same length."

    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    colors = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]
    hist_list = []

    for i, (fpath, var, label) in enumerate(zip(file_paths, variables, legends)):
        h = _build_tree_hist(
            file_path=fpath,
            tree_name=tree_name,
            variable=var,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            hist_name=f"h{i}",
            weight_branch=weight_branch,
            cuts=cuts,
            vector_mode=vector_mode,
            rebin=rebin,
        )

        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        h.SetLineColor(colors[i % len(colors)])
        h.SetLineWidth(3)
        h.SetTitle("")
        hist_list.append(h)
        legend.AddEntry(h, label, "l")

    result = _draw_overlay_plot(
        hist_list=hist_list,
        legend=legend,
        output_pdf=output_pdf,
        x_title=variables[0],
        y_title="Normalized Events" if normalize else "Events",
        canvas_name="c_tree_compare",
        canvas_title="Tree Comparison",
        show_ratio=show_ratio,
    )
    if result == "No histograms created.":
        return result

    return f"Saved comparison histogram to {output_pdf}"


def _plot_hist_compare(
    file_paths: List[str],
    hist_names: List[str],
    legends: List[str],
    output_pdf: str,
    xlabel: str,
    ylabel: str,
    logy: bool,
    normalize: bool,
    show_ratio: bool,
    rebin: int,
):
    if not (len(file_paths) == len(hist_names) == len(legends)):
        return "Error: file_paths, hist_names, and legends must have the same length."

    legend = ROOT.TLegend(0.65, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)

    colors = [ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7, ROOT.kCyan + 2]
    hist_list = []
    skipped = []

    for i, (fpath, hname, label) in enumerate(zip(file_paths, hist_names, legends)):
        h, err = _load_hist(fpath, hname, rebin)
        if err:
            skipped.append(err)
            continue

        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        h.SetLineColor(colors[i % len(colors)])
        h.SetLineWidth(3)
        h.SetLineStyle(1 + i % 4)
        hist_list.append(h)
        legend.AddEntry(h, label, "l")

    if not hist_list:
        return "Error: No histograms were found."

    x_axis_title = xlabel if xlabel else hist_list[0].GetXaxis().GetTitle()
    result = _draw_overlay_plot(
        hist_list=hist_list,
        legend=legend,
        output_pdf=output_pdf,
        x_title=x_axis_title,
        y_title="Normalized Events" if normalize else ylabel,
        canvas_name="c_hist_compare",
        canvas_title="Histogram Comparison",
        show_ratio=show_ratio,
        logy=logy,
    )
    if result == "No histograms created.":
        return "Error: No histograms were found."

    msg = f"Saved combined histogram plot to {output_pdf}"
    if skipped:
        msg += "\nWarnings:\n" + "\n".join(f"  - {s}" for s in skipped)
    return msg


def _plot_signal_vs_backgrounds(
    signal_file: str,
    signal_files: Optional[List[str]],
    signal_label: str,
    signal_labels: Optional[List[str]],
    background_files: Optional[List[str]],
    background_labels: Optional[List[str]],
    data_file: str,
    data_label: str,
    plot_data: bool,
    tree_name: str,
    variable: str,
    bins: int,
    xmin: float,
    xmax: float,
    output_pdf: str,
    normalize: bool,
    show_ratio: bool,
    rebin: int,
    weight_branch: str,
    cuts: Optional[List[str]],
    vector_mode: str,
    apply_cuts_before_plot: bool,
):
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

    effective_cuts = cuts if apply_cuts_before_plot else None

    signal_hists = []
    background_hists = []

    signal_line_colors = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]
    signal_hatch_styles = [3345, 3354, 3395, 3444, 3490]
    background_fill_colors = [
        ROOT.kAzure - 9,
        ROOT.kOrange - 2,
        ROOT.kSpring - 6,
        ROOT.kPink - 8,
        ROOT.kViolet - 6,
        ROOT.kCyan - 6,
    ]

    for i, fpath in enumerate(background_paths):
        h = _build_tree_hist(
            file_path=fpath,
            tree_name=tree_name,
            variable=variable,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            hist_name=f"h_sigbkg_bkg_{i}",
            weight_branch=weight_branch,
            cuts=effective_cuts,
            vector_mode=vector_mode,
            rebin=rebin,
        )
        h.SetFillColor(background_fill_colors[i % len(background_fill_colors)])
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(1)
        background_hists.append(h)

    for i, fpath in enumerate(signal_paths):
        h = _build_tree_hist(
            file_path=fpath,
            tree_name=tree_name,
            variable=variable,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            hist_name=f"h_sigbkg_sig_{i}",
            weight_branch=weight_branch,
            cuts=effective_cuts,
            vector_mode=vector_mode,
            rebin=rebin,
        )
        line_color = signal_line_colors[i % len(signal_line_colors)]
        h.SetFillStyle(0)
        h.SetLineColor(line_color)
        h.SetLineWidth(3)
        h.SetLineStyle(1 + (i % 4))
        signal_hists.append(h)

    data_hist = None
    if wants_data:
        data_hist = _build_tree_hist(
            file_path=data_file,
            tree_name=tree_name,
            variable=variable,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            hist_name="h_sigbkg_data",
            weight_branch="",
            cuts=effective_cuts,
            vector_mode=vector_mode,
            rebin=rebin,
        )
        data_hist.SetFillStyle(0)
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerSize(1.0)
        data_hist.SetMarkerColor(ROOT.kBlack)
        data_hist.SetLineColor(ROOT.kBlack)
        data_hist.SetLineWidth(1)

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

    use_stack = wants_data and not normalize

    if use_stack:
        for i, h in enumerate(signal_hists):
            line_color = signal_line_colors[i % len(signal_line_colors)]
            h.SetFillStyle(signal_hatch_styles[i % len(signal_hatch_styles)])
            h.SetFillColor(line_color)
            h.SetLineColor(line_color)
            h.SetLineWidth(2)
            h.SetLineStyle(1)

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

    canvas, upper_pad, lower_pad = _create_plot_pads(
        _unique_canvas_name("c_sig_bkg"), "Signal vs Backgrounds", show_ratio
    )
    draw_pad = upper_pad if upper_pad else canvas
    draw_pad.cd()

    if not use_stack:
        hist_list = background_hists + signal_hists + ([data_hist] if data_hist else [])
        draw_options = ["HIST"] * (len(background_hists) + len(signal_hists))
        if data_hist:
            draw_options.append("PE1")

        result = _draw_overlay_plot(
            hist_list=hist_list,
            legend=legend,
            output_pdf=output_pdf,
            x_title=variable,
            y_title=y_title,
            canvas_name="c_sig_bkg_overlay",
            canvas_title="Signal vs Backgrounds",
            show_ratio=show_ratio,
            draw_options=draw_options,
        )
        if result == "No histograms created.":
            return result

        if show_ratio and lower_pad is not None:
            # Keep a physics-meaningful ratio panel for this mode.
            all_bkg = background_hists[0].Clone("h_all_bkg_ratio")
            all_bkg.SetDirectory(0)
            ROOT.SetOwnership(all_bkg, False)
            for h in background_hists[1:]:
                all_bkg.Add(h)

            lower_pad.cd()
            if data_hist:
                ratio = data_hist.Clone("h_data_bkg_ratio")
                ratio.SetDirectory(0)
                ROOT.SetOwnership(ratio, False)
                ratio.Divide(all_bkg)
                _style_ratio_hist(ratio, variable, "Data/Bkg")
                ratio.Draw("PE1")
            else:
                ratio = _build_signal_background_ratio(signal_hists[0], background_hists)
                _style_ratio_hist(ratio, variable, "Sig/Bkg")
                ratio.Draw("HIST")

            x_axis = ratio.GetXaxis()
            unity = ROOT.TLine(x_axis.GetXmin(), 1.0, x_axis.GetXmax(), 1.0)
            ROOT.SetOwnership(unity, False)
            unity.SetLineStyle(2)
            unity.SetLineColor(ROOT.kGray + 2)
            unity.Draw()

            canvas.Update()
            canvas.SaveAs(output_pdf)

        return f"Saved signal-vs-background comparison to {output_pdf}"

    stack = ROOT.THStack("hs_sig_bkg", "")
    ROOT.SetOwnership(stack, False)
    for h in background_hists:
        stack.Add(h)
    for h in signal_hists:
        stack.Add(h)

    stack.Draw("HIST")
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

    all_mc_hists = background_hists + signal_hists
    mc_stat_band = all_mc_hists[0].Clone("h_mc_stat_band")
    mc_stat_band.SetDirectory(0)
    ROOT.SetOwnership(mc_stat_band, False)
    for h in all_mc_hists[1:]:
        mc_stat_band.Add(h)
    mc_stat_band.SetFillStyle(3354)
    mc_stat_band.SetFillColor(ROOT.kGray + 2)
    mc_stat_band.SetLineColor(ROOT.kGray + 2)
    mc_stat_band.SetMarkerSize(0)
    mc_stat_band.Draw("E2 SAME")

    if data_hist:
        data_hist.Draw("PE1 SAME")

    legend.Draw()

    if show_ratio and lower_pad is not None:
        total_mc = all_mc_hists[0].Clone("h_total_mc_for_ratio")
        total_mc.SetDirectory(0)
        ROOT.SetOwnership(total_mc, False)
        for h in all_mc_hists[1:]:
            total_mc.Add(h)

        ratio = data_hist.Clone("h_data_mc_ratio")
        ratio.SetDirectory(0)
        ROOT.SetOwnership(ratio, False)
        ratio.Divide(data_hist, total_mc, 1.0, 1.0, "B")

        lower_pad.cd()
        _style_ratio_hist(ratio, variable, "Data/MC")
        ratio.Draw("PE1")

        x_axis = ratio.GetXaxis()
        unity = ROOT.TLine(x_axis.GetXmin(), 1.0, x_axis.GetXmax(), 1.0)
        ROOT.SetOwnership(unity, False)
        unity.SetLineStyle(2)
        unity.SetLineColor(ROOT.kGray + 2)
        unity.Draw()

    canvas.Update()
    canvas.SaveAs(output_pdf)
    return f"Saved signal-vs-background comparison to {output_pdf}"


def _plot_2d_hist(
    file_path: str,
    hist_name: str,
    output_pdf: str,
    xlabel: str,
    ylabel: str,
    color_palette: int,
    normalize: bool,
):
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return f"Error: Could not open file {file_path}"

    h = f.Get(hist_name)
    if not h:
        f.Close()
        return f"Error: Histogram {hist_name} not found in file {file_path}"

    h.SetDirectory(0)
    ROOT.SetOwnership(h, False)
    f.Close()

    if normalize and h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

    canv = ROOT.TCanvas(_unique_canvas_name("c2d"), "2D Histogram", 900, 700)

    h.GetXaxis().SetTitle(xlabel)
    h.GetYaxis().SetTitle(ylabel)

    ROOT.gStyle.SetPalette(color_palette)
    h.Draw("COLZ")

    canv.Update()
    canv.SaveAs(output_pdf)

    return f"Saved 2D histogram to {output_pdf}"


def _plot_2d_tree(
    file_path: str,
    tree_name: str,
    x_branch: str,
    y_branch: str,
    output_pdf: str,
    bins_x: int,
    xmin: float,
    xmax: float,
    bins_y: int,
    ymin: float,
    ymax: float,
    xlabel: str,
    ylabel: str,
    color_palette: int,
):
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return f"Error: Could not open file {file_path}"

    tree = f.Get(tree_name)
    if not tree:
        f.Close()
        return f"Error: TTree {tree_name} not found in {file_path}"

    h2 = ROOT.TH2F(
        _unique_canvas_name("h2"),
        "",
        bins_x,
        xmin,
        xmax,
        bins_y,
        ymin,
        ymax,
    )
    ROOT.SetOwnership(h2, False)

    tree.Draw(f"{y_branch}:{x_branch} >> {h2.GetName()}", "", "goff")

    canv = ROOT.TCanvas(_unique_canvas_name("c2d_tree"), "2D Histogram", 900, 700)
    h2.GetXaxis().SetTitle(xlabel if xlabel else x_branch)
    h2.GetYaxis().SetTitle(ylabel if ylabel else y_branch)

    ROOT.gStyle.SetPalette(color_palette)
    h2.Draw("COLZ")

    canv.Update()
    canv.SaveAs(output_pdf)
    f.Close()

    return f"Saved 2D histogram ({y_branch} vs {x_branch}) to {output_pdf}"


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
    signal_file: str = "",
    signal_files: Optional[List[str]] = None,
    signal_label: str = "Signal",
    signal_labels: Optional[List[str]] = None,
    background_files: Optional[List[str]] = None,
    background_labels: Optional[List[str]] = None,
    data_file: str = "",
    data_label: str = "Data",
    plot_data: bool = False,
    apply_cuts_before_plot: bool = False,
) -> str:
    """Unified 1D plotting tool.

    Modes
    -----
    - `hist`: draw one TH1 from one file.
    - `tree`: draw one branch from one tree/file.
    - `tree_compare`: compare multiple tree variables/files.
    - `hist_compare`: compare multiple existing TH1 histograms.
    - `signal_background`: compare signal and background processes.

    Returns
    -------
    str
        Saved PDF path message or a clear error string.
    """
    mode_key = (mode or "").strip().lower()

    if mode_key == "hist":
        if not file_path or not hist_name:
            return "Error: file_path and hist_name are required for mode='hist'."
        return _plot_hist(
            file_path=file_path,
            hist_name=hist_name,
            output_pdf=output_pdf,
            xlabel=xlabel,
            ylabel=ylabel,
            logy=logy,
            normalize=normalize,
            rebin=rebin,
        )

    if mode_key == "tree":
        if not file_path or not tree_name or not variable:
            return "Error: file_path, tree_name, and variable are required for mode='tree'."
        return _plot_tree(
            file_path=file_path,
            tree_name=tree_name,
            variable=variable,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            output_pdf=output_pdf,
            normalize=normalize,
            weight_branch=weight_branch,
            cuts=cuts,
            rebin=rebin,
            vector_mode=vector_mode,
        )

    if mode_key == "tree_compare":
        if not file_paths or not tree_name or not variables or not legends:
            return "Error: file_paths, tree_name, variables, and legends are required for mode='tree_compare'."
        return _plot_tree_compare(
            file_paths=file_paths,
            tree_name=tree_name,
            variables=variables,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            legends=legends,
            output_pdf=output_pdf,
            normalize=normalize,
            show_ratio=show_ratio,
            rebin=rebin,
            weight_branch=weight_branch,
            cuts=cuts,
            vector_mode=vector_mode,
        )

    if mode_key == "hist_compare":
        if not file_paths or not hist_names or not legends:
            return "Error: file_paths, hist_names, and legends are required for mode='hist_compare'."
        return _plot_hist_compare(
            file_paths=file_paths,
            hist_names=hist_names,
            legends=legends,
            output_pdf=output_pdf,
            xlabel=xlabel,
            ylabel=ylabel,
            logy=logy,
            normalize=normalize,
            show_ratio=show_ratio,
            rebin=rebin,
        )

    if mode_key in {"signal_background", "signal_vs_backgrounds"}:
        return _plot_signal_vs_backgrounds(
            signal_file=signal_file,
            signal_files=signal_files,
            signal_label=signal_label,
            signal_labels=signal_labels,
            background_files=background_files,
            background_labels=background_labels,
            data_file=data_file,
            data_label=data_label,
            plot_data=plot_data,
            tree_name=tree_name,
            variable=variable,
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            output_pdf=output_pdf,
            normalize=normalize,
            show_ratio=show_ratio,
            rebin=rebin,
            weight_branch=weight_branch,
            cuts=cuts,
            vector_mode=vector_mode,
            apply_cuts_before_plot=apply_cuts_before_plot,
        )

    return "Error: unsupported mode. Use one of: hist, tree, tree_compare, hist_compare, signal_background."


@tool
def plot_2d(
    mode: str,
    output_pdf: str,
    file_path: str = "",
    tree_name: str = "",
    x_branch: str = "",
    y_branch: str = "",
    hist_name: str = "",
    bins_x: int = 50,
    xmin: float = 0.0,
    xmax: float = 100.0,
    bins_y: int = 50,
    ymin: float = 0.0,
    ymax: float = 100.0,
    xlabel: str = "",
    ylabel: str = "",
    color_palette: int = 55,
    normalize: bool = False,
) -> str:
    """Unified 2D plotting tool.

    Modes
    -----
    - `hist`: draw one TH2 from a ROOT file.
    - `tree`: build a TH2 from two TTree branches and draw it.

    Returns
    -------
    str
        Saved PDF path message or a clear error string.
    """
    mode_key = (mode or "").strip().lower()

    if mode_key == "hist":
        if not file_path or not hist_name:
            return "Error: file_path and hist_name are required for mode='hist'."
        return _plot_2d_hist(
            file_path=file_path,
            hist_name=hist_name,
            output_pdf=output_pdf,
            xlabel=xlabel,
            ylabel=ylabel,
            color_palette=color_palette,
            normalize=normalize,
        )

    if mode_key == "tree":
        if not file_path or not tree_name or not x_branch or not y_branch:
            return "Error: file_path, tree_name, x_branch, and y_branch are required for mode='tree'."
        return _plot_2d_tree(
            file_path=file_path,
            tree_name=tree_name,
            x_branch=x_branch,
            y_branch=y_branch,
            output_pdf=output_pdf,
            bins_x=bins_x,
            xmin=xmin,
            xmax=xmax,
            bins_y=bins_y,
            ymin=ymin,
            ymax=ymax,
            xlabel=xlabel,
            ylabel=ylabel,
            color_palette=color_palette,
        )

    return "Error: unsupported mode. Use one of: hist, tree."

from typing import List, Optional
from pathlib import Path
import os
import re
import math
import ROOT


# Module-level counter for unique canvas names shared across tools
_canvas_counter = [0]
_roostats_counter = [0]


###############################
# tfile_tools.py helpers
###############################


# Open a ROOT TFile and return it, or None if the file cannot be opened.
def _open_root_file(file_path: str):
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        try:
            if f:
                f.Close()
        except Exception:
            pass
        return None
    return f


# Return a list of .root filenames in `directory` (non-recursive).
def _get_root_files(directory: str = ".") -> List[str]:
    try:
        return [f for f in os.listdir(directory) if f.lower().endswith(".root")]
    except Exception:
        return []


# Extract TTree names from an opened ROOT file object.
def _get_trees(root_file: ROOT.TFile) -> List[str]:
    trees: List[str] = []
    try:
        for key in root_file.GetListOfKeys():
            obj = key.ReadObj()
            if obj and hasattr(obj, "InheritsFrom") and obj.InheritsFrom("TTree"):
                trees.append(obj.GetName())
    except Exception:
        pass
    return trees


# Recursively list objects in a ROOT directory, returning "name (Class)" strings.
def _list_objects_recursive(root_dir: ROOT.TDirectory, prefix: str = "") -> List[str]:
    entries: List[str] = []
    try:
        for key in root_dir.GetListOfKeys():
            obj = key.ReadObj()
            if obj is None:
                continue
            name = obj.GetName()
            class_name = obj.ClassName() if hasattr(obj, "ClassName") else type(obj).__name__
            full = f"{prefix}{name} ({class_name})"
            entries.append(full)

            if obj.InheritsFrom("TDirectory") if hasattr(obj, "InheritsFrom") else False:
                try:
                    entries.extend(_list_objects_recursive(obj, prefix=f"{prefix}{name}/"))
                except Exception:
                    pass
    except Exception:
        pass
    return entries


###############################
# histogram_tools.py helpers
# (also used by plot_tools.py)
###############################


def _rebin_hist(hist, rebin: int):
    # Rebin a histogram by integer factor `rebin`, returning the rebinned object.
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


###############################
# rdataframe_tools.py helpers
# (also used by plot_tools.py)
###############################


# Parse single or multiple background file inputs into a unique ordered list.
def _parse_paths(primary: Optional[str] = None, additional: Optional[List[str]] = None) -> List[str]:
    parsed: List[str] = []

    if primary:
        parsed.extend([p.strip() for p in primary.split(",") if p.strip()])

    if additional:
        parsed.extend([p.strip() for p in additional if p and p.strip()])

    return list(dict.fromkeys(parsed))


# Note: Use _parse_paths directly for parsing file inputs (replaces removed
# _parse_background_inputs and _parse_file_inputs wrappers).


def _to_float_list(v):
    # Convert string/iterable/number input into a list of floats.
    # Accepts a comma-separated string, a list/tuple of numbers/strings, or a
    # single numeric value. Returns an empty list for None.
    if v is None:
        return []
    if isinstance(v, str):
        return [float(s.strip()) for s in v.split(",") if s.strip()]
    if isinstance(v, (list, tuple)):
        return [float("nan") if x is None else float(x) for x in v]
    return [float(v)]


# Detect vector-like branches in a TTree by examining branch classnames.
def _get_vector_branches(file_path: str, tree_name: str) -> List[str]:
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return []

    tree = f.Get(tree_name)
    if not tree:
        f.Close()
        return []

    vector_branches = []

    for branch in tree.GetListOfBranches():
        classname = branch.GetClassName()
        name = branch.GetName()

        if "vector" in classname:
            vector_branches.append(name)

    f.Close()
    return vector_branches


# Rewrite python-like logicals and vector comparisons to RDataFrame/VecOps form.
def _rewrite_vector_cut(cut: str, vector_vars: List[str], mode: str = "any") -> str:
    if mode not in ["any", "all"]:
        return cut

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


###############################
# Additional rdataframe_helpers moved from rdataframe_tools.py
###############################


# Build an RDataFrame from one or more input ROOT files.
def _build_dataframe(tree_name: str, files: List[str]):
    if len(files) == 1:
        return ROOT.RDataFrame(tree_name, files[0])
    return ROOT.RDataFrame(tree_name, files)


# Return True if the RDataFrame contains a column with the given name.
def _has_column(df, column_name: str) -> bool:
    if not column_name:
        return False

    cols = [str(c) for c in df.GetColumnNames()]
    return column_name in cols


# Return the weighted sum or count after applying `cut` to an RDataFrame.
def _filtered_yield(df, cut: str, weight: Optional[str] = None):
    filtered = df.Filter(cut)
    if weight and _has_column(filtered, weight):
        return filtered.Sum(weight).GetValue()
    return filtered.Count().GetValue()


# Return total yield from an RDataFrame, using `weight` if present.
def _total_yield(df, weight: Optional[str] = None):
    if weight and _has_column(df, weight):
        return df.Sum(weight).GetValue()
    return df.Count().GetValue()


###############################
# plot_tools.py helpers
# (also used by fit_tools.py and rdataframe_tools.py)
###############################


# Produce a short unique canvas name using an internal counter.
def _unique_canvas_name(base: str) -> str:
    _canvas_counter[0] += 1
    return f"{base}_{_canvas_counter[0]}"


# Apply a list of cuts to an RDataFrame, rewriting vector comparisons safely.
def _apply_cuts(df, file_path: str, tree_name: str, cuts: Optional[List[str]], vector_mode: str):
    if not cuts:
        return df

    try:
        vector_vars = _get_vector_branches(file_path, tree_name)
    except Exception:
        vector_vars = []

    for cut in cuts:
        df = df.Filter(_rewrite_vector_cut(cut, vector_vars, vector_mode))
    return df


# Return the weight branch name if it exists in the dataframe, else empty string.
def _resolve_weight_branch(df, weight_branch: str) -> str:
    if not weight_branch:
        return ""
    return weight_branch if _has_column(df, weight_branch) else ""


# Load a TH1 and optionally rebin it; returns (hist, None) or (None, error_message).
def _load_hist(file_path: str, hist_name: str, rebin: int):
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return None, f"Error: could not open file {file_path}."
    h = f.Get(hist_name)
    if not h:
        f.Close()
        return None, f"Error: histogram '{hist_name}' not found in {file_path}."
    h.SetDirectory(0)
    ROOT.SetOwnership(h, False)
    f.Close()
    h = _rebin_hist(h, rebin)
    return h, None


# Build a TH1 from a TTree using RDataFrame, applying optional cuts and weights.
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
    h = _rebin_hist(h, rebin)
    return h


# Build and return a TH1 for `variable` from a TTree (used by rdataframe_tools)
def _tree_variable_to_histogram(
    file_path: str,
    tree_name: str,
    variable: str,
    bins: int,
    xmin: float,
    xmax: float,
    cuts: Optional[List[str]] = None,
    vector_mode: str = "all",
    weight: Optional[str] = None,
):
    df = ROOT.RDataFrame(tree_name, file_path)

    if cuts:
        vector_vars = _get_vector_branches(file_path, tree_name)
        for cut in cuts:
            safe_cut = _rewrite_vector_cut(cut, vector_vars, vector_mode)
            df = df.Filter(safe_cut)

    if weight and _has_column(df, weight):
        wname = weight
    else:
        wname = "__rooagent_unit_weight"
        df = df.Define(wname, "1.0")

    stem = os.path.splitext(os.path.basename(file_path))[0]
    hist_name = f"h_{stem}_{variable}"
    hist_ptr = df.Histo1D((hist_name, hist_name, int(bins), float(xmin), float(xmax)), variable, wname)
    h = hist_ptr.GetValue()
    try:
        h.SetDirectory(0)
        ROOT.SetOwnership(h, False)
    except Exception:
        pass
    return h


# Create a canvas and optional upper/lower pads for plotting ratio panels.
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


# Apply styling to a ratio histogram for clear axis sizing and limits.
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


# Build ratio histograms of each histogram relative to the first one.
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


# Build a signal / summed-background ratio histogram from signal and backgrounds.
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


# Draw the lower ratio panel using reference ratio histograms.
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


# Draw multiple histograms on a canvas with legend and optional ratio panel.
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


# The following plotting helper wrappers were moved from plot_tools.py so
# that only @tool-decorated entry points remain in plot_tools.  These
# functions are plain helpers (not LangChain tools) and belong in utils.


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
        return "Error: No histograms were found."

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

    signal_paths = _parse_paths(signal_file, signal_files)
    if not signal_paths:
        return "Error: at least one signal file must be provided via signal_file or signal_files."

    background_paths = _parse_paths(additional=background_files)
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
    zlabel: str = "",
    logz: bool = False,
    normalize: bool = False,
    rebin_x: int = 1,
    rebin_y: int = 2,
    color_palette: int = 1,
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

    if rebin_x > 1:
        h.RebinX(rebin_x)
    if rebin_y > 1:
        h.RebinY(rebin_y)

    if normalize and h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

    canv = ROOT.TCanvas(_unique_canvas_name("c2d"), "2D Histogram", 900, 700)

    h.GetXaxis().SetTitle(xlabel)
    h.GetYaxis().SetTitle(ylabel)
    h.GetZaxis().SetTitle(zlabel)

    if logz:
        canv.SetLogz(1)

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


###############################
# stat_tools.py helpers
###############################


# Safely retrieve a histogram from an open ROOT file and detach it from the file.
def _get_hist(f, name: str):
    if not f:
        return None
    h = f.Get(name)
    if not h:
        return None
    try:
        h.SetDirectory(0)
        ROOT.SetOwnership(h, False)
    except Exception:
        pass
    return h


# Minimal counting: use the provided single `file_path` and explicit
# histogram names. Upstream stages are expected to validate inputs.
def _counting_window_inputs(
    file_path: str,
    bkg_name: str,
    sig_name: str,
    data_name: str,
    center: float,
    window: float,
):
    if not file_path:
        return None, "Error: file_path is required."

    p = str(file_path).strip()
    if not p:
        return None, "Error: file_path is required."

    f = _open_root_file(p)
    if not f:
        return None, f"Error: could not open file {p}."

    hbkg = _get_hist(f, bkg_name)
    hsig = _get_hist(f, sig_name)
    hdata = _get_hist(f, data_name) if data_name and data_name.strip() else None

    f.Close()

    if hbkg is None or hsig is None:
        return None, "Error: required histograms not found in file."

    n_bkg = _fractional_integral(hbkg, center, window)
    n_sig = _fractional_integral(hsig, center, window)
    n_obs = None
    if hdata is not None:
        n_obs = max(0, int(round(_fractional_integral(hdata, center, window))))

    return {"n_bkg": n_bkg, "n_sig": n_sig, "n_obs": n_obs}, None


# RooStats NumberCountingUtils is unstable at exactly zero uncertainty.
_ROOSTATS_FRACTIONAL_B_UNCERTAINTY = 1e-4


# Create unique names for RooFit/RooStats objects.
def _next_roostats_tag() -> str:
    _roostats_counter[0] += 1
    return f"{_roostats_counter[0]}"


# Keep probabilities in the physical range.
def _roostats_clip_probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


# Map expected background to a stable NumberCountingUtils uncertainty input.
def _roostats_fractional_b_uncertainty(mean: float) -> float:
    if mean <= 0.0:
        return 0.0
    return _ROOSTATS_FRACTIONAL_B_UNCERTAINTY


# Build a one-channel counting model for RooStats asymptotic tools.
def _build_roostats_counting_model(n_obs: int, n_bkg: float, n_sig: float):
    n_obs_val = max(0, int(n_obs))
    n_bkg_val = max(0.0, float(n_bkg))
    n_sig_val = max(0.0, float(n_sig))

    tag = _next_roostats_tag()
    ws = ROOT.RooWorkspace(f"ws_stat_{tag}")

    max_obs = max(
        float(n_obs_val) + 10.0 * math.sqrt(max(float(n_obs_val), 1.0)) + 20.0,
        n_bkg_val + n_sig_val + 20.0,
        50.0,
    )

    ws.factory(f"nobs_{tag}[{float(n_obs_val):.12g},0,{max_obs:.12g}]")
    ws.factory(f"bexp_{tag}[{n_bkg_val:.12g}]")
    ws.factory(f"sexp_{tag}[{n_sig_val:.12g}]")
    ws.factory(f"mu_{tag}[1,0,10000]")
    ws.factory(f"prod::sigterm_{tag}(mu_{tag},sexp_{tag})")
    ws.factory(f"sum::mean_{tag}(sigterm_{tag},bexp_{tag})")
    ws.factory(f"Poisson::model_{tag}(nobs_{tag},mean_{tag})")

    nobs_var = ws.var(f"nobs_{tag}")
    bexp_var = ws.var(f"bexp_{tag}")
    sexp_var = ws.var(f"sexp_{tag}")
    mu_var = ws.var(f"mu_{tag}")
    model_pdf = ws.pdf(f"model_{tag}")

    bexp_var.setConstant(True)
    sexp_var.setConstant(True)

    observables = ROOT.RooArgSet(nobs_var)
    poi = ROOT.RooArgSet(mu_var)

    data = ROOT.RooDataSet(f"data_{tag}", f"data_{tag}", observables)
    nobs_var.setVal(float(n_obs_val))
    data.add(observables)

    sb_model = ROOT.RooStats.ModelConfig(f"sb_{tag}", ws)
    sb_model.SetPdf(model_pdf)
    sb_model.SetObservables(observables)
    sb_model.SetParametersOfInterest(poi)
    mu_var.setVal(1.0)
    sb_snapshot = ROOT.RooArgSet(mu_var)
    sb_model.SetSnapshot(sb_snapshot)

    b_model = ROOT.RooStats.ModelConfig(f"b_{tag}", ws)
    b_model.SetPdf(model_pdf)
    b_model.SetObservables(observables)
    b_model.SetParametersOfInterest(poi)
    mu_var.setVal(0.0)
    b_snapshot = ROOT.RooArgSet(mu_var)
    b_model.SetSnapshot(b_snapshot)

    mu_var.setVal(1.0)
    return {
        "workspace": ws,
        "data": data,
        "mu_var": mu_var,
        "sb_model": sb_model,
        "b_model": b_model,
    }


# Get a RooStats HypoTestResult with explicit null/alternate ordering.
def _roostats_hypotest_result(n_obs: int, n_bkg: float, n_sig: float, null_is_sb: bool):
    try:
        model = _build_roostats_counting_model(n_obs, n_bkg, n_sig)
        null_model = model["sb_model"] if null_is_sb else model["b_model"]
        alt_model = model["b_model"] if null_is_sb else model["sb_model"]

        calculator = ROOT.RooStats.AsymptoticCalculator(model["data"], alt_model, null_model)
        calculator.SetOneSided(True)
        calculator.SetQTilde(True)
        return calculator.GetHypoTest()
    except Exception:
        return None


# Compute CLs, CLs+b, CLb using RooStats exclusion convention (null = s+b).
def _roostats_exclusion_summary(n_obs: int, n_bkg: float, n_sig: float):
    result = _roostats_hypotest_result(n_obs, n_bkg, n_sig, null_is_sb=True)
    if result is None:
        return float("nan"), float("nan"), float("nan")

    clspb = _roostats_clip_probability(float(result.CLsplusb()))
    clb = _roostats_clip_probability(float(result.CLb()))
    cls = float(result.CLs())
    if not math.isfinite(cls):
        cls = clspb / clb if clb > 0.0 else float("inf")
    return cls, clspb, clb


# Compute mu upper limit at confidence level cl using RooStats HypoTestInverter.
def _roostats_upper_limit(
    n_obs: int,
    n_bkg: float,
    n_sig_nominal: float,
    cl: float = 0.95,
) -> float:
    if n_sig_nominal <= 0.0:
        return float("inf")

    try:
        model = _build_roostats_counting_model(n_obs, n_bkg, n_sig_nominal)
        mu_var = model["mu_var"]
        calculator = ROOT.RooStats.AsymptoticCalculator(
            model["data"], model["b_model"], model["sb_model"]
        )
        calculator.SetOneSided(True)
        calculator.SetQTilde(True)

        mu_scan_max = max(
            5.0,
            (
                max(float(n_obs), float(n_bkg) + 5.0 * math.sqrt(max(float(n_bkg), 1.0)))
                - float(n_bkg)
            )
            / max(float(n_sig_nominal), 1e-9)
            + 5.0,
        )

        for _ in range(8):
            mu_var.setRange(0.0, max(10.0, mu_scan_max * 2.0))

            inverter = ROOT.RooStats.HypoTestInverter(calculator, mu_var, 1.0 - float(cl))
            inverter.SetConfidenceLevel(float(cl))
            inverter.UseCLs(True)
            inverter.SetVerbose(0)
            inverter.RunFixedScan(120, 0.0, float(mu_scan_max))

            interval = inverter.GetInterval()
            if interval is not None:
                mu_up = float(interval.UpperLimit())
                if math.isfinite(mu_up) and mu_up < 0.98 * mu_scan_max:
                    return mu_up

            mu_scan_max *= 2.0
            if mu_scan_max > 1e6:
                break
    except Exception:
        return float("inf")

    return float("inf")


# Compute the fractional integral of `hist` over [center-window, center+window], handling partial bins.
def _fractional_integral(hist, center: float, window: float) -> float:
    xlow = center - window
    xhigh = center + window
    ax = hist.GetXaxis()
    nb = hist.GetNbinsX()
    total = 0.0
    for ibin in range(1, nb + 1):
        low = ax.GetBinLowEdge(ibin)
        high = ax.GetBinUpEdge(ibin)
        overlap = max(0.0, min(high, xhigh) - max(low, xlow))
        if overlap <= 0.0:
            continue
        width = high - low
        frac = overlap / width if width > 0.0 else 0.0
        total += hist.GetBinContent(ibin) * frac
    return total


# Convert an upper-tail p-value to a one-sided discovery Z (Z >= 0).
def _significance_from_pvalue(pvalue: float) -> float:
    pvalue_val = _roostats_clip_probability(float(pvalue))
    if not (0.0 < pvalue_val < 1.0):
        return 0.0
    try:
        z_value = float(ROOT.RooStats.PValueToSignificance(pvalue_val))
        if not math.isfinite(z_value):
            return 0.0
        return max(0.0, z_value)
    except Exception:
        return 0.0


# Note: profile-likelihood extraction was removed; convert p-values to
# discovery significance using the canonical `_significance_from_pvalue`.


# Asymptotic (Cowan) significance for expected S+B vs B yields.
# Valid when B >> ~10 and no background uncertainty is modelled.
# Use for expected sensitivity estimates and discovery reach projections.
def _asymptotic_significance(n_sig: float, n_bkg: float) -> float:
    try:
        s = float(n_sig)
        b = float(n_bkg)
        if b <= 0.0:
            return float("nan")
        return max(0.0, float(ROOT.RooStats.AsimovSignificance(s, b, 0.0)))
    except Exception:
        return float("nan")


# RooStats number-counting expected significance from S and B.
def _number_counting_expected_significance(n_sig: float, n_bkg: float) -> float:
    try:
        s = max(0.0, float(n_sig))
        b = max(0.0, float(n_bkg))
        if b <= 0.0:
            return float("nan")
        rel_unc = _roostats_fractional_b_uncertainty(b)
        z_value = float(ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s, b, rel_unc))
        if not math.isfinite(z_value):
            return float("nan")
        return max(0.0, z_value)
    except Exception:
        return float("nan")


#Compute discovery significance from expected signal S and background B yields.
def _compute_significance_from_yields(S: float, B: float, method: str = "simple") -> float:

    s = float(S)
    b = float(B)
    if method == "asymptotic":
        return _asymptotic_significance(s, b)
    # default: RooStats number-counting expected significance
    if b <= 0.0:
        return float("nan")
    return _number_counting_expected_significance(s, b)


# Return compact p0 / Z / CLs summary string: always expected (S+B Asimov), observed if n_obs given.
def _stat_summary(n_bkg: float, n_sig: float, n_obs: Optional[int] = None) -> str:
    n_bkg_val = max(0.0, float(n_bkg))
    n_sig_val = max(0.0, float(n_sig))
    rel_bkg_unc = _roostats_fractional_b_uncertainty(n_bkg_val)

    def _line(n: int) -> str:
        n_val = max(0, int(n))
        p0 = _roostats_clip_probability(
            float(
                ROOT.RooStats.NumberCountingUtils.BinomialObsP(
                    float(n_val), n_bkg_val, rel_bkg_unc
                )
            )
        )
        # Convert the p-value to a one-sided discovery Z using the
        # canonical `_significance_from_pvalue` converter.
        z = _significance_from_pvalue(p0)
        cls, clspb, clb = _roostats_exclusion_summary(n_val, n_bkg_val, n_sig_val)
        return (
            f"N={n_val}  p0={p0:.4g}  Z={z:.4g}sigma  "
            f"CLs={cls:.4g}  CLs+b={clspb:.4g}  CLb={clb:.4g}"
        )

    n_exp = max(0, int(round(n_bkg_val + n_sig_val)))
    result = f"Expected(S+B Asimov): {_line(n_exp)}"
    if n_obs is not None:
        result += f" | Observed: {_line(n_obs)}"
    return result


from typing import Dict, List, Optional
from pathlib import Path
import os
import re
import math
import ROOT
from scipy.stats import norm, poisson



_canvas_counter = [0]


# ============================================================================
# FILE I/O & DISCOVERY: ROOT file and tree introspection
# ============================================================================

def _open_root_file(file_path: Optional[str] = None):
    # Open the first resolved ROOT file and return the TFile handle, or None on failure.
    paths = _parse_paths(file_path)
    if not paths:
        return None

    target = paths[0]
    f = ROOT.TFile.Open(target)
    if not f or f.IsZombie():
        try:
            if f:
                f.Close()
        except Exception:
            pass
        return None
    return f


def _get_root_files(directory: str = ".") -> List[str]:
    # List all .root files in the given directory.
    try:
        return [f for f in os.listdir(directory) if f.lower().endswith(".root")]
    except Exception:
        return []


def _get_trees(root_file: ROOT.TFile) -> List[str]:
    # Return the names of all TTrees in an open ROOT file.
    trees: List[str] = []
    try:
        for key in root_file.GetListOfKeys():
            obj = key.ReadObj()
            if obj and hasattr(obj, "InheritsFrom") and obj.InheritsFrom("TTree"):
                trees.append(obj.GetName())
    except Exception:
        pass
    return trees


def _list_objects_recursive(root_dir: ROOT.TDirectory, prefix: str = "") -> List[str]:
    # Recursively enumerate all named objects in a ROOT directory hierarchy.
    entries: List[str] = []
    if root_dir is None:
        return entries

    stack = [(root_dir, prefix)]
    while stack:
        cur_dir, cur_prefix = stack.pop()
        try:
            for key in cur_dir.GetListOfKeys():
                try:
                    obj = key.ReadObj()
                except Exception:
                    continue
                if obj is None:
                    continue
                name = obj.GetName()
                class_name = obj.ClassName() if hasattr(obj, "ClassName") else type(obj).__name__
                entries.append(f"{cur_prefix}{name} ({class_name})")

                try:
                    is_dir = obj.InheritsFrom("TDirectory") if hasattr(obj, "InheritsFrom") else False
                except Exception:
                    is_dir = False
                if is_dir:
                    stack.append((obj, f"{cur_prefix}{name}/"))
        except Exception:
            # ignore unreadable directories and continue
            continue
    return entries


# ============================================================================
# HISTOGRAM UTILITIES: Binning, path parsing, type conversion
# ============================================================================

def _rebin_hist(hist, rebin: int):
    # Rebin a TH1 by the given factor; returns the original histogram when rebin <= 1.
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


def _parse_paths(primary: Optional[str] = None, additional: Optional[List[str]] = None) -> List[str]:
    # Merge comma-separated string and list file/directory paths into a deduplicated absolute-path list.
    parsed: List[str] = []

    def _add_entry(p: str):
        p = p.strip()
        if not p:
            return
        if os.path.isdir(p):
            for f in _get_root_files(p):
                parsed.append(os.path.abspath(os.path.join(p, f)))
        else:
            # treat as a file path (may or may not exist)
            parsed.append(os.path.abspath(p))

    # Primary may be a comma-separated list
    if primary:
        parts = [s.strip() for s in str(primary).split(",") if s.strip()]
        for part in parts:
            _add_entry(part)

    # Additional may be a list of paths/directories
    if additional:
        for part in additional:
            if not part:
                continue
            _add_entry(part)

    # If nothing resolved, default to current working directory
    if not parsed:
        cwd = os.getcwd()
        for f in _get_root_files(cwd):
            parsed.append(os.path.abspath(os.path.join(cwd, f)))

    # Deduplicate while preserving order
    seen = set()
    result: List[str] = []
    for p in parsed:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _to_float_list(v):
    # Parse a scalar, comma-string, or list into a list of floats.
    if isinstance(v, str):
        return [float(s.strip()) for s in v.split(",") if s.strip()]
    if isinstance(v, (list, tuple)):
        return [float("nan") if x is None else float(x) for x in v]
    return [float(v)]


def _to_int_list(v):
    # Parse a scalar, comma-string, or list into a list of ints; returns [] on any invalid input.
    if isinstance(v, str):
        items = [s.strip() for s in v.split(",") if s.strip()]
    elif isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = [v]

    parsed: List[int] = []
    for item in items:
        if item is None:
            return []
        value = float(item)
        if not math.isfinite(value):
            return []
        parsed.append(int(round(value)))
    return parsed


def _parse_numeric_array(values, integer: bool = False):
    # Parse input into a list of finite floats or ints, raising ValueError on non-finite values.
    parsed = _to_int_list(values) if integer else _to_float_list(values)
    if any(not math.isfinite(value) for value in parsed):
        raise ValueError("values must be finite numeric values")
    return parsed


def _normalize_parallel_arrays(reference_name: str, reference_values, series: Dict[str, object], integer_series=None):
    # Validate and parse all series arrays against a reference, ensuring equal lengths.
    integer_series = set(integer_series or [])
    reference = _parse_numeric_array(reference_values)
    if not reference:
        raise ValueError(f"{reference_name} must contain at least one value")

    normalized = {}
    for name, values in series.items():
        parsed = _parse_numeric_array(values, integer=name in integer_series)
        if len(parsed) != len(reference):
            raise ValueError(
                f"{name} has length {len(parsed)}, expected {len(reference)} to match {reference_name}"
            )
        normalized[name] = parsed

    return reference, normalized


def _default_scan_sort(series_names: List[str]) -> str:
    # Choose the preferred sort key from available scan series names (favours z_obs, then z_exp, then cls).
    preferred = [
        "z_obs",
        "observed_significance",
        "observed_pvalue",
        "z_exp",
        "expected_significance",
        "expected_pvalue",
        "cls",
    ]
    lowered = {name.lower(): name for name in series_names}
    for candidate in preferred:
        if candidate in lowered:
            return lowered[candidate]
    return series_names[0]


def _scan_sort_descending(series_name: str) -> bool:
    # Return True if larger values are better for this series (e.g. Z), False if smaller is better (e.g. CLs, p0).
    lowered = series_name.lower().replace("_", " ")
    lower_better_tokens = ("cls", "p0",  "p value",  "yield up")
    return not any(token in lowered for token in lower_better_tokens)


def _format_scan_summary(
    parameter_values,
    series: Dict[str, List[float]],
    parameter_name: str = "parameter",
    parameter_unit: str = "",
    sort_by: str = "",
    top_n: int = 5,
    descending: Optional[bool] = None,
) -> str:
    # Format a ranked summary table for a completed parameter scan.
    if not series:
        raise ValueError("series must contain at least one named array")

    sort_field = sort_by or _default_scan_sort(list(series.keys()))
    if sort_field not in series:
        raise ValueError(f"sort_by='{sort_field}' is not present in series")

    sort_descending = _scan_sort_descending(sort_field) if descending is None else bool(descending)
    unit_suffix = f" {parameter_unit}" if parameter_unit else ""

    records = []
    for index, parameter_value in enumerate(parameter_values):
        record = {parameter_name: parameter_value}
        for series_name, series_values in series.items():
            record[series_name] = series_values[index]
        records.append(record)

    records.sort(key=lambda record: (record[sort_field], record[parameter_name]), reverse=sort_descending)

    def _format_record(record) -> str:
        parts = [f"{parameter_name} = {record[parameter_name]:.4g}{unit_suffix}"]
        for series_name, series_values in series.items():
            if series_name in record:
                parts.append(f"{series_name} = {record[series_name]:.4g}")
        return ", ".join(parts)

    lines = [f"Ranking (by {sort_field}, {'descending' if sort_descending else 'ascending'}):"]
    lines.append(f"- Best point: {_format_record(records[0])}")
    lines.append("- Top candidates:")
    for record in records[:max(1, int(top_n))]:
        lines.append(f"  - {_format_record(record)}")
    return "\n".join(lines)


# ============================================================================
# VECTOR BRANCH HANDLING: Detection and RDataFrame VecOps rewriting
# ============================================================================

def _get_vector_branches(file_path: str, tree_name: str) -> List[str]:
    # Return branch names whose type contains 'vector' (used to trigger VecOps rewriting).
    f = _open_root_file(file_path)
    if not f:
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


def _rewrite_vector_cut(cut: str, vector_vars: List[str], mode: str = "any") -> str:
    # Wrap vector-branch comparisons with ROOT::VecOps::Any or All to make them valid RDataFrame filters.
    cut = re.sub(r"\bTrue\b", "true", cut)
    cut = re.sub(r"\bFalse\b", "false", cut)
    cut = re.sub(r"\band\b", "&&", cut)
    cut = re.sub(r"\bor\b", "||", cut)

    mode_lower = (mode or "any").strip().lower()
    if mode_lower not in ["any", "all"]:
        return cut

    reducer = "Any" if mode_lower == "any" else "All"

    for var in vector_vars:
        pattern = rf"({var}\s*[<>!=]=?\s*[-+]?\d*\.?\d+)"
        matches = re.findall(pattern, cut)

        for match in matches:
            wrapped = f"ROOT::VecOps::{reducer}({match})"
            cut = cut.replace(match, wrapped)

    return cut


# ============================================================================
# RDATAFRAME OPERATIONS: Lazy evaluation, event selection, yield extraction
# ============================================================================

def _build_dataframe(tree_name: str, files: List[str]):
    # Create an RDataFrame from one or more ROOT files.
    if len(files) == 1:
        return ROOT.RDataFrame(tree_name, files[0])
    return ROOT.RDataFrame(tree_name, files)


def _has_column(df, column_name: str) -> bool:
    # Check whether a column exists in an RDataFrame.
    if not column_name:
        return False

    cols = [str(c) for c in df.GetColumnNames()]
    return column_name in cols


def _filtered_yield(df, cut: str, weight: Optional[str] = None):
    # Return the (weighted) event yield passing a selection cut.
    filtered = df.Filter(cut)
    if weight and _has_column(filtered, weight):
        return filtered.Sum(weight).GetValue()
    return filtered.Count().GetValue()


def _total_yield(df, weight: Optional[str] = None):
    # Return the total (weighted) event count of a dataframe.
    if weight and _has_column(df, weight):
        return df.Sum(weight).GetValue()
    return df.Count().GetValue()


def _unique_canvas_name(base: str) -> str:
    # Generate a unique ROOT canvas/histogram name by appending a global counter.
    _canvas_counter[0] += 1
    return f"{base}_{_canvas_counter[0]}"


def _apply_cuts(df, file_path: str, tree_name: str, cuts: Optional[List[str]], vector_mode: str):
    # Apply a list of physics selection cuts to an RDataFrame, rewriting vector expressions as needed.
    if not cuts:
        return df

    try:
        vector_vars = _get_vector_branches(file_path, tree_name)
    except Exception:
        vector_vars = []

    for cut in cuts:
        df = df.Filter(_rewrite_vector_cut(cut, vector_vars, vector_mode))
    return df


def _resolve_weight_branch(df, weight_branch: str) -> str:
    # Return the weight branch name if it exists in the dataframe, otherwise return an empty string.
    if not weight_branch:
        return ""
    return weight_branch if _has_column(df, weight_branch) else ""


# ============================================================================
# HISTOGRAM I/O & BUILDING: Load, create, and manipulate histograms
# ============================================================================

def _load_hist(file_path: str, hist_name: str, rebin: int):
    # Load a named TH1 from a ROOT file and return (histogram, None) or (None, error_string).
    f = _open_root_file(file_path)
    if not f:
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
    vector_mode: str = "any",
    rebin: int = 1,
    apply_cuts_before_plot: bool = True,
):
    # Project a TTree branch into a TH1 via RDataFrame, applying optional cuts and weights.
    files = _parse_paths(file_path)
    if not files:
        raise RuntimeError("No ROOT files found to build histogram.")

    df = _build_dataframe(tree_name, files)
    if apply_cuts_before_plot:
        # use the first file as a representative for vector-branch discovery
        df = _apply_cuts(df, files[0], tree_name, cuts, vector_mode)

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




# ============================================================================
# PLOTTING INFRASTRUCTURE: Canvas setup, styling, overlay mechanics
# ============================================================================

def _create_plot_pads(canvas_name: str, canvas_title: str, show_ratio: bool):
    # Create a ROOT canvas with an optional split upper/lower pad layout for ratio panels.
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
    # Apply axis styling appropriate for a ratio panel (enlarged labels, fixed y-range [0, 2]).
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
    # Divide every histogram in the list by the first, returning a list of ratio histograms.
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
    # Divide the signal histogram by the sum of all background histograms.
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
    # Draw pairwise ratio histograms (each divided by the first) in the lower pad.
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
    # Draw multiple histograms on a shared canvas, optionally with a ratio panel, and save to file.
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


# ============================================================================
# 1D PLOTTING: Histograms, trees, comparisons, signal vs background
# ============================================================================

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
    # Plot a stored TH1 and save to PDF.
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
    apply_cuts_before_plot: bool = True,
):
    # Project a TTree branch to a histogram and save the plot to PDF.
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
        apply_cuts_before_plot=apply_cuts_before_plot,
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
    apply_cuts_before_plot: bool = True,
):
    # Overlay the same variable from multiple files on a single canvas with an optional ratio panel.
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
            apply_cuts_before_plot=apply_cuts_before_plot,
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
    # Overlay stored TH1 histograms from multiple files and save the comparison plot.
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
    apply_cuts_before_plot: bool = True,
    stack_signal: bool = True,
):
    # Produce a stacked-backgrounds plot with optional signal overlay and data points.
    # stack_signal=True (default): signal is added to the stack on top of backgrounds.
    # stack_signal=False: backgrounds are stacked; signals are overlaid as separate lines.
    signal_paths = _parse_paths(signal_file, signal_files)

    background_paths = _parse_paths(additional=background_files)
    if not background_paths:
        return "Error: background_files cannot be empty."

    if signal_labels is None:
        if not signal_paths:
            signal_labels = []
        elif len(signal_paths) == 1:
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
        ROOT.kGray + 2,
        ROOT.kYellow - 7,
        ROOT.TColor.GetColor("#cfc9b3"),
        ROOT.TColor.GetColor("#bcd9d3"),
        ROOT.TColor.GetColor("#e8dff8"),
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
            cuts=cuts,
            vector_mode=vector_mode,
            rebin=rebin,
            apply_cuts_before_plot=apply_cuts_before_plot,
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
            cuts=cuts,
            vector_mode=vector_mode,
            rebin=rebin,
            apply_cuts_before_plot=apply_cuts_before_plot,
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
            cuts=cuts,
            vector_mode=vector_mode,
            rebin=rebin,
            apply_cuts_before_plot=apply_cuts_before_plot,
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

    # Drawing mode: full-stack (bkg+sig stacked), bkg-stack (bkg stacked, sig as lines), or overlay.
    use_full_stack = wants_data and not normalize and stack_signal and bool(signal_hists)
    use_bkg_stack = wants_data and not normalize and (not stack_signal) and bool(background_hists)

    if use_full_stack:
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

    sig_legend_opt = "f" if use_full_stack else "l"
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

    if not use_full_stack and not use_bkg_stack:
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

    if use_bkg_stack:
        # Backgrounds only in the stack; signals drawn as separate lines on top.
        bkg_stack = ROOT.THStack("hs_bkg_only", "")
        ROOT.SetOwnership(bkg_stack, False)
        for h in background_hists:
            bkg_stack.Add(h)

        draw_pad.cd()
        bkg_stack.Draw("HIST")
        bkg_stack.GetXaxis().SetTitle(variable)
        bkg_stack.GetYaxis().SetTitle(y_title)
        if show_ratio:
            bkg_stack.GetXaxis().SetLabelSize(0)
            bkg_stack.GetXaxis().SetTitleSize(0)
            bkg_stack.GetYaxis().SetTitleSize(0.055)
            bkg_stack.GetYaxis().SetLabelSize(0.045)

        sig_max = max((h.GetMaximum() for h in signal_hists), default=0.0)
        data_max = data_hist.GetMaximum() if data_hist else 0.0
        bkg_stack.SetMaximum(max(bkg_stack.GetMaximum(), sig_max, data_max) * 1.35)

        # MC stat band (backgrounds only)
        bkg_band = background_hists[0].Clone("h_bkg_stat_band")
        bkg_band.SetDirectory(0)
        ROOT.SetOwnership(bkg_band, False)
        for h in background_hists[1:]:
            bkg_band.Add(h)
        bkg_band.SetFillStyle(3354)
        bkg_band.SetFillColor(ROOT.kGray + 2)
        bkg_band.SetLineColor(ROOT.kGray + 2)
        bkg_band.SetMarkerSize(0)
        bkg_band.Draw("E2 SAME")

        for h in signal_hists:
            h.Draw("HIST SAME")

        if data_hist:
            data_hist.Draw("PE1 SAME")

        legend.Draw()

        if show_ratio and lower_pad is not None:
            total_bkg = background_hists[0].Clone("h_total_bkg_for_ratio")
            total_bkg.SetDirectory(0)
            ROOT.SetOwnership(total_bkg, False)
            for h in background_hists[1:]:
                total_bkg.Add(h)

            lower_pad.cd()
            if data_hist:
                ratio = data_hist.Clone("h_data_bkg_ratio")
                ratio.SetDirectory(0)
                ROOT.SetOwnership(ratio, False)
                ratio.Divide(data_hist, total_bkg, 1.0, 1.0, "B")
                _style_ratio_hist(ratio, variable, "Data/Bkg")
                ratio.Draw("PE1")
            elif signal_hists:
                ratio = _build_signal_background_ratio(signal_hists[0], background_hists)
                _style_ratio_hist(ratio, variable, "Sig/Bkg")
                ratio.Draw("HIST")

            if data_hist or signal_hists:
                x_axis = ratio.GetXaxis()
                unity = ROOT.TLine(x_axis.GetXmin(), 1.0, x_axis.GetXmax(), 1.0)
                ROOT.SetOwnership(unity, False)
                unity.SetLineStyle(2)
                unity.SetLineColor(ROOT.kGray + 2)
                unity.Draw()

        canvas.Update()
        canvas.SaveAs(output_pdf)
        return f"Saved signal-vs-background comparison to {output_pdf}"

    # use_full_stack: backgrounds + signals both in the stack, data overlay
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


# ============================================================================
# 2D PLOTTING: 2D histograms and kinematic correlations
# ============================================================================

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
    # Plot a stored TH2 with a colour-z palette and save to PDF.
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
    normalize: bool = False,
    rebin_x: int = 1,
    rebin_y: int = 1,
    weight_branch: str = "",
    cuts: Optional[List[str]] = None,
    vector_mode: str = "any",
    apply_cuts_before_plot: bool = True,
):
    # Build a 2D histogram from two TTree branches via RDataFrame and save the scatter plot.
    try:
        df = ROOT.RDataFrame(tree_name, file_path)
    except Exception:
        # Fallback to TTree::Draw if RDataFrame not available
        df = None

    h2 = None
    if df is not None:
        # Apply cuts via RDataFrame filters if requested
        if apply_cuts_before_plot:
            df = _apply_cuts(df, file_path, tree_name, cuts, vector_mode)

        resolved_weight = _resolve_weight_branch(df, weight_branch)
        if resolved_weight:
            wname = resolved_weight
        else:
            wname = "__rooagent_unit_weight_2d"
            df = df.Define(wname, "1.0")

        try:
            hist_ptr = df.Histo2D(( _unique_canvas_name("h2"), "", bins_x, xmin, xmax, bins_y, ymin, ymax), x_branch, y_branch, wname)
            h2 = hist_ptr.GetValue()
        except Exception:
            h2 = None

    if h2 is None:
        # Fallback: use TTree.Draw with an optional selection string
        f = ROOT.TFile.Open(file_path)
        if not f or f.IsZombie():
            return f"Error: Could not open file {file_path}"

        tree = f.Get(tree_name)
        if not tree:
            f.Close()
            return f"Error: TTree {tree_name} not found in {file_path}"

        h2 = ROOT.TH2F(_unique_canvas_name("h2"), "", bins_x, xmin, xmax, bins_y, ymin, ymax)
        ROOT.SetOwnership(h2, False)

        sel = ""
        if apply_cuts_before_plot and cuts:
            try:
                vector_vars = _get_vector_branches(file_path, tree_name)
            except Exception:
                vector_vars = []
            sel = " && ".join([_rewrite_vector_cut(c, vector_vars, vector_mode) for c in cuts])

        tree.Draw(f"{y_branch}:{x_branch} >> {h2.GetName()}", sel, "goff")
        f.Close()

    ROOT.SetOwnership(h2, False)
    try:
        h2.SetDirectory(0)
    except Exception:
        pass

    if rebin_x > 1:
        try:
            h2.RebinX(rebin_x)
        except Exception:
            pass
    if rebin_y > 1:
        try:
            h2.RebinY(rebin_y)
        except Exception:
            pass

    if normalize:
        try:
            integral = h2.Integral() if hasattr(h2, "Integral") else h2.GetSumOfWeights()
            if integral and integral > 0:
                h2.Scale(1.0 / integral)
        except Exception:
            pass

    canv = ROOT.TCanvas(_unique_canvas_name("c2d_tree"), "2D Histogram", 900, 700)
    h2.GetXaxis().SetTitle(xlabel if xlabel else x_branch)
    h2.GetYaxis().SetTitle(ylabel if ylabel else y_branch)

    ROOT.gStyle.SetPalette(color_palette)
    h2.Draw("COLZ")

    canv.Update()
    canv.SaveAs(output_pdf)

    return f"Saved 2D histogram ({y_branch} vs {x_branch}) to {output_pdf}"


# ============================================================================
# HISTOGRAM RETRIEVAL & MASS WINDOWS: Load histograms and extract counts
# ============================================================================

def _get_hist(f, name: str):
    # Retrieve a named histogram from an open ROOT file, detaching it from the file's memory.
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


def _counting_window_inputs(
    file_path: str,
    bkg_name: str,
    sig_name: str,
    data_name: str,
    center: float,
    window: float,
):
    # Load signal, background and optional data histograms and integrate them over the mass window.
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

    # Auto-detect histogram range if center/window not provided
    if center is None or window is None:
        ax = hbkg.GetXaxis()
        xmin = ax.GetXmin()
        xmax = ax.GetXmax()
        center = (xmin + xmax) / 2.0
        window = (xmax - xmin) / 2.0

    n_bkg = _fractional_integral(hbkg, center, window)
    n_sig = _fractional_integral(hsig, center, window)
    n_obs = None
    if hdata is not None:
        n_obs = max(0, int(round(_fractional_integral(hdata, center, window))))

    return {"n_bkg": n_bkg, "n_sig": n_sig, "n_obs": n_obs, "center": center, "window": window}, None





# ============================================================================
# COUNTING-STATISTICS UTILITIES: CLs and significance helpers
# ============================================================================

def _clip_probability(value: float) -> float:
    # Clamp a value to [0, 1]; maps non-finite values to 0.
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _poisson_cdf_leq(n: int, mu: float) -> float:
    # Compute P(N <= n | mu) using the Poisson CDF (lower tail).
    n_val = max(0, int(n))
    mu_val = max(0.0, float(mu))
    return _clip_probability(float(poisson.cdf(n_val, mu_val)))


def _counting_cls_poisson_fallback(n_obs: int, n_bkg: float, n_sig: float):
    # Compute CLs, CLs+b and CLb via Poisson CDF lower tails, ensuring CLs in [0, 1].
    clb   = _poisson_cdf_leq(n_obs, max(0.0, float(n_bkg)))
    clspb = _poisson_cdf_leq(n_obs, max(0.0, float(n_bkg) + float(n_sig)))

    cls = float("inf") if clb <= 0.0 else (clspb / clb)
    return _clip_probability(cls), _clip_probability(clspb), _clip_probability(clb)


# ============================================================================
# STATISTICAL CALCULATIONS: Significance and p-values
# ============================================================================

def _fractional_integral(hist, center: float, window: float) -> float:
    # Integrate a TH1 over [center-window, center+window] with fractional bin-edge interpolation.
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


def _poisson_sf_geq(n: int, mu: float) -> float:
    # Compute P(N >= n | mu) using the Poisson survival function (upper tail, p0 convention).
    n_val = max(0, int(n))
    mu_val = max(0.0, float(mu))
    if n_val <= 0:
        return 1.0
    return _clip_probability(float(poisson.sf(n_val - 1, mu_val)))


def _significance_from_pvalue(pvalue: float) -> float:
    # Convert a one-sided p-value to a Gaussian significance Z = Phi^{-1}(1 - p).
    pvalue_val = _clip_probability(float(pvalue))
    if not (0.0 < pvalue_val < 1.0):
        return 0.0
    try:
        return max(0.0, float(norm.isf(pvalue_val)))
    except Exception:
        return 0.0


def _compute_significance_from_yields(S: float, B: float) -> float:
    # Compute Asimov discovery significance Z from signal and background yields.
    s = float(S)
    b = float(B)
    if s <= 0.0 and b <= 0.0:
        return float("nan")
    if b <= 0.0:
        return float("inf") if s > 0.0 else float("nan")

    # Use Asimov observed count n = round(S+B) against background-only hypothesis.
    n_obs = max(0, int(round(s + b)))
    p0 = _poisson_sf_geq(n_obs, b)
    return _significance_from_pvalue(p0)


def _optimal_cut_significance(S: float, B: float) -> float:
    # Return S/sqrt(S+B) as a fast cut-optimisation figure of merit.
    s = max(0.0, float(S))
    b = max(0.0, float(B))
    denom = s + b
    if denom <= 0.0:
        return 0.0
    return s / math.sqrt(denom)


def _compute_counting_stats(n_obs: Optional[int], n_bkg: float, n_sig: float,
                             cl: float = 0.95, compute_cls: bool = True):
    # Compute discovery (p0, Z) and optionally exclusion (CLs, CLs+b, CLb) counting statistics.
    n_b = max(0.0, float(n_bkg))
    n_s = max(0.0, float(n_sig))

    if n_obs is None:
        n_obs_for_observed = max(0, int(round(n_b + n_s)))
    else:
        n_obs_for_observed = max(0, int(n_obs))

    p0_obs = _poisson_sf_geq(n_obs_for_observed, n_b)
    z_obs = _significance_from_pvalue(p0_obs)

    n_exp = max(0, int(round(n_b + n_s)))
    p0_exp = _poisson_sf_geq(n_exp, n_b)
    z_exp = _significance_from_pvalue(p0_exp)

    result = {
        "p0_obs": p0_obs,
        "z_obs": z_obs,
        "p0_exp": p0_exp,
        "z_exp": z_exp,
    }

    if compute_cls:
        cls, clspb, clb = _counting_cls_poisson_fallback(n_obs_for_observed, n_b, n_s)
        result["CLs"] = cls
        result["CLs+b"] = clspb
        result["CLb"] = clb

    return result


def _stat_summary(n_bkg: float, n_sig: float, n_obs: Optional[int] = None,
                  compute_cls: bool = True) -> str:
    # Format a human-readable expected (and optionally observed) statistics summary line.
    n_bkg_val = max(0.0, float(n_bkg))
    n_sig_val = max(0.0, float(n_sig))

    def _line(n: int) -> str:
        n_val = max(0, int(n))
        stats = _compute_counting_stats(n_val, n_bkg_val, n_sig_val, compute_cls=compute_cls)
        p0 = stats.get("p0_obs", float("nan"))
        z = stats.get("z_obs", float("nan"))
        z_str = f"{z:.4g}sigma"
        base = f"N={n_val}  p0={p0:.4g}  Z={z_str}"
        if compute_cls:
            cls = stats.get("CLs", float("nan"))
            clspb = stats.get("CLs+b", float("nan"))
            clb = stats.get("CLb", float("nan"))
            base += f"  CLs={cls:.4g}  CLs+b={clspb:.4g}  CLb={clb:.4g}"
        return base

    n_exp = max(0, int(round(n_bkg_val + n_sig_val)))
    result = f"Expected(S+B Asimov): {_line(n_exp)}"
    if n_obs is not None:
        result += f" | Observed: {_line(n_obs)}"
    return result


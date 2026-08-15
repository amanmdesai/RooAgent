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
    # Returns None on failure instead of raising, since callers format their own error strings.
    paths = _parse_paths(file_path)
    if not paths:
        return None

    target = paths[0]
    try:
        f = ROOT.TFile.Open(target)
    except Exception:
        return None
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
            continue
    return entries


# ============================================================================
# HISTOGRAM UTILITIES: Binning, path parsing, type conversion
# ============================================================================

def _detach(hist):
    # Remove a histogram from its file's ownership so it survives after the file is closed.
    hist.SetDirectory(0)
    ROOT.SetOwnership(hist, False)
    return hist


def _normalize_hist(hist) -> None:
    # Scale a histogram to unit integral in place; no-op if absent or empty.
    if hist is not None and hist.Integral() > 0:
        hist.Scale(1.0 / hist.Integral())


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
        return _detach(hist.Rebin(r, newname))
    except Exception:
        return hist


def _parse_paths(
    primary: Optional[str] = None,
    additional: Optional[List[str]] = None,
    allow_cwd_fallback: bool = True,
) -> List[str]:
    # Directories expand to their .root files; an empty result falls back to cwd's .root files.
    parsed: List[str] = []

    def _add_entry(p: str):
        p = p.strip()
        if not p:
            return
        if os.path.isdir(p):
            for f in _get_root_files(p):
                parsed.append(os.path.abspath(os.path.join(p, f)))
        else:
            parsed.append(os.path.abspath(p))

    if primary:
        for part in [s.strip() for s in str(primary).split(",") if s.strip()]:
            _add_entry(part)

    if additional:
        for part in additional:
            if part:
                _add_entry(part)

    if allow_cwd_fallback and not parsed:
        cwd = os.getcwd()
        for f in _get_root_files(cwd):
            parsed.append(os.path.abspath(os.path.join(cwd, f)))

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
        leaf = branch.GetLeaf(branch.GetName())
        leaf_type = leaf.GetTypeName() if leaf else ""
        type_text = f"{classname} {leaf_type}".lower()
        if "vector" in type_text or "rvec" in type_text:
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

def _open_root_file_or_error(file_path: str):
    # Open a ROOT file, returning (file, None) or (None, error_string) with a standard message.
    f = _open_root_file(file_path)
    if not f:
        return None, f"Error: could not open file {file_path}."
    return f, None


def _load_hist(file_path: str, hist_name: str, rebin: int):
    # Load a named TH1 from a ROOT file and return (histogram, None) or (None, error_string).
    f, err = _open_root_file_or_error(file_path)
    if err:
        return None, err

    h = _get_hist(f, hist_name)
    f.Close()
    if not h:
        return None, f"Error: histogram '{hist_name}' not found in {file_path}."
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
    defines: Optional[Dict[str, str]] = None,
):
    # Project a TTree branch into a TH1 via RDataFrame, applying optional defines, cuts and weights.
    files = _parse_paths(file_path, allow_cwd_fallback=False)
    if not files:
        raise RuntimeError("No ROOT files found to build histogram.")

    df = _build_dataframe(tree_name, files)
    for name, expr in (defines or {}).items():
        df = df.Define(name, expr)
    if apply_cuts_before_plot:
        df = _apply_cuts(df, files[0], tree_name, cuts, vector_mode)

    resolved_weight = _resolve_weight_branch(df, weight_branch)
    if resolved_weight:
        wname = resolved_weight
    else:
        wname = "__rooagent_unit_weight"
        df = df.Define(wname, "1.0")

    hist_ptr = df.Histo1D((hist_name, variable, bins, xmin, xmax), variable, wname)
    h = _detach(hist_ptr.GetValue())
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


def _sum_hists(hist_list: List, name: str):
    # Clone the first histogram and add the rest into it, returning a detached sum.
    total = _detach(hist_list[0].Clone(name))
    for hist in hist_list[1:]:
        total.Add(hist)
    return total


def _draw_unity_line(x_axis):
    # Draw a dashed gray reference line at y=1 across a ratio panel's x-range.
    unity = ROOT.TLine(x_axis.GetXmin(), 1.0, x_axis.GetXmax(), 1.0)
    ROOT.SetOwnership(unity, False)
    unity.SetLineStyle(2)
    unity.SetLineColor(ROOT.kGray + 2)
    unity.Draw()
    return unity


def _styled_legend(x1: float, y1: float, x2: float, y2: float):
    # Common legend chrome (border, font, text size) shared by the comparison/stacked plots.
    legend = ROOT.TLegend(x1, y1, x2, y2)
    legend.SetBorderSize(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)
    return legend


def _build_ratio_hist(numerator, denominator, name: str, binomial: bool = False):
    # Clone numerator and divide by denominator; binomial=True uses "B" (data/MC efficiency-style) errors.
    ratio = _detach(numerator.Clone(name))
    if binomial:
        ratio.Divide(numerator, denominator, 1.0, 1.0, "B")
    else:
        ratio.Divide(denominator)
    return ratio


def _draw_ratio_hist(ratio, lower_pad, x_title: str, y_title: str, draw_option: str):
    # Style, draw, and annotate a prepared ratio histogram in the lower pad with a unity line.
    lower_pad.cd()
    _style_ratio_hist(ratio, x_title, y_title)
    ratio.Draw(draw_option)
    _draw_unity_line(ratio.GetXaxis())


def _build_reference_ratio_hists(hist_list: List):
    # Divide every histogram in the list by the first, returning a list of ratio histograms.
    if len(hist_list) < 2:
        return []

    reference = hist_list[0]
    ratio_hists = []
    for i, hist in enumerate(hist_list[1:], start=1):
        ratio = _detach(hist.Clone(f"{hist.GetName()}_ratio_{i}"))
        ratio.Divide(reference)
        ratio_hists.append(ratio)

    return ratio_hists


def _build_signal_background_ratio(signal_hist, background_hists: List):
    # Divide the signal histogram by the sum of all background histograms.
    if signal_hist is None or not background_hists:
        return None

    summed_background = _sum_hists(background_hists, f"{signal_hist.GetName()}_bkg_sum")
    return _build_ratio_hist(signal_hist, summed_background, f"{signal_hist.GetName()}_over_bkg_ratio")


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

    _draw_unity_line(ratio_hists[0].GetXaxis())


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

def _plot_single_hist(
    h,
    label: str,
    output_pdf: str,
    x_title: str,
    y_title: str,
    normalize: bool,
    logy: bool,
    canvas_name: str,
    canvas_title: str,
):
    # Normalize, style, and draw a single histogram; shared by _plot_hist and _plot_tree.
    if normalize:
        _normalize_hist(h)

    h.SetLineColor(ROOT.kBlue + 1)
    h.SetLineWidth(3)

    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    legend.AddEntry(h, label, "l")

    return _draw_overlay_plot(
        hist_list=[h],
        legend=legend,
        output_pdf=output_pdf,
        x_title=x_title,
        y_title="Normalized Events" if normalize else y_title,
        canvas_name=canvas_name,
        canvas_title=canvas_title,
        show_ratio=False,
        logy=logy,
    )


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

    result = _plot_single_hist(h, hist_name, output_pdf, xlabel, ylabel, normalize, logy, "c_hist", "Histogram")
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

    result = _plot_single_hist(h, variable, output_pdf, variable, "Events", normalize, False, "c_tree", "Tree Variable")
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

        if normalize:
            _normalize_hist(h)

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

    legend = _styled_legend(0.65, 0.7, 0.9, 0.9)
    legend.SetFillColor(0)

    colors = [ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7, ROOT.kCyan + 2]
    hist_list = []
    skipped = []

    for i, (fpath, hname, label) in enumerate(zip(file_paths, hist_names, legends)):
        h, err = _load_hist(fpath, hname, rebin)
        if err:
            skipped.append(err)
            continue

        if normalize:
            _normalize_hist(h)

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


_SIG_BKG_SIGNAL_LINE_COLORS = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]
_SIG_BKG_SIGNAL_HATCH_STYLES = [3345, 3354, 3395, 3444, 3490]
_SIG_BKG_BACKGROUND_FILL_COLORS = [
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


def _draw_comparison_ratio_panel(
    lower_pad,
    variable: str,
    data_hist,
    mc_sum_hist,
    data_ratio_label: str,
    binomial: bool,
    signal_hists: Optional[List] = None,
    background_hists: Optional[List] = None,
):
    # Draw Data/MC (or Sig/Bkg fallback when no data) ratio panel; shared by all three sig-vs-bkg draw paths.
    if lower_pad is None:
        return
    if data_hist is not None:
        ratio = _build_ratio_hist(data_hist, mc_sum_hist, "h_data_ratio", binomial=binomial)
        _draw_ratio_hist(ratio, lower_pad, variable, data_ratio_label, "PE1")
    elif signal_hists and background_hists:
        ratio = _build_signal_background_ratio(signal_hists[0], background_hists)
        _draw_ratio_hist(ratio, lower_pad, variable, "Sig/Bkg", "HIST")


def _build_sig_bkg_hists(
    tree_name: str,
    variable: str,
    bins: int,
    xmin: float,
    xmax: float,
    signal_paths: List[str],
    background_paths: List[str],
    data_file: str,
    wants_data: bool,
    weight_branch: str,
    cuts: Optional[List[str]],
    vector_mode: str,
    apply_cuts_before_plot: bool,
    rebin: int,
):
    # Build and style the raw (unnormalized) background, signal, and optional data histograms.
    background_hists = []
    for i, fpath in enumerate(background_paths):
        h = _build_tree_hist(
            file_path=fpath, tree_name=tree_name, variable=variable, bins=bins, xmin=xmin, xmax=xmax,
            hist_name=f"h_sigbkg_bkg_{i}", weight_branch=weight_branch, cuts=cuts, vector_mode=vector_mode,
            rebin=rebin, apply_cuts_before_plot=apply_cuts_before_plot,
        )
        h.SetFillColor(_SIG_BKG_BACKGROUND_FILL_COLORS[i % len(_SIG_BKG_BACKGROUND_FILL_COLORS)])
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(1)
        background_hists.append(h)

    signal_hists = []
    for i, fpath in enumerate(signal_paths):
        h = _build_tree_hist(
            file_path=fpath, tree_name=tree_name, variable=variable, bins=bins, xmin=xmin, xmax=xmax,
            hist_name=f"h_sigbkg_sig_{i}", weight_branch=weight_branch, cuts=cuts, vector_mode=vector_mode,
            rebin=rebin, apply_cuts_before_plot=apply_cuts_before_plot,
        )
        line_color = _SIG_BKG_SIGNAL_LINE_COLORS[i % len(_SIG_BKG_SIGNAL_LINE_COLORS)]
        h.SetFillStyle(0)
        h.SetLineColor(line_color)
        h.SetLineWidth(3)
        h.SetLineStyle(1 + (i % 4))
        signal_hists.append(h)

    data_hist = None
    if wants_data:
        data_hist = _build_tree_hist(
            file_path=data_file, tree_name=tree_name, variable=variable, bins=bins, xmin=xmin, xmax=xmax,
            hist_name="h_sigbkg_data", weight_branch="", cuts=cuts, vector_mode=vector_mode,
            rebin=rebin, apply_cuts_before_plot=apply_cuts_before_plot,
        )
        data_hist.SetFillStyle(0)
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerSize(1.0)
        data_hist.SetMarkerColor(ROOT.kBlack)
        data_hist.SetLineColor(ROOT.kBlack)
        data_hist.SetLineWidth(1)

    return background_hists, signal_hists, data_hist


def _normalize_sig_bkg_hists(background_hists: List, signal_hists: List, data_hist):
    # Scale each histogram to unit integral and reset background line styling for the normalized overlay look.
    for i, h in enumerate(background_hists):
        _normalize_hist(h)
        h.SetFillStyle(0)
        h.SetLineColor(_SIG_BKG_BACKGROUND_FILL_COLORS[i % len(_SIG_BKG_BACKGROUND_FILL_COLORS)])
        h.SetLineWidth(2)

    for h in signal_hists:
        _normalize_hist(h)

    _normalize_hist(data_hist)


def _build_sig_bkg_legend(
    signal_hists: List, signal_labels: List[str], background_hists: List, background_labels: List[str],
    data_hist, data_label: str, use_full_stack: bool, normalize: bool,
):
    # Assemble the shared legend (data, then signals, then backgrounds, each reversed for stack order).
    legend = _styled_legend(0.62, 0.68, 0.88, 0.88)
    legend.SetFillStyle(0)

    if data_hist:
        legend.AddEntry(data_hist, data_label, "lep")

    sig_legend_opt = "f" if use_full_stack else "l"
    for h, label in reversed(list(zip(signal_hists, signal_labels))):
        legend.AddEntry(h, label, sig_legend_opt)

    bkg_legend_opt = "l" if normalize else "f"
    for h, label in reversed(list(zip(background_hists, background_labels))):
        legend.AddEntry(h, label, bkg_legend_opt)

    return legend


def _draw_sig_bkg_overlay(
    background_hists: List, signal_hists: List, data_hist, legend, variable: str, y_title: str,
    output_pdf: str, show_ratio: bool, canvas, lower_pad,
):
    # Draw the (non-stacked) overlay path: all hists as lines/markers on one pad, no THStack.
    hist_list = background_hists + signal_hists + ([data_hist] if data_hist else [])
    draw_options = ["HIST"] * (len(background_hists) + len(signal_hists))
    if data_hist:
        draw_options.append("PE1")

    result = _draw_overlay_plot(
        hist_list=hist_list, legend=legend, output_pdf=output_pdf, x_title=variable, y_title=y_title,
        canvas_name="c_sig_bkg_overlay", canvas_title="Signal vs Backgrounds",
        show_ratio=show_ratio, draw_options=draw_options,
    )
    if result == "No histograms created.":
        return result

    if show_ratio and lower_pad is not None:
        mc_sum_hist = _sum_hists(background_hists, "h_all_bkg_ratio") if data_hist else None
        _draw_comparison_ratio_panel(
            lower_pad, variable, data_hist, mc_sum_hist, "Data/Bkg", binomial=False,
            signal_hists=signal_hists, background_hists=background_hists,
        )
        canvas.Update()
        canvas.SaveAs(output_pdf)

    return f"Saved signal-vs-background comparison to {output_pdf}"


def _build_and_draw_stack(name: str, stacked_hists: List, variable: str, y_title: str,
                           show_ratio: bool, extra_max_hists: Optional[List] = None):
    # Build/draw a THStack sized to fit both its own contents and any unstacked overlay hists.
    stack = ROOT.THStack(name, "")
    ROOT.SetOwnership(stack, False)
    for h in stacked_hists:
        stack.Add(h)

    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(variable)
    stack.GetYaxis().SetTitle(y_title)
    if show_ratio:
        stack.GetXaxis().SetLabelSize(0)
        stack.GetXaxis().SetTitleSize(0)
        stack.GetYaxis().SetTitleSize(0.055)
        stack.GetYaxis().SetLabelSize(0.045)

    extra_max = max((h.GetMaximum() for h in (extra_max_hists or [])), default=0.0)
    stack.SetMaximum(max(stack.GetMaximum(), extra_max) * 1.35)
    return stack


def _draw_stat_band(hist_list: List, name: str):
    # Sum hists into a hashed gray stat-error band and draw it as E2 on the current pad.
    band = _sum_hists(hist_list, name)
    band.SetFillStyle(3354)
    band.SetFillColor(ROOT.kGray + 2)
    band.SetLineColor(ROOT.kGray + 2)
    band.SetMarkerSize(0)
    band.Draw("E2 SAME")
    return band


def _draw_sig_bkg_stack_bkg_only(
    background_hists: List, signal_hists: List, data_hist, legend, variable: str, y_title: str,
    output_pdf: str, show_ratio: bool, canvas, draw_pad, lower_pad,
):
    # Draw backgrounds as a THStack with signal(s) and data overlaid unstacked (stack_signal=False).
    draw_pad.cd()
    _build_and_draw_stack(
        "hs_bkg_only", background_hists, variable, y_title, show_ratio,
        extra_max_hists=signal_hists + ([data_hist] if data_hist else []),
    )
    bkg_sum = _draw_stat_band(background_hists, "h_bkg_stat_band")

    for h in signal_hists:
        h.Draw("HIST SAME")

    if data_hist:
        data_hist.Draw("PE1 SAME")

    legend.Draw()

    if show_ratio and lower_pad is not None:
        _draw_comparison_ratio_panel(
            lower_pad, variable, data_hist, bkg_sum, "Data/Bkg", binomial=True,
            signal_hists=signal_hists, background_hists=background_hists,
        )

    canvas.Update()
    canvas.SaveAs(output_pdf)
    return f"Saved signal-vs-background comparison to {output_pdf}"


def _draw_sig_bkg_full_stack(
    background_hists: List, signal_hists: List, data_hist, legend, variable: str, y_title: str,
    output_pdf: str, show_ratio: bool, canvas, lower_pad,
):
    # Draw signal(s) stacked on top of backgrounds in a single THStack, with data overlaid.
    _build_and_draw_stack(
        "hs_sig_bkg", background_hists + signal_hists, variable, y_title, show_ratio,
        extra_max_hists=[data_hist] if data_hist else [],
    )
    mc_sum = _draw_stat_band(background_hists + signal_hists, "h_mc_stat_band")

    if data_hist:
        data_hist.Draw("PE1 SAME")

    legend.Draw()

    if show_ratio and lower_pad is not None:
        _draw_comparison_ratio_panel(lower_pad, variable, data_hist, mc_sum, "Data/MC", binomial=True)

    canvas.Update()
    canvas.SaveAs(output_pdf)
    return f"Saved signal-vs-background comparison to {output_pdf}"


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
    # stack_signal toggles whether signal is stacked on top of backgrounds or overlaid as lines.
    signal_paths = _parse_paths(signal_file, signal_files, allow_cwd_fallback=False)

    background_paths = _parse_paths(additional=background_files, allow_cwd_fallback=False)
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

    background_hists, signal_hists, data_hist = _build_sig_bkg_hists(
        tree_name, variable, bins, xmin, xmax, signal_paths, background_paths, data_file, wants_data,
        weight_branch, cuts, vector_mode, apply_cuts_before_plot, rebin,
    )

    y_title = "Normalized Events" if normalize else "Events"
    if normalize:
        _normalize_sig_bkg_hists(background_hists, signal_hists, data_hist)

    use_full_stack = wants_data and not normalize and stack_signal and bool(signal_hists)
    use_bkg_stack = wants_data and not normalize and (not stack_signal) and bool(background_hists)

    if use_full_stack:
        for i, h in enumerate(signal_hists):
            line_color = _SIG_BKG_SIGNAL_LINE_COLORS[i % len(_SIG_BKG_SIGNAL_LINE_COLORS)]
            h.SetFillStyle(_SIG_BKG_SIGNAL_HATCH_STYLES[i % len(_SIG_BKG_SIGNAL_HATCH_STYLES)])
            h.SetFillColor(line_color)
            h.SetLineColor(line_color)
            h.SetLineWidth(2)
            h.SetLineStyle(1)

    legend = _build_sig_bkg_legend(
        signal_hists, signal_labels, background_hists, background_labels,
        data_hist, data_label, use_full_stack, normalize,
    )

    canvas, upper_pad, lower_pad = _create_plot_pads(
        _unique_canvas_name("c_sig_bkg"), "Signal vs Backgrounds", show_ratio
    )
    draw_pad = upper_pad if upper_pad else canvas
    draw_pad.cd()

    if not use_full_stack and not use_bkg_stack:
        return _draw_sig_bkg_overlay(
            background_hists, signal_hists, data_hist, legend, variable, y_title,
            output_pdf, show_ratio, canvas, lower_pad,
        )

    if use_bkg_stack:
        return _draw_sig_bkg_stack_bkg_only(
            background_hists, signal_hists, data_hist, legend, variable, y_title,
            output_pdf, show_ratio, canvas, draw_pad, lower_pad,
        )

    return _draw_sig_bkg_full_stack(
        background_hists, signal_hists, data_hist, legend, variable, y_title,
        output_pdf, show_ratio, canvas, lower_pad,
    )


# ============================================================================
# 2D PLOTTING: 2D histograms and kinematic correlations
# ============================================================================

def _draw_2d_hist_and_save(h2, canvas_name: str, xlabel: str, ylabel: str, color_palette: int,
                            output_pdf: str, zlabel: str = "", logz: bool = False):
    # Style a TH2 (axis titles, z-palette), draw it COLZ, and save; shared by _plot_2d_hist/_plot_2d_tree.
    canv = ROOT.TCanvas(_unique_canvas_name(canvas_name), "2D Histogram", 900, 700)
    h2.GetXaxis().SetTitle(xlabel)
    h2.GetYaxis().SetTitle(ylabel)
    if zlabel:
        h2.GetZaxis().SetTitle(zlabel)
    if logz:
        canv.SetLogz(1)

    ROOT.gStyle.SetPalette(color_palette)
    h2.Draw("COLZ")
    canv.Update()
    canv.SaveAs(output_pdf)
    return canv


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
    h, err = _load_hist(file_path, hist_name, 1)
    if err:
        return err

    if rebin_x > 1:
        h.RebinX(rebin_x)
    if rebin_y > 1:
        h.RebinY(rebin_y)

    if normalize:
        _normalize_hist(h)

    _draw_2d_hist_and_save(h, "c2d", xlabel, ylabel, color_palette, output_pdf, zlabel=zlabel, logz=logz)
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
        df = None

    h2 = None
    if df is not None:
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
        f, err = _open_root_file_or_error(file_path)
        if err:
            return err

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

    _draw_2d_hist_and_save(h2, "c2d_tree", xlabel if xlabel else x_branch, ylabel if ylabel else y_branch,
                            color_palette, output_pdf)
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
        _detach(h)
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

    f, err = _open_root_file_or_error(p)
    if err:
        return None, err

    hbkg = _get_hist(f, bkg_name)
    hsig = _get_hist(f, sig_name)
    hdata = _get_hist(f, data_name) if data_name and data_name.strip() else None

    f.Close()

    if hbkg is None or hsig is None:
        return None, "Error: required histograms not found in file."

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
    if pvalue_val <= 0.0:
        return float("inf")
    if pvalue_val >= 1.0:
        return 0.0
    try:
        return max(0.0, float(norm.isf(pvalue_val)))
    except Exception:
        return 0.0


def _asimov_discovery_significance(n_sig: float, n_bkg: float) -> float:
    # Compute Asimov discovery significance Z from signal and background yields.
    s = max(0.0, float(n_sig))
    b = max(0.0, float(n_bkg))
    if s <= 0.0 and b <= 0.0:
        return float("nan")
    if b <= 0.0:
        return float("inf") if s > 0.0 else float("nan")

    return math.sqrt(2.0 * ((s + b) * math.log1p(s / b) - s))


def _compute_significance_from_yields(S: float, B: float) -> float:
    return _asimov_discovery_significance(S, B)


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

    def _line(n: int, expected_asimov: bool = False) -> str:
        n_val = max(0, int(n))
        if expected_asimov:
            z = _asimov_discovery_significance(n_sig_val, n_bkg_val)
            p0 = _clip_probability(float(norm.sf(z))) if math.isfinite(z) else 0.0
            stats = _compute_counting_stats(n_val, n_bkg_val, n_sig_val, compute_cls=compute_cls)
        else:
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
    result = f"Expected(S+B Asimov): {_line(n_exp, expected_asimov=True)}"
    if n_obs is not None:
        result += f" | Observed: {_line(n_obs)}"
    return result


# ROOFIT MODEL BUILDING: shared dataset/pdf construction for fit_model and the RooStats tools.

_SIGNAL_SHAPES = {"gauss", "crystalball", "voigt"}
_BACKGROUND_SHAPES = {"", "expo", "chebychev", "poly"}


def _build_roofit_dataset(source, file_path, tree_name, variable, hist_name, xmin, xmax):
    # Build a RooFit observable + dataset from a TTree branch (unbinned) or a stored TH1 (binned).
    if source == "tree":
        if not tree_name or not variable:
            return None, "Error: tree_name and variable are required when source='tree'."
        if xmin >= xmax:
            return None, "Error: xmin must be less than xmax."

        f = _open_root_file(file_path)
        if not f:
            return None, "Error: could not open ROOT file."

        tree = f.Get(tree_name)
        if not tree:
            f.Close()
            return None, f"Error: tree '{tree_name}' not found."

        x = ROOT.RooRealVar(variable, variable, xmin, xmax)
        data = ROOT.RooDataSet(
            _unique_canvas_name("ds"), "ds", ROOT.RooArgSet(x), ROOT.RooFit.Import(tree)
        )
        f.Close()
        return {
            "x": x, "data": data, "n_events": data.numEntries(),
            "label": variable, "xmin": xmin, "xmax": xmax,
        }, None

    elif source == "hist":
        if not hist_name:
            return None, "Error: hist_name is required when source='hist'."

        f = _open_root_file(file_path)
        if not f:
            return None, "Error: could not open ROOT file."

        h = _get_hist(f, hist_name)
        f.Close()
        if not h:
            return None, f"Error: histogram '{hist_name}' not found."

        hxmin, hxmax = h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax()
        x = ROOT.RooRealVar(hist_name, hist_name, hxmin, hxmax)
        data = ROOT.RooDataHist(_unique_canvas_name("dh"), "dh", ROOT.RooArgList(x), h)
        return {
            "x": x, "data": data, "n_events": data.sumEntries(),
            "label": hist_name, "xmin": hxmin, "xmax": hxmax,
        }, None

    else:
        return None, "Error: source must be 'tree' or 'hist'."


def _build_signal_pdf(x, shape_key, mean=0.0, mean_range=None, sigma=1.0, sigma_range=None,
                       alpha=1.5, alpha_range=None, n=2.0, n_range=None,
                       width=0.1, width_range=None, xmin=None, xmax=None):
    # Build a signal RooAbsPdf on observable x from one of fit_model's shape presets.
    mean_lo, mean_hi = mean_range or (xmin, xmax)
    sigma_lo, sigma_hi = sigma_range or (1e-3, xmax - xmin)

    if shape_key == "gauss":
        mean_v = ROOT.RooRealVar("mean", "mean", mean, mean_lo, mean_hi)
        sigma_v = ROOT.RooRealVar("sigma", "sigma", sigma, sigma_lo, sigma_hi)
        pdf = ROOT.RooGaussian("gauss", "gauss", x, mean_v, sigma_v)
        params = [mean_v, sigma_v]
    elif shape_key == "crystalball":
        mean_v = ROOT.RooRealVar("mean", "mean", mean, mean_lo, mean_hi)
        sigma_v = ROOT.RooRealVar("sigma", "sigma", sigma, sigma_lo, sigma_hi)
        alpha_lo, alpha_hi = alpha_range or (0.05, 10.0)
        n_lo, n_hi = n_range or (0.5, 60.0)
        alpha_v = ROOT.RooRealVar("alpha", "alpha", alpha, alpha_lo, alpha_hi)
        n_v = ROOT.RooRealVar("n", "n", n, n_lo, n_hi)
        pdf = ROOT.RooCBShape("crystalball", "crystalball", x, mean_v, sigma_v, alpha_v, n_v)
        params = [mean_v, sigma_v, alpha_v, n_v]
    elif shape_key == "voigt":
        mean_v = ROOT.RooRealVar("mean", "mean", mean, mean_lo, mean_hi)
        sigma_v = ROOT.RooRealVar("sigma", "sigma", sigma, sigma_lo, sigma_hi)
        width_lo, width_hi = width_range or (1e-3, xmax - xmin)
        width_v = ROOT.RooRealVar("width", "width", width, width_lo, width_hi)
        pdf = ROOT.RooVoigtian("voigt", "voigt", x, mean_v, width_v, sigma_v)
        params = [mean_v, sigma_v, width_v]
    else:
        return None, f"Error: signal_shape must be one of {sorted(_SIGNAL_SHAPES)}."

    return {"pdf": pdf, "params": params}, None


def _build_background_pdf(x, bkg_key, tau=-0.1, tau_range=None, poly_coeffs=None, poly_coeff_range=1.0):
    # Build a background RooAbsPdf on observable x, or (None, None) if bkg_key == ''.
    if not bkg_key:
        return None, None

    if bkg_key == "expo":
        tau_lo, tau_hi = tau_range or (-10.0, 10.0)
        tau_v = ROOT.RooRealVar("tau", "tau", tau, tau_lo, tau_hi)
        pdf = ROOT.RooExponential("expo", "expo", x, tau_v)
        return {"pdf": pdf, "params": [tau_v]}, None

    if bkg_key in ("chebychev", "poly"):
        coeffs = poly_coeffs or [0.0]
        coeff_vars = ROOT.RooArgList()
        params = []
        for i, c0 in enumerate(coeffs, start=1):
            cv = ROOT.RooRealVar(f"c{i}", f"c{i}", c0, -poly_coeff_range, poly_coeff_range)
            coeff_vars.add(cv)
            params.append(cv)
        if bkg_key == "chebychev":
            pdf = ROOT.RooChebychev("chebychev", "chebychev", x, coeff_vars)
        else:
            pdf = ROOT.RooPolynomial("poly", "poly", x, coeff_vars)
        return {"pdf": pdf, "params": params}, None

    return None, f"Error: background_shape must be one of {sorted(s for s in _BACKGROUND_SHAPES if s)} or ''."


def _build_extended_model(sig_pdf, bkg_pdf, n_events, nsig_init=100.0, nsig_range=None,
                           nbkg_init=100.0, nbkg_range=None):
    # Compose an extended RooAddPdf(signal, background) with floating nsig/nbkg yields.
    if bkg_pdf is None:
        return None, "Error: _build_extended_model requires a background pdf."
    nsig_lo, nsig_hi = nsig_range or (0.0, 2 * max(n_events, 1.0))
    nbkg_lo, nbkg_hi = nbkg_range or (0.0, 2 * max(n_events, 1.0))
    nsig = ROOT.RooRealVar("nsig", "nsig", nsig_init, nsig_lo, nsig_hi)
    nbkg = ROOT.RooRealVar("nbkg", "nbkg", nbkg_init, nbkg_lo, nbkg_hi)
    model = ROOT.RooAddPdf(
        _unique_canvas_name("model"), "model",
        ROOT.RooArgList(sig_pdf, bkg_pdf), ROOT.RooArgList(nsig, nbkg),
    )
    return {"model": model, "nsig": nsig, "nbkg": nbkg}, None


# ============================================================================
# ROOSTATS MODEL BUILDING & CLS SCAN: shared by roostats_tools.py
# ============================================================================

_REQUIRED_BACKGROUND_SHAPES = sorted(s for s in _BACKGROUND_SHAPES if s)

# Wider than fit_model's default (0, 2*N_events) since CLs limits can sit well above N_events.
_ROOSTATS_YIELD_RANGE_MULT = 20.0
_POI_BOUNDARY_REL_TOL = 1e-3
_POI_EXPAND_FACTOR = 10.0
_POI_MAX_EXPANSIONS = 5


def _prepare_roostats_model(
    source, signal_shape, background_shape, file_path, tree_name, variable, hist_name,
    xmin, xmax, mean, mean_range, sigma, sigma_range, alpha, alpha_range, n, n_range,
    width, width_range, tau, tau_range, poly_coeffs, poly_coeff_range,
    nsig_init, nsig_range, nbkg_init, nbkg_range,
):
    # Shared by compute_discovery_significance/compute_upper_limit; unlike fit_model, background_shape is required.
    source_key = (source or "").strip().lower()
    shape_key = (signal_shape or "").strip().lower()
    bkg_key = (background_shape or "").strip().lower()

    if source_key not in {"tree", "hist"}:
        return None, "Error: source must be 'tree' or 'hist'."
    if shape_key not in _SIGNAL_SHAPES:
        return None, f"Error: signal_shape must be one of {sorted(_SIGNAL_SHAPES)}."
    if bkg_key not in _REQUIRED_BACKGROUND_SHAPES:
        return None, (
            f"Error: background_shape is required and must be one of {_REQUIRED_BACKGROUND_SHAPES} "
            "(a background model is required for discovery/limit calculations)."
        )

    ds_info, err = _build_roofit_dataset(source_key, file_path, tree_name, variable, hist_name, xmin, xmax)
    if err:
        return None, err
    x = ds_info["x"]
    data = ds_info["data"]
    n_events = ds_info["n_events"]
    label = ds_info["label"]
    xmin, xmax = ds_info["xmin"], ds_info["xmax"]

    sig_info, err = _build_signal_pdf(
        x, shape_key, mean, mean_range, sigma, sigma_range, alpha, alpha_range,
        n, n_range, width, width_range, xmin, xmax,
    )
    if err:
        return None, err

    bkg_info, err = _build_background_pdf(x, bkg_key, tau, tau_range, poly_coeffs, poly_coeff_range)
    if err:
        return None, err

    if nsig_range is None:
        nsig_range = (0.0, _ROOSTATS_YIELD_RANGE_MULT * max(n_events, 1.0))
    if nbkg_range is None:
        nbkg_range = (0.0, _ROOSTATS_YIELD_RANGE_MULT * max(n_events, 1.0))

    ext_info, err = _build_extended_model(
        sig_info["pdf"], bkg_info["pdf"], n_events, nsig_init, nsig_range, nbkg_init, nbkg_range
    )
    if err:
        return None, err

    return {
        "x": x, "data": data, "n_events": n_events, "label": label,
        "model": ext_info["model"], "nsig": ext_info["nsig"], "nbkg": ext_info["nbkg"],
        "shapes": f"{shape_key}+{bkg_key}",
        "_sig_info": sig_info, "_bkg_info": bkg_info,
    }, None


def _build_sb_model_configs(x, data, model, nsig):
    # Two ModelConfigs sharing one imported pdf, differing only by the nsig snapshot (S+B vs. 0).
    try:
        w = ROOT.RooWorkspace(_unique_canvas_name("ws"))
        getattr(w, "import")(model)
        data_name = _unique_canvas_name("obsdata")
        getattr(w, "import")(data, ROOT.RooFit.Rename(data_name))

        w_pdf = w.pdf(model.GetName())
        w_x = w.var(x.GetName())
        w_nsig = w.var(nsig.GetName())
        w_obs = ROOT.RooArgSet(w_x)

        all_params = w_pdf.getParameters(w_obs)
        nuisance_params = all_params.Clone(_unique_canvas_name("nuisance"))
        nuisance_params.remove(w_nsig)

        mc_sb = ROOT.RooStats.ModelConfig(_unique_canvas_name("ModelConfig_sb"), w)
        mc_sb.SetPdf(w_pdf)
        mc_sb.SetParametersOfInterest(ROOT.RooArgSet(w_nsig))
        mc_sb.SetObservables(w_obs)
        mc_sb.SetNuisanceParameters(nuisance_params)
        w_nsig.setVal(nsig.getVal())
        mc_sb.SetSnapshot(ROOT.RooArgSet(w_nsig))
        getattr(w, "import")(mc_sb)

        mc_b = mc_sb.Clone(_unique_canvas_name("ModelConfig_bonly"))
        w_nsig.setVal(0.0)
        mc_b.SetSnapshot(ROOT.RooArgSet(w_nsig))
        getattr(w, "import")(mc_b)

        return {
            "w": w, "mc_sb": mc_sb, "mc_b": mc_b,
            "data_name": data_name, "nsig_name": w_nsig.GetName(),
        }, None
    except Exception as exc:
        return None, f"Error: {exc}."


def _run_cls_scan(w, ws_info, confidence_level, scan_points, poi_min, poi_max):
    # Fresh calculator each call: the adaptive scan reads the POI range at construction time.
    ac = ROOT.RooStats.AsymptoticCalculator(w.data(ws_info["data_name"]), ws_info["mc_b"], ws_info["mc_sb"])
    ac.SetOneSided(True)

    hti = ROOT.RooStats.HypoTestInverter(ac)
    hti.SetConfidenceLevel(confidence_level)
    hti.UseCLs(True)
    hti.SetVerbose(False)

    if scan_points is not None:
        w_nsig = w.var(ws_info["nsig_name"])
        lo = poi_min if poi_min is not None else w_nsig.getMin()
        hi = poi_max if poi_max is not None else w_nsig.getMax()
        hti.SetFixedScan(int(scan_points), lo, hi)

    result = hti.GetInterval()
    obs_ul = result.UpperLimit()
    med = result.GetExpectedUpperLimit(0)
    p1 = result.GetExpectedUpperLimit(1)
    m1 = result.GetExpectedUpperLimit(-1)
    p2 = result.GetExpectedUpperLimit(2)
    m2 = result.GetExpectedUpperLimit(-2)
    return obs_ul, med, p1, m1, p2, m2


def _at_poi_boundary(values, hi, rel_tol=_POI_BOUNDARY_REL_TOL):
    return any(v is not None and v >= hi * (1.0 - rel_tol) for v in values)

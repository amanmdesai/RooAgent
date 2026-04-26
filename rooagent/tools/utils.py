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

# File I/O: Open ROOT file with zombie check
def _open_root_file(file_path: Optional[str] = None):
    """Open a ROOT file resolving directories and defaults.

    If `file_path` is a directory, or falsy, try to find .root files in the
    directory (current working directory when omitted) and open the first one.
    Returns a TFile or None on failure.
    """
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


# File discovery: List .root files in directory
def _get_root_files(directory: str = ".") -> List[str]:
    try:
        return [f for f in os.listdir(directory) if f.lower().endswith(".root")]
    except Exception:
        return []


# File introspection: Extract TTree names from ROOT file
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


# File exploration: Recursively list all ROOT objects in directory hierarchy
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


# ============================================================================
# HISTOGRAM UTILITIES: Binning, path parsing, type conversion
# ============================================================================

# Histogram coarsening: Combine bins to reduce resolution
def _rebin_hist(hist, rebin: int):
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


# Path aggregation: Merge comma-separated and list file paths, removing duplicates
def _parse_paths(primary: Optional[str] = None, additional: Optional[List[str]] = None) -> List[str]:
    """Resolve primary and additional path inputs to a list of file paths.

    Behavior:
    - Accepts comma-separated strings in `primary` and lists in `additional`.
    - If an entry is a directory, expand to all `.root` files inside it.
    - If no paths are provided or nothing resolves, default to all `.root`
      files in the current working directory.
    - Returns absolute, deduplicated paths preserving order.
    """
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


# Type conversion: Parse string/list/scalar into float array
def _to_float_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [float(s.strip()) for s in v.split(",") if s.strip()]
    if isinstance(v, (list, tuple)):
        return [float("nan") if x is None else float(x) for x in v]
    return [float(v)]


def _to_int_list(v):
    # Parse input (scalar, list/tuple, or CSV string) into a list of ints.
    # Returns an empty list on invalid input.
    if v is None:
        return []
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
    # Parse and validate numeric input into a list of numbers.
    # `integer=True` returns rounded integers, otherwise floats.
    parsed = _to_int_list(values) if integer else _to_float_list(values)
    if any(not math.isfinite(value) for value in parsed):
        raise ValueError("values must be finite numeric values")
    return parsed


def _normalize_parallel_arrays(reference_name: str, reference_values, series: Dict[str, object], integer_series=None):
    # Parse and validate parallel arrays against a reference array.
    # Optionally parse specified series as integers.
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
    # Choose a sensible default sort key from available series names.
    preferred = [
        "z_obs",
        "observed_significance",
        "significance",
        "z_exp",
        "expected_significance",
        "cls",
    ]
    lowered = {name.lower(): name for name in series_names}
    for candidate in preferred:
        if candidate in lowered:
            return lowered[candidate]
    return series_names[0]


def _scan_sort_descending(series_name: str) -> bool:
    # Return True when larger series values should be considered better.
    lowered = series_name.lower().replace("_", " ")
    lower_better_tokens = ("cls", "p0", "p value", "p-value", "mu up", "yield up")
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
    # Format a compact, human-readable summary of a parameter scan.
    # Combines parameter values with series and lists top-ranked candidates.
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

# Vector detection: Identify branches with std::vector type for VecOps rewriting
def _get_vector_branches(file_path: str, tree_name: str) -> List[str]:
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


# Vector cut rewriting: Convert vector comparisons to ROOT::VecOps::Any/All for RDataFrame
def _rewrite_vector_cut(cut: str, vector_vars: List[str], mode: str = "any") -> str:
    # Always normalise Python boolean/logical syntax to C++ equivalents so that
    # cuts written naturally by an LLM (True, False, and, or) are valid for
    # RDataFrame regardless of whether vector rewriting is needed.
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

# Lazy evaluation: Initialize RDataFrame from tree (single or merged files)
def _build_dataframe(tree_name: str, files: List[str]):
    if len(files) == 1:
        return ROOT.RDataFrame(tree_name, files[0])
    return ROOT.RDataFrame(tree_name, files)


# Introspection: Check if column exists in RDataFrame
def _has_column(df, column_name: str) -> bool:
    if not column_name:
        return False

    cols = [str(c) for c in df.GetColumnNames()]
    return column_name in cols


# Event counting: Sum weighted events passing a selection cut
def _filtered_yield(df, cut: str, weight: Optional[str] = None):
    filtered = df.Filter(cut)
    if weight and _has_column(filtered, weight):
        return filtered.Sum(weight).GetValue()
    return filtered.Count().GetValue()


# Total yield: Sum all events (weighted or unweighted)
def _total_yield(df, weight: Optional[str] = None):
    if weight and _has_column(df, weight):
        return df.Sum(weight).GetValue()
    return df.Count().GetValue()


# Plotting utility: Generate unique canvas identifier
def _unique_canvas_name(base: str) -> str:
    _canvas_counter[0] += 1
    return f"{base}_{_canvas_counter[0]}"


# Event selection: Apply ordered physics cuts (auto-rewrites vector expressions)
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


# Weight validation: Ensure weight branch exists in dataframe
def _resolve_weight_branch(df, weight_branch: str) -> str:
    if not weight_branch:
        return ""
    return weight_branch if _has_column(df, weight_branch) else ""


# ============================================================================
# HISTOGRAM I/O & BUILDING: Load, create, and manipulate histograms
# ============================================================================

# Histogram I/O: Load stored histogram from ROOT file
def _load_hist(file_path: str, hist_name: str, rebin: int):
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


# Histogram creation: Build 1D histogram from tree variable with cuts and weights
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

# Plotting canvas: Setup upper plot + optional lower ratio pad
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


# Ratio styling: Format ratio histogram axes and range for data/MC or S/B
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


# Ratio histograms: Compute histogram ratios (each hist divided by first)
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


# Signal-background ratio: Divide signal by summed backgrounds
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


# Ratio panel: Draw ratio histograms in lower pad with reference line
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


# Overlay plot: Draw multiple histograms on same axes with optional ratio pad
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


# ============================================================================
# 1D PLOTTING: Histograms, trees, comparisons, signal vs background
# ============================================================================

# 1D histogram plot: Visualize stored histogram with optional normalization
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


# Tree variable plot: Visualize tree branch with cuts and weights
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


# Overlaid tree comparison: Plot same variable from multiple files with ratio
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


# Overlaid histogram comparison: Plot stored histograms from multiple files
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


# Signal-background visualization: Overlay signal+background+optional data with S/B ratio panel
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
):
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


# ============================================================================
# 2D PLOTTING: 2D histograms and kinematic correlations
# ============================================================================

# 2D histogram plot: Visualize stored 2D histogram with color palette
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


# 2D scatter plot: Visualize kinematic correlations from tree branches
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
    # Prefer RDataFrame->Histo2D so we can apply cuts and weights consistently.
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

# Histogram retrieval: Load histogram from open ROOT file
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


# Counting window: Extract signal/background counts in mass window from file
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

# Probability clipping: Ensure probability is in valid [0,1] range
def _clip_probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _poisson_cdf_leq(n: int, mu: float) -> float:
    n_val = max(0, int(n))
    mu_val = max(0.0, float(mu))
    return _clip_probability(float(poisson.cdf(n_val, mu_val)))


def _counting_cls_poisson_fallback(n_obs: int, n_bkg: float, n_sig: float):
    # Use right-tail (>= n_obs) Poisson survival function for counting CLs.
    # This matches the standard CLs convention where larger counts are
    # considered more signal-like. Returns (CLs, CLs+b, CLb).
    clb = _poisson_sf_geq(n_obs, max(0.0, float(n_bkg)))
    clspb = _poisson_sf_geq(n_obs, max(0.0, float(n_bkg) + float(n_sig)))

    cls = float("inf") if clb <= 0.0 else (clspb / clb)
    return _clip_probability(cls), _clip_probability(clspb), _clip_probability(clb)


def _exclusion_summary(n_obs: int, n_bkg: float, n_sig: float):
    """Pure-Python CLs summary for counting experiments.
    Returns (CLs, CLs+b, CLb) computed from Poisson survival functions using
    the right-tail convention (>= n_obs), consistent with common HEP
    CLs definitions.
    """
    return _counting_cls_poisson_fallback(n_obs, n_bkg, n_sig)
# ============================================================================
# STATISTICAL CALCULATIONS: Significance and p-values
# ============================================================================

# Mass window counting: Integrate histogram over fractional bins in window
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


def _poisson_sf_geq(n: int, mu: float) -> float:
    # Survival function P(N >= n) = 1 - CDF(n-1)
    n_val = max(0, int(n))
    mu_val = max(0.0, float(mu))
    if n_val <= 0:
        return 1.0
    return _clip_probability(float(poisson.sf(n_val - 1, mu_val)))


# P-value to Z: Convert p-value to Gaussian discovery significance
def _significance_from_pvalue(pvalue: float) -> float:
    pvalue_val = _clip_probability(float(pvalue))
    if not (0.0 < pvalue_val < 1.0):
        return 0.0
    try:
        return max(0.0, float(norm.isf(pvalue_val)))
    except Exception:
        return 0.0


# Statistical test significance: p-value inversion (fast counting approximation)
def _compute_significance_from_yields(S: float, B: float) -> float:
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


# Cut-optimization significance: S/sqrt(S+B)
def _optimal_cut_significance(S: float, B: float) -> float:
    s = max(0.0, float(S))
    b = max(0.0, float(B))
    denom = s + b
    if denom <= 0.0:
        return 0.0
    return s / math.sqrt(denom)


# Compute a full set of counting-statistics metrics (p0/Z, CLs)
def _compute_counting_stats(n_obs: Optional[int], n_bkg: float, n_sig: float, cl: float = 0.95):
    n_b = max(0.0, float(n_bkg))
    n_s = max(0.0, float(n_sig))

    # Observed-like line (use n_obs if provided, else use Asimov expected)
    if n_obs is None:
        n_obs_for_observed = max(0, int(round(n_b + n_s)))
    else:
        n_obs_for_observed = max(0, int(n_obs))
    # Observed discovery p0/Z (pure-Python counting)
    p0_obs = _poisson_sf_geq(n_obs_for_observed, n_b)
    z_obs = _significance_from_pvalue(p0_obs)

    # Expected (Asimov) discovery via p-value inversion
    n_exp = max(0, int(round(n_b + n_s)))
    p0_exp = _poisson_sf_geq(n_exp, n_b)
    z_exp = _significance_from_pvalue(p0_exp)

    # Exclusion (CLs)
    n_for_exclusion = n_obs_for_observed
    cls, clspb, clb = _exclusion_summary(n_for_exclusion, n_b, n_s)

    return {
        "p0_obs": p0_obs,
        "z_obs": z_obs,
        "p0_exp": p0_exp,
        "z_exp": z_exp,
        "CLs": cls,
        "CLs+b": clspb,
        "CLb": clb,
    }


# Summary: format expected and observed p0/Z/CLs for display
def _stat_summary(n_bkg: float, n_sig: float, n_obs: Optional[int] = None) -> str:
    n_bkg_val = max(0.0, float(n_bkg))
    n_sig_val = max(0.0, float(n_sig))

    def _line(n: int) -> str:
        n_val = max(0, int(n))
        stats = _compute_counting_stats(n_val, n_bkg_val, n_sig_val)
        p0 = stats.get("p0_obs", float("nan"))
        z = stats.get("z_obs", float("nan"))
        cls = stats.get("CLs", float("nan"))
        clspb = stats.get("CLs+b", float("nan"))
        clb = stats.get("CLb", float("nan"))
        # Report the one-sided discovery significance (non-negative).
        # Negative signs for deficits are intentionally suppressed so the
        # reported Z is always >= 0. This mirrors the behavior of
        # `_significance_from_pvalue` which clips to 0.
        z_str = f"{z:.4g}sigma"
        return (
            f"N={n_val}  p0={p0:.4g}  Z={z_str}  "
            f"CLs={cls:.4g}  CLs+b={clspb:.4g}  CLb={clb:.4g}"
        )

    n_exp = max(0, int(round(n_bkg_val + n_sig_val)))
    result = f"Expected(S+B Asimov): {_line(n_exp)}"
    if n_obs is not None:
        result += f" | Observed: {_line(n_obs)}"
    return result


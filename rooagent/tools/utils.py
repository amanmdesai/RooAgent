from typing import List, Optional
import os
import re
import math
import ROOT


# Module-level counter for unique canvas names shared across tools
_canvas_counter = [0]


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


# Load a TH1 by name from a ROOT file; detach it and return (hist, None) or (None, err).
def _load_histogram(file_path: str, hist_name: str):
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
    return h, None


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
def _parse_background_inputs(
    background_file: Optional[str] = None,
    background_files: Optional[List[str]] = None,
) -> List[str]:
    parsed: List[str] = []

    if background_file:
        parsed.extend([p.strip() for p in background_file.split(",") if p.strip()])

    if background_files:
        parsed.extend([p.strip() for p in background_files if p and p.strip()])

    unique = list(dict.fromkeys(parsed))
    return unique


# Parse primary and additional file inputs into a deduplicated ordered list.
def _parse_file_inputs(
    primary_file: Optional[str] = None,
    additional_files: Optional[List[str]] = None,
) -> List[str]:
    parsed: List[str] = []

    if primary_file:
        parsed.extend([p.strip() for p in primary_file.split(",") if p.strip()])

    if additional_files:
        parsed.extend([p.strip() for p in additional_files if p and p.strip()])

    return list(dict.fromkeys(parsed))


def _to_float_list(v):
    """Convert string/iterable/number input into a list of floats.

    Accepts a comma-separated string, a list/tuple of numbers/strings, or a
    single numeric value. Returns an empty list for None.
    """
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

    columns = [str(c) for c in df.GetColumnNames()]
    return weight_branch if weight_branch in columns else ""


# Load a TH1 and optionally rebin it; returns (hist, None) or (None, error_message).
def _load_hist(file_path: str, hist_name: str, rebin: int):
    h, err = _load_histogram(file_path, hist_name)
    if err:
        return None, err

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


# Compute the Poisson tail P(N >= n_obs) using ROOT when available, else fallback numeric.
def _poisson_tail_ge(n_obs: int, mean: float) -> float:
    n_obs = int(n_obs)
    mean = max(0.0, float(mean))
    if n_obs <= 0:
        return 1.0
    if mean == 0.0:
        return 0.0
    try:
        return float(ROOT.Math.poisson_cdf_c(n_obs - 1, mean))
    except Exception:
        total = 0.0
        for k in range(0, n_obs):
            total += math.exp(k * math.log(mean) - math.lgamma(k + 1) - mean)
        return max(0.0, min(1.0, 1.0 - total))


# Compute the Poisson tail P(N <= n_obs) using ROOT when available, else fallback numeric.
def _poisson_tail_le(n_obs: int, mean: float) -> float:
    n_obs = int(n_obs)
    mean = max(0.0, float(mean))
    if n_obs < 0:
        return 0.0
    if mean == 0.0:
        return 1.0
    try:
        return float(ROOT.Math.poisson_cdf(n_obs, mean))
    except Exception:
        total = 0.0
        for k in range(0, n_obs + 1):
            total += math.exp(k * math.log(mean) - math.lgamma(k + 1) - mean)
        return max(0.0, min(1.0, total))


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
    # p=0 → machine-precision overflow; p=1 → Z=0 by convention
    if not (0.0 < pvalue < 1.0):
        return 0.0
    return max(0.0, float(ROOT.Math.normal_quantile_c(float(pvalue), 1.0)))


# Compute CLs at signal strength `mu` using Poisson tails (CLs+b / CLb).
def _cls_at_mu(n_obs: int, n_bkg: float, n_sig_nominal: float, mu: float) -> float:
    expected_sb = n_bkg + mu * n_sig_nominal
    clb = _poisson_tail_le(n_obs, n_bkg)
    clsplusb = _poisson_tail_le(n_obs, expected_sb)
    if clb <= 0.0:
        return float("inf")
    return clsplusb / clb


# Find the CLs upper limit on mu by bisection to the target confidence level.
def _upper_limit_bisect(
    n_obs: int,
    n_bkg: float,
    n_sig_nominal: float,
    cl: float = 0.95,
    mu_max: float = 20.0,
    n_iter: int = 60,
) -> float:
    target = 1.0 - cl  # e.g. 0.05 for 95% CL

    # CLs is a decreasing function of mu; ensure bracket exists
    if _cls_at_mu(n_obs, n_bkg, n_sig_nominal, 0.0) <= target:
        return 0.0  # already excluded at mu=0 (degenerate case)

    # Expand upper bracket if needed
    while _cls_at_mu(n_obs, n_bkg, n_sig_nominal, mu_max) > target:
        mu_max *= 2.0
        if mu_max > 1e6:
            return float("inf")

    lo, hi = 0.0, mu_max
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if _cls_at_mu(n_obs, n_bkg, n_sig_nominal, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Asymptotic (Cowan) significance for expected S+B vs B yields.
# Valid when B >> ~10 and no background uncertainty is modelled.
# Use for expected sensitivity estimates and discovery reach projections.
def _asymptotic_significance(n_sig: float, n_bkg: float) -> float:
    try:
        s = float(n_sig)
        b = float(n_bkg)
        if b <= 0.0:
            return float("nan")
        # Cowan et al. 2010, Eur.Phys.J. C71 (2011) 1554, eq. 17
        # Z = sqrt(2 * [(s+b) * ln(1 + s/b) - s])
        # Assumes Asimov dataset (n_obs = s+b); valid when B >> ~10.
        val = 2.0 * ((s + b) * math.log(1.0 + (s / b)) - s)
        return math.sqrt(val) if val > 0.0 else 0.0
    except Exception:
        return float("nan")


#Compute discovery significance from expected signal S and background B yields.
def _compute_significance_from_yields(S: float, B: float, method: str = "simple") -> float:

    s = float(S)
    b = float(B)
    if method == "asymptotic":
        return _asymptotic_significance(s, b)
    # default: "simple"  — S / sqrt(B)
    if b <= 0.0:
        return float("nan")
    return s / math.sqrt(b)


# Return compact p0 / Z / CLs summary string: always expected (S+B Asimov), observed if n_obs given.
def _stat_summary(n_bkg: float, n_sig: float, n_obs: Optional[int] = None) -> str:
    def _line(n: int) -> str:
        p0   = _poisson_tail_ge(n, n_bkg)
        clb  = _poisson_tail_le(n, n_bkg)
        clspb = _poisson_tail_le(n, n_bkg + n_sig)
        cls  = clspb / clb if clb > 0.0 else float("inf")
        z    = _significance_from_pvalue(p0)
        return f"N={n}  p0={p0:.4g}  Z={z:.4g}\u03c3  CLs={cls:.4g}"

    n_exp = max(0, int(round(n_bkg + n_sig)))
    result = f"Expected(S+B Asimov): {_line(n_exp)}"
    if n_obs is not None:
        result += f" | Observed: {_line(n_obs)}"
    return result


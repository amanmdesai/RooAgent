from typing import List, Optional
import os
import re
import array as _arr
import math
import ROOT


# Module-level counter for unique canvas names shared across tools
_canvas_counter = [0]


###############################
# tfile_tools.py helpers
###############################


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


def _get_root_files(directory: str = ".") -> List[str]:
    try:
        return [f for f in os.listdir(directory) if f.lower().endswith(".root")]
    except Exception:
        return []


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


def _maybe_rebin_hist(hist, rebin: int):
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
# plot_tools.py helpers
# (also used by fit_tools.py and rdataframe_tools.py)
###############################


def _unique_canvas_name(base: str) -> str:
    _canvas_counter[0] += 1
    return f"{base}_{_canvas_counter[0]}"


###############################
# stat_tools.py helpers
###############################


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


def _significance_from_pvalue(pvalue: float, n_obs: int, mean: float) -> float:
    try:
        if 0.0 < pvalue < 1.0:
            return float(ROOT.Math.normal_quantile_c(float(pvalue), 1.0))
    except Exception:
        pass

    try:
        if mean > 0.0:
            return (float(n_obs) - float(mean)) / math.sqrt(float(mean))
    except Exception:
        pass

    return float("nan")


def _cls_at_mu(n_obs: int, n_bkg: float, n_sig_nominal: float, mu: float) -> float:
    expected_sb = n_bkg + mu * n_sig_nominal
    clb = _poisson_tail_le(n_obs, n_bkg)
    clsplusb = _poisson_tail_le(n_obs, expected_sb)
    if clb <= 0.0:
        return float("inf")
    return clsplusb / clb


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


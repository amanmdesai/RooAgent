from typing import List, Optional

import ROOT

from .utils import (
    _prepare_roostats_model,
    _build_sb_model_configs,
    _run_cls_scan,
    _at_poi_boundary,
    _POI_EXPAND_FACTOR,
    _POI_MAX_EXPANSIONS,
)


def compute_discovery_significance(
    source: str,
    signal_shape: str,
    background_shape: str,
    file_path: str,
    tree_name: str = "",
    variable: str = "",
    hist_name: str = "",
    xmin: float = -5.0,
    xmax: float = 5.0,
    mean: float = 0.0,
    mean_range: Optional[List[float]] = None,
    sigma: float = 1.0,
    sigma_range: Optional[List[float]] = None,
    alpha: float = 1.5,
    alpha_range: Optional[List[float]] = None,
    n: float = 2.0,
    n_range: Optional[List[float]] = None,
    width: float = 0.1,
    width_range: Optional[List[float]] = None,
    tau: float = -0.1,
    tau_range: Optional[List[float]] = None,
    poly_coeffs: Optional[List[float]] = None,
    poly_coeff_range: float = 1.0,
    nsig_init: float = 100.0,
    nsig_range: Optional[List[float]] = None,
    nbkg_init: float = 100.0,
    nbkg_range: Optional[List[float]] = None,
) -> str:
    """Compute asymptotic profile-likelihood discovery significance (RooStats).

    Args:
        source: 'tree' (unbinned) or 'hist' (binned).
        signal_shape: 'gauss' | 'crystalball' | 'voigt'.
        background_shape: 'expo' | 'chebychev' | 'poly' (required).
        file_path: ROOT file path.
        tree_name / variable: Tree + branch (source='tree').
        hist_name: Histogram name (source='hist').
        xmin / xmax: Fit range (source='tree' only).
        mean, sigma, alpha, n, width, tau, poly_coeffs (+ *_range): shape params.
        nsig_init / nsig_range / nbkg_init / nbkg_range: signal/background yields.

    Returns: One-line summary with p0/Z, or error string.
    """
    try:
        bundle, err = _prepare_roostats_model(
            source, signal_shape, background_shape, file_path, tree_name, variable,
            hist_name, xmin, xmax, mean, mean_range, sigma, sigma_range, alpha,
            alpha_range, n, n_range, width, width_range, tau, tau_range, poly_coeffs,
            poly_coeff_range, nsig_init, nsig_range, nbkg_init, nbkg_range,
        )
        if err:
            return err

        ws_info, err = _build_sb_model_configs(bundle["x"], bundle["data"], bundle["model"], bundle["nsig"])
        if err:
            return err

        w = ws_info["w"]
        ac = ROOT.RooStats.AsymptoticCalculator(w.data(ws_info["data_name"]), ws_info["mc_sb"], ws_info["mc_b"])
        ac.SetOneSidedDiscovery(True)
        result = ac.GetHypoTest()

        p0 = result.NullPValue()
        z = result.Significance()

        return (
            f"RooStats discovery [{bundle['shapes']}] on {bundle['label']} "
            f"(N={bundle['n_events']:.4g}): p0={p0:.4g} Z={z:.4g}sigma "
            f"(nsig_init={nsig_init:.4g})"
        )
    except Exception as exc:
        return f"Error: {exc}."


def compute_upper_limit(
    source: str,
    signal_shape: str,
    background_shape: str,
    file_path: str,
    tree_name: str = "",
    variable: str = "",
    hist_name: str = "",
    xmin: float = -5.0,
    xmax: float = 5.0,
    mean: float = 0.0,
    mean_range: Optional[List[float]] = None,
    sigma: float = 1.0,
    sigma_range: Optional[List[float]] = None,
    alpha: float = 1.5,
    alpha_range: Optional[List[float]] = None,
    n: float = 2.0,
    n_range: Optional[List[float]] = None,
    width: float = 0.1,
    width_range: Optional[List[float]] = None,
    tau: float = -0.1,
    tau_range: Optional[List[float]] = None,
    poly_coeffs: Optional[List[float]] = None,
    poly_coeff_range: float = 1.0,
    nsig_init: float = 100.0,
    nsig_range: Optional[List[float]] = None,
    nbkg_init: float = 100.0,
    nbkg_range: Optional[List[float]] = None,
    confidence_level: float = 0.95,
    scan_points: Optional[int] = None,
    poi_min: Optional[float] = None,
    poi_max: Optional[float] = None,
) -> str:
    """Compute an asymptotic CLs upper limit on the signal yield (RooStats).

    Args:
        source: 'tree' (unbinned) or 'hist' (binned).
        signal_shape: 'gauss' | 'crystalball' | 'voigt'.
        background_shape: 'expo' | 'chebychev' | 'poly' (required).
        file_path: ROOT file path.
        tree_name / variable: Tree + branch (source='tree').
        hist_name: Histogram name (source='hist').
        xmin / xmax: Fit range (source='tree' only).
        mean, sigma, alpha, n, width, tau, poly_coeffs (+ *_range): shape params.
        nsig_init / nsig_range / nbkg_init / nbkg_range: signal/background yields
            (nsig is the POI).
        confidence_level: CLs confidence level (default 0.95).
        scan_points: Fixed scan instead of adaptive.
        poi_min / poi_max: POI scan bounds (used with scan_points).

    Returns: One-line summary with observed/expected UL and bands, or error string.
    """
    try:
        if not (0.0 < confidence_level < 1.0):
            return "Error: confidence_level must be between 0 and 1."

        bundle, err = _prepare_roostats_model(
            source, signal_shape, background_shape, file_path, tree_name, variable,
            hist_name, xmin, xmax, mean, mean_range, sigma, sigma_range, alpha,
            alpha_range, n, n_range, width, width_range, tau, tau_range, poly_coeffs,
            poly_coeff_range, nsig_init, nsig_range, nbkg_init, nbkg_range,
        )
        if err:
            return err

        ws_info, err = _build_sb_model_configs(bundle["x"], bundle["data"], bundle["model"], bundle["nsig"])
        if err:
            return err

        w = ws_info["w"]
        w_nsig = w.var(ws_info["nsig_name"])
        # Only auto-expand if the caller left the range up to us.
        auto_expand = nsig_range is None and scan_points is None

        obs_ul, med, p1, m1, p2, m2 = _run_cls_scan(w, ws_info, confidence_level, scan_points, poi_min, poi_max)

        expansions = 0
        while auto_expand and expansions < _POI_MAX_EXPANSIONS and _at_poi_boundary((obs_ul, med, p1, p2), w_nsig.getMax()):
            w_nsig.setMax(w_nsig.getMax() * _POI_EXPAND_FACTOR)
            obs_ul, med, p1, m1, p2, m2 = _run_cls_scan(w, ws_info, confidence_level, scan_points, poi_min, poi_max)
            expansions += 1

        final_hi = w_nsig.getMax()
        if _at_poi_boundary((obs_ul, med, p1, p2), final_hi):
            note = (
                f" | WARNING: result reached the nsig scan boundary ({final_hi:.4g}) -- "
                "the true limit may be higher than reported; pass a larger nsig_range to confirm."
            )
        elif expansions:
            note = f" | (nsig scan range auto-expanded to [0, {final_hi:.4g}] to clear the boundary)"
        else:
            note = ""

        return (
            f"RooStats CLs upper limit [{bundle['shapes']}] on {bundle['label']} "
            f"(N={bundle['n_events']:.4g}, CL={confidence_level * 100:.4g}%): "
            f"Observed UL(nsig)={obs_ul:.4g} | Expected UL(median)={med:.4g} "
            f"[+1sigma={p1:.4g}, -1sigma={m1:.4g}, +2sigma={p2:.4g}, -2sigma={m2:.4g}]{note}"
        )
    except Exception as exc:
        return f"Error: {exc}."

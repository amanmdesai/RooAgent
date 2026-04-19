import array as _arr
from typing import Optional
import ROOT
from langchain_core.tools import tool

from .histogram_tools import _load_histogram, _maybe_rebin_hist


# ─── Private RooStats helpers ──────────────────────────────────────────────────

def _silence_roo():
    """Suppress RooFit/RooStats INFO and PROGRESS messages."""
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)


def _roostat_obs_significance(
    n_obs: float, b_exp: float, rel_unc: float = 0.0
) -> tuple:
    """Observed discovery significance and p0 via RooStats::NumberCountingUtils.

    Parameters
    ----------
    rel_unc : fractional background uncertainty (0 = pure Poisson profile-likelihood).

    Returns
    -------
    (Z, p0) — significance in units of sigma and one-sided local p-value.
    """
    if n_obs <= b_exp:
        return 0.0, 1.0
    z = float(ROOT.RooStats.NumberCountingUtils.BinomialObsZ(n_obs, b_exp, rel_unc))
    p = float(ROOT.RooStats.NumberCountingUtils.BinomialObsP(n_obs, b_exp, rel_unc))
    return max(0.0, z), min(max(p, 0.0), 1.0)


def _roostat_exp_significance(
    s_exp: float, b_exp: float, rel_unc: float = 0.0
) -> float:
    """Expected (median) discovery significance via RooStats::NumberCountingUtils."""
    if s_exp <= 0 or b_exp <= 0:
        return 0.0
    z = float(ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s_exp, b_exp, rel_unc))
    return max(0.0, z)


def _build_poisson_workspace(S: float, B: float, N_obs: float, mu_max: float):
    """Build a minimal Poisson counting workspace for RooStats.

    Model: n ~ Poisson(mu*S + B)  with mu as the POI.

    Returns
    -------
    (workspace, sb_ModelConfig, bonly_ModelConfig, RooDataSet)
    """
    _silence_roo()

    w = ROOT.RooWorkspace("w", True)

    w.factory(f"mu[1,0,{mu_max:.6g}]")
    w.factory(f"expr::n_exp('mu*{S:.10g}+{B:.10g}', mu)")

    n_hi = max(int(N_obs * 5 + 20), int(B * 5 + 20))
    w.factory(f"Poisson::model(n[{N_obs:.6g},0,{n_hi}], n_exp)")

    # S+B ModelConfig (mu = 1 snapshot)
    sb_model = ROOT.RooStats.ModelConfig("sb_model", w)
    sb_model.SetPdf(w.pdf("model"))
    sb_model.SetParametersOfInterest(w.var("mu"))
    sb_model.SetObservables(w.var("n"))
    w.var("mu").setVal(1.0)
    sb_model.SetSnapshot(ROOT.RooArgSet(w.var("mu")))

    # Background-only ModelConfig (mu = 0 snapshot)
    bonly_model = sb_model.Clone("bonly_model")
    w.var("mu").setVal(0.0)
    bonly_model.SetSnapshot(ROOT.RooArgSet(w.var("mu")))

    # Observed dataset
    w.var("n").setVal(N_obs)
    obs_data = ROOT.RooDataSet("obs_data", "obs_data", ROOT.RooArgSet(w.var("n")))
    obs_data.add(ROOT.RooArgSet(w.var("n")))

    return w, sb_model, bonly_model, obs_data


def _roostat_cls_upper_limit(
    S: float, B: float, N_obs: float, cl: float = 0.95, n_scan: int = 30
) -> dict:
    """Asymptotic CLs upper limit on signal strength µ via RooStats.

    Uses RooStats::AsymptoticCalculator + HypoTestInverter with a simple
    Poisson counting workspace: n ~ Poisson(mu*S + B).

    Returns
    -------
    dict with keys: obs, exp, p1s, m1s, p2s, m2s  (all µ upper limits).
    """
    inf = float("inf")
    if S <= 0:
        return dict(obs=inf, exp=inf, p1s=inf, m1s=inf, p2s=inf, m2s=inf)

    mu_hat = max(0.0, (N_obs - B) / S)
    mu_max = max(20.0, mu_hat * 5 + 5)

    _silence_roo()

    w, sb_model, bonly_model, obs_data = _build_poisson_workspace(S, B, N_obs, mu_max)

    calc = ROOT.RooStats.AsymptoticCalculator(obs_data, sb_model, bonly_model)
    calc.SetOneSided(True)
    calc.SetPrintLevel(-1)

    inverter = ROOT.RooStats.HypoTestInverter(calc)
    inverter.SetConfidenceLevel(cl)
    inverter.UseCLs(True)
    inverter.SetVerbose(False)
    inverter.SetFixedScan(n_scan, 0.0, mu_max)

    result = inverter.GetInterval()

    return dict(
        obs=float(result.UpperLimit()),
        exp=float(result.GetExpectedUpperLimit(0)),
        p1s=float(result.GetExpectedUpperLimit(1)),
        m1s=float(result.GetExpectedUpperLimit(-1)),
        p2s=float(result.GetExpectedUpperLimit(2)),
        m2s=float(result.GetExpectedUpperLimit(-2)),
    )


# ─── Public tools ─────────────────────────────────────────────────────────────

@tool
def histogram_significance_and_limits(
    signal_file: str,
    signal_hist_name: str,
    background_file: str,
    background_hist_name: str,
    data_file: Optional[str] = None,
    data_hist_name: Optional[str] = None,
    x_low: Optional[float] = None,
    x_high: Optional[float] = None,
    confidence_level: float = 0.95,
    background_uncertainty_fraction: float = 0.0,
    rebin: int = 1,
) -> str:
    """Full statistical analysis on signal/background (and optionally data) histograms
    using ROOT RooStats.

    Use this tool when inputs are **pre-built TH1 histograms** already stored in ROOT
    files.  For tree-level event selection use `compute_significance` instead.

    Significance and p-value are computed with RooStats::NumberCountingUtils
    (BinomialObsZ / BinomialObsP / BinomialExpZ).  CLs upper limits on the
    signal-strength µ and the ±1σ/±2σ expected bands are derived with
    RooStats::AsymptoticCalculator and HypoTestInverter on a Poisson counting
    workspace: n ~ Poisson(mu*S + B).

    Parameters
    ----------
    data_file, data_hist_name : optional
        Observed data histogram.  When omitted the observed count N is set to
        S + B (the Asimov / expected-data assumption), which gives purely
        expected sensitivity numbers (no observed excess).
    background_uncertainty_fraction : float
        Fractional systematic uncertainty on the background (e.g. 0.1 = 10 %).
        Propagated to NumberCountingUtils significance; the CLs calculation
        uses a stat-only Poisson model.
    """
    h_sig, e = _load_histogram(signal_file, signal_hist_name)
    if e:
        return e
    h_bkg, e = _load_histogram(background_file, background_hist_name)
    if e:
        return e

    h_sig  = _maybe_rebin_hist(h_sig,  rebin)
    h_bkg  = _maybe_rebin_hist(h_bkg,  rebin)

    # ── Optional data histogram ───────────────────────────────────────────
    h_data = None
    if data_file and data_hist_name:
        h_data, e = _load_histogram(data_file, data_hist_name)
        if e:
            return e
        h_data = _maybe_rebin_hist(h_data, rebin)

    # ── Restrict axis range ───────────────────────────────────────────────
    warn_range = ""
    if x_low is not None and x_high is not None:
        if x_low >= x_high:
            return f"Error: x_low ({x_low}) must be less than x_high ({x_high})."
        hists_to_range = [h_sig, h_bkg] + ([h_data] if h_data else [])
        for h in hists_to_range:
            ax = h.GetXaxis()
            alo, ahi = ax.GetXmin(), ax.GetXmax()
            if x_low >= ahi or x_high <= alo:
                return (f"Error: range [{x_low:.5g},{x_high:.5g}] outside "
                        f"axis [{alo:.5g},{ahi:.5g}] for '{h.GetName()}'.")
            cl_v = max(x_low, alo)
            ch_v = min(x_high, ahi)
            if cl_v != x_low or ch_v != x_high:
                warn_range = f" [range clipped to [{cl_v:.5g},{ch_v:.5g}]]"
            h.GetXaxis().SetRangeUser(cl_v, ch_v)

    # ── Integrated yields ─────────────────────────────────────────────────
    e_s = _arr.array('d', [0.0])
    e_b = _arr.array('d', [0.0])
    S = h_sig.IntegralAndError(
        h_sig.GetXaxis().GetFirst(), h_sig.GetXaxis().GetLast(), e_s)
    B = h_bkg.IntegralAndError(
        h_bkg.GetXaxis().GetFirst(), h_bkg.GetXaxis().GetLast(), e_b)

    if h_data is not None:
        e_d = _arr.array('d', [0.0])
        N = h_data.IntegralAndError(
            h_data.GetXaxis().GetFirst(), h_data.GetXaxis().GetLast(), e_d)
        n_err_str = f"±{e_d[0]:.3g}"
        data_note = ""
    else:
        N = S + B           # Asimov: expected data = S + B
        n_err_str = "(S+B)"
        data_note = " [no data: N=S+B assumed]"

    # ── RooStats significance (NumberCountingUtils) ───────────────────────
    rel_unc = background_uncertainty_fraction
    z_obs, p0_obs = _roostat_obs_significance(N, B, rel_unc)
    z_exp          = _roostat_exp_significance(S, B, rel_unc)

    # ── RooStats CLs upper limits (AsymptoticCalculator) ─────────────────
    lim = _roostat_cls_upper_limit(S, B, N, cl=confidence_level)

    # ── Format output ─────────────────────────────────────────────────────
    range_str = (
        f"x ∈ [{x_low:.5g}, {x_high:.5g}]"
        if x_low is not None and x_high is not None
        else "full range"
    )
    syst_str = f" (bkg_unc={rel_unc*100:.0f}%)" if rel_unc > 0 else ""
    return (
        f"StatAnalysis {range_str}{syst_str}{data_note} | "
        f"N={N:.4g}{n_err_str}  S={S:.4g}±{e_s[0]:.3g}  B={B:.4g}±{e_b[0]:.3g} | "
        f"RooStats [NumberCountingUtils]: "
        f"obs_Z={z_obs:.3g}σ  exp_Z={z_exp:.3g}σ  p0={p0_obs:.3e} | "
        f"CLs{confidence_level*100:.0f}% [AsymptoticCalc+HypoTestInverter]: "
        f"obs<{lim['obs']:.4g}  exp<{lim['exp']:.4g}  "
        f"[+1σ<{lim['p1s']:.4g}  -1σ<{lim['m1s']:.4g}  "
        f"+2σ<{lim['p2s']:.4g}  -2σ<{lim['m2s']:.4g}]"
        f"{warn_range}"
    )

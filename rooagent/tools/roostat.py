import math
import ROOT
from langchain_core.tools import tool
from .utils import _open_root_file, _get_hist


def _poisson_tail_ge(n_obs: int, mean: float) -> float:
    if n_obs <= 0:
        return 1.0
    try:
        return float(ROOT.Math.poisson_cdf_c(int(n_obs) - 1, float(mean)))
    except Exception:
        s = 0.0
        for k in range(0, int(n_obs)):
            s += math.exp(k * math.log(mean) - math.lgamma(k + 1) - mean)
        return max(0.0, 1.0 - s)


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



@tool
def histogram_significance_and_limits(
    file_path: str,
    data_name: str = "",
    bkg_name: str = "bkg",
    sig_name: str = "sig",
    center: float = 50.0,
    window: float = 4.0,
    rebin: int = 1,
) -> str:
    """Compute counting p-values and a CLs-like summary from TH1 histograms.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing histograms.
    data_name : str
        Name of the data histogram. If empty, expected sensitivity is computed
        assuming N = S + B.
    bkg_name, sig_name : str
        Names of the background and signal histograms in the file.
    center, window : float
        Center and half-width (same units as histogram x-axis) of the counting
        window.
    rebin : int
        Optional rebin factor to apply before integration.

    Returns
    -------
    str
        One-line summary with p-values and CLs/Significance when available.
    """
    if not file_path:
        return "Error: file_path is required."

    f = _open_root_file(file_path)
    if not f:
        return f"Error: cannot open ROOT file '{file_path}'"

    hbkg = _get_hist(f, bkg_name)
    hsig = _get_hist(f, sig_name)
    hdata = _get_hist(f, data_name) if data_name else None

    missing = []
    if hbkg is None:
        missing.append(bkg_name)
    if hsig is None:
        missing.append(sig_name)
    if missing:
        f.Close()
        return f"Error: missing histograms: {', '.join(missing)} in {file_path}"

    # Optional rebinning (safe best-effort)
    try:
        r = int(rebin) if rebin is not None else 1
    except Exception:
        r = 1
    if r > 1:
        try:
            hbkg = hbkg.Rebin(r, f"{hbkg.GetName()}_rebin{r}")
            hsig = hsig.Rebin(r, f"{hsig.GetName()}_rebin{r}")
            if hdata:
                hdata = hdata.Rebin(r, f"{hdata.GetName()}_rebin{r}")
        except Exception:
            pass

    n_bkg_frac = _fractional_integral(hbkg, center, window)
    n_sig_frac = _fractional_integral(hsig, center, window)
    if hdata:
        n_data_frac = _fractional_integral(hdata, center, window)
        n_obs = int(round(n_data_frac))
    else:
        # expected sensitivity: assume observed = S + B
        n_obs = int(round(n_bkg_frac + n_sig_frac))

    p_b = _poisson_tail_ge(n_obs, n_bkg_frac)
    p_sb = _poisson_tail_ge(n_obs, n_bkg_frac + n_sig_frac)

    lines = []
    try:
        res = ROOT.RooStats.HypoTestResult("res", float(p_b), float(p_sb))
        lines.append(f"NullPValue: {res.NullPValue():.6g}")
        lines.append(f"Significance: {res.Significance():.6g}")
        lines.append(f"CLs: {res.CLs():.6g}")
        lines.append(f"CLs+b: {res.CLsplusb():.6g}")
        lines.append(f"CLb: {res.CLb():.6g}")
    except Exception:
        # graceful fallback: return p-values and a simple Gaussian approx for Z
        try:
            if n_bkg_frac > 0:
                z_approx = (n_obs - n_bkg_frac) / math.sqrt(n_bkg_frac)
            else:
                z_approx = float("nan")
        except Exception:
            z_approx = float("nan")
        lines.append(f"p_b={p_b:.6g}")
        lines.append(f"p_sb={p_sb:.6g}")
        if not math.isnan(z_approx):
            lines.append(f"approx_significance: {z_approx:.6g}")
        else:
            lines.append("approx_significance: n/a")

    f.Close()
    summary = " | ".join(lines)
    return summary


# CLI compatibility
def main():
    import argparse

    p = argparse.ArgumentParser(description="RooAgent minimal RooStats helper")
    p.add_argument("rootfile", nargs="?", default="toy.root")
    p.add_argument("--data", default="")
    p.add_argument("--bkg", default="bkg")
    p.add_argument("--sig", default="sig")
    p.add_argument("--center", type=float, default=50.0)
    p.add_argument("--window", type=float, default=4.0)
    p.add_argument("--rebin", type=int, default=1)
    args = p.parse_args()

    out = histogram_significance_and_limits(
        file_path=args.rootfile,
        data_name=args.data,
        bkg_name=args.bkg,
        sig_name=args.sig,
        center=args.center,
        window=args.window,
        rebin=args.rebin,
    )
    print(out)


if __name__ == "__main__":
    main()

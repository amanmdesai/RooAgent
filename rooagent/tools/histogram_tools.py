import array as _arr
import ROOT
from .utils import _load_hist


# ─── Public tools ─────────────────────────────────────────────────────────────

def get_histogram_stats(file_path: str, hist_name: str, rebin: int = 1) -> str:
    """Return mean, RMS, and entry count of a TH1 from a ROOT file.

    Args:
        file_path: Path to ROOT file.
        hist_name: Histogram name inside the file.
        rebin: Merge bins before computing stats (default 1 = no rebin).

    Returns: "<name> -> Mean: <v>, RMS: <v>, Entries: <N>" or error string.
    """
    h, err = _load_hist(file_path, hist_name, rebin)
    if err:
        return err

    mean = h.GetMean()
    rms = h.GetRMS()
    entries = int(h.GetEntries())
    return f"{hist_name} -> Mean: {mean:.3f}, RMS: {rms:.3f}, Entries: {entries}"


def histogram_integral(
    file_path: str,
    hist_name: str,
    x_low: float,
    x_high: float,
    include_overflow: bool = False,
    rebin: int = 1,
) -> str:
    """Integrate a TH1 over [x_low, x_high) and return value±error.

    Args:
        file_path: Path to ROOT file.
        hist_name: Histogram name.
        x_low: Lower edge of integration range.
        x_high: Upper edge (exclusive; clipped to axis bounds with warning).
        include_overflow: Include under/overflow bins (default False).
        rebin: Merge bins before integrating (default 1).

    Returns: "Integral '<name>' x=[lo,hi] bins=[lo,hi]: <val>±<err>" or error string.
    """
    h, err = _load_hist(file_path, hist_name, rebin)
    if err:
        return err

    nb = h.GetNbinsX()
    ax = h.GetXaxis()
    axis_lo = ax.GetXmin()
    axis_hi = ax.GetXmax()

    # ── Validate requested range ───────────────────────────────────────────
    if x_low >= x_high:
        return f"Error: x_low ({x_low}) must be less than x_high ({x_high})."
    if x_low >= axis_hi or x_high <= axis_lo:
        return (f"Error: requested range [{x_low:.5g}, {x_high:.5g}] is entirely "
                f"outside histogram axis [{axis_lo:.5g}, {axis_hi:.5g}].")

    # Warn (but continue) when the range is only partially inside
    warn = ""
    if x_low < axis_lo:
        warn = f" [warn: x_low clipped to axis min {axis_lo:.5g}]"
        x_low = axis_lo
    if x_high > axis_hi:
        warn += f" [warn: x_high clipped to axis max {axis_hi:.5g}]"
        x_high = axis_hi

    bin_lo = ax.FindBin(x_low)
    bin_hi = ax.FindBin(x_high)

    # If x_high falls exactly on a bin's lower edge, exclude that bin so the
    # integral is half-open [x_low, x_high).  Use a relative-epsilon comparison
    # to handle floating-point imprecision in user-supplied bounds.
    _eps = 1e-9 * (axis_hi - axis_lo)
    if abs(ax.GetBinLowEdge(bin_hi) - x_high) < _eps and bin_hi > 1:
        bin_hi -= 1

    lo_clamp = 0 if include_overflow else 1
    hi_clamp = nb + 1 if include_overflow else nb
    bin_lo = max(bin_lo, lo_clamp)
    bin_hi = min(bin_hi, hi_clamp)

    if bin_lo > bin_hi:
        return f"Error: effective bin range is empty after clamping (bin_lo={bin_lo} > bin_hi={bin_hi})."
    e_ref = _arr.array('d', [0.0])
    integral = h.IntegralAndError(bin_lo, bin_hi, e_ref)
    error_val = e_ref[0]

    x_actual_lo = ax.GetBinLowEdge(bin_lo)
    x_actual_hi = ax.GetBinUpEdge(bin_hi)

    return (f"Integral '{hist_name}' x=[{x_actual_lo:.5g},{x_actual_hi:.5g}]"
            f" bins=[{bin_lo},{bin_hi}]: {integral:.6g}+-{error_val:.6g}{warn}")
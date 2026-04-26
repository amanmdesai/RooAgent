import array as _arr
import ROOT
from langchain_core.tools import tool
from .utils import _load_hist


# ─── Public tools ─────────────────────────────────────────────────────────────

@tool
def get_histogram_stats(file_path: str, hist_name: str, rebin: int = 1) -> str:
    """Extract basic descriptive statistics from a 1D histogram stored in a ROOT file.

    Reads the named TH1 histogram from `file_path`, optionally rebins it, and returns the
    mean, RMS (standard deviation) and total number of entries. This is a quick sanity check
    to verify that a histogram was filled correctly before running integrals or fits.

    Args:
        file_path (str): Path to the ROOT file containing the histogram.
        hist_name (str): Name of the TH1 histogram to query.
        rebin (int, optional): Merge every `rebin` consecutive bins before computing statistics
            (default 1, no rebinning). Useful for smoothing noisy distributions.

    Returns:
        str: A formatted string like "<hist_name> -> Mean: <m>, RMS: <r>, Entries: <n>" or
            an error message if the file or histogram cannot be found.
    """
    h, err = _load_hist(file_path, hist_name, rebin)
    if err:
        return err

    mean = h.GetMean()
    rms = h.GetRMS()
    entries = int(h.GetEntries())
    return f"{hist_name} -> Mean: {mean:.3f}, RMS: {rms:.3f}, Entries: {entries}"


@tool
def histogram_integral(
    file_path: str,
    hist_name: str,
    x_low: float,
    x_high: float,
    include_overflow: bool = False,
    rebin: int = 1,
) -> str:
    """Integrate a 1D histogram over a specified x-range and return the integral with uncertainty.

    The integral is computed over the half-open interval [x_low, x_high) by summing bin contents
    (weighted by the fractional overlap when boundaries fall inside a bin). This is useful for
    computing event yields in a signal window or sideband region from a stored histogram.

    Args:
        file_path (str): Path to the ROOT file containing the histogram.
        hist_name (str): Name of the TH1 histogram to integrate.
        x_low (float): Lower integration bound (inclusive).
        x_high (float): Upper integration bound (exclusive). Must be strictly greater than `x_low`.
        include_overflow (bool, optional): Include under/overflow bins in the sum (default False).
        rebin (int, optional): Merge every `rebin` bins before integrating (default 1).

    Returns:
        str: Result in the form
            "Integral '<name>' x=[lo,hi] bins=[b1,b2]: <value>+-<error>"
        Returns a descriptive error string when x_low ≥ x_high, the range lies entirely outside
        the histogram axis, or the file/histogram cannot be opened.

    Notes:
        - x boundaries are clamped to the histogram axis range with a warning appended to the output.
        - Use histogram_significance_and_cls instead when you need significance or CLs from a window,
          as it handles fractional bin contributions automatically.
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
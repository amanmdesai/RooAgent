from typing import List, Optional

import ROOT

from .utils import (
    _unique_canvas_name,
    _build_roofit_dataset,
    _build_signal_pdf,
    _build_background_pdf,
    _build_extended_model,
    _SIGNAL_SHAPES,
    _BACKGROUND_SHAPES,
)


def fit_model(
    source: str,
    signal_shape: str,
    output_plot: str,
    file_path: str,
    tree_name: str = "",
    variable: str = "",
    hist_name: str = "",
    bins: int = 50,
    xmin: float = -5.0,
    xmax: float = 5.0,
    background_shape: str = "",
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
    """Fit a RooFit signal(+background) model via max likelihood.

    Args:
        source: 'tree' (unbinned) or 'hist' (binned).
        signal_shape: 'gauss' | 'crystalball' | 'voigt'.
        output_plot: Output plot path.
        file_path: ROOT file path.
        tree_name / variable: Tree + branch (source='tree').
        hist_name: Histogram name (source='hist').
        bins: Display bins (source='tree' only).
        xmin / xmax: Fit range (source='tree' only).
        background_shape: '' | 'expo' | 'chebychev' | 'poly'.
        mean, sigma, alpha, n, width, tau, poly_coeffs (+ *_range): shape params.
        nsig_init / nsig_range / nbkg_init / nbkg_range: signal/background yields.

    Returns: One-line fit summary or error string.
    """
    try:
        source_key = (source or "").strip().lower()
        shape_key = (signal_shape or "").strip().lower()
        bkg_key = (background_shape or "").strip().lower()

        if source_key not in {"tree", "hist"}:
            return "Error: source must be 'tree' or 'hist'."
        if shape_key not in _SIGNAL_SHAPES:
            return f"Error: signal_shape must be one of {sorted(_SIGNAL_SHAPES)}."
        if bkg_key not in _BACKGROUND_SHAPES:
            return f"Error: background_shape must be one of {sorted(s for s in _BACKGROUND_SHAPES if s)} or ''."

        ds_info, err = _build_roofit_dataset(source_key, file_path, tree_name, variable, hist_name, xmin, xmax)
        if err:
            return err
        x = ds_info["x"]
        data = ds_info["data"]
        n_events = ds_info["n_events"]
        label = ds_info["label"]
        xmin, xmax = ds_info["xmin"], ds_info["xmax"]

        sig_info, err = _build_signal_pdf(
            x, shape_key, mean, mean_range, sigma, sigma_range,
            alpha, alpha_range, n, n_range, width, width_range, xmin, xmax,
        )
        if err:
            return err
        sig_pdf = sig_info["pdf"]

        bkg_info, err = _build_background_pdf(x, bkg_key, tau, tau_range, poly_coeffs, poly_coeff_range)
        if err:
            return err
        bkg_pdf = bkg_info["pdf"] if bkg_info else None

        if bkg_pdf is None:
            model = sig_pdf
            res = model.fitTo(data, ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1))
        else:
            ext_info, err = _build_extended_model(
                sig_pdf, bkg_pdf, n_events, nsig_init, nsig_range, nbkg_init, nbkg_range
            )
            if err:
                return err
            model = ext_info["model"]
            res = model.fitTo(
                data,
                ROOT.RooFit.Save(True),
                ROOT.RooFit.PrintLevel(-1),
                ROOT.RooFit.Extended(True),
            )

        canvas = ROOT.TCanvas(_unique_canvas_name("c_roofit"), "", 900, 700)
        frame = x.frame()
        if source_key == "tree":
            data.plotOn(frame, ROOT.RooFit.Binning(bins))
        else:
            data.plotOn(frame)
        model.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue + 1))
        if bkg_pdf is not None:
            model.plotOn(
                frame,
                ROOT.RooFit.Components(bkg_pdf.GetName()),
                ROOT.RooFit.LineStyle(ROOT.kDashed),
                ROOT.RooFit.LineColor(ROOT.kRed),
            )
        frame.GetXaxis().SetTitle(label)
        frame.Draw()
        canvas.Update()
        canvas.SaveAs(output_plot)

        shapes = f"{shape_key}+{bkg_key}" if bkg_key else shape_key
        params = ", ".join(
            f"{p.GetName()}={p.getVal():.4g}+/-{p.getError():.4g}"
            for p in res.floatParsFinal()
        )
        return (
            f"RooFit ML fit [{shapes}] on {label} (N={n_events:.4g}): {params} "
            f"| status={res.status()} edm={res.edm():.3g} covQual={res.covQual()} -> {output_plot}"
        )
    except Exception as exc:
        return f"Error: {exc}."

"""Shared tool registry used by both the MCP server and the LangGraph agent.

Owns the ROOT environment bootstrap, the canonical list of exposed tool
functions, and the workflow-instructions prose, so rooagent.py and
cli.py cannot drift apart on either front.
"""

import os as _os
import sys as _sys

# Bootstrap ROOT (must match this interpreter's Python version) before any ROOT-dependent imports.
_ROOTSYS = _os.environ.get("ROOTSYS")
if _ROOTSYS:
    _root_lib_candidates = [
        _os.path.join(_ROOTSYS, "lib", "root"),
        _os.path.join(_ROOTSYS, "lib"),
    ]
    _root_bin = _os.path.join(_ROOTSYS, "bin")

    for _p in _root_lib_candidates:
        if _os.path.isdir(_p) and _p not in _sys.path:
            _sys.path.insert(0, _p)

    _ld_parts = [p for p in _os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    for _p in reversed(_root_lib_candidates):
        if _os.path.isdir(_p) and _p not in _ld_parts:
            _ld_parts.insert(0, _p)
    if _ld_parts:
        _os.environ["LD_LIBRARY_PATH"] = ":".join(_ld_parts)

    _path_parts = [p for p in _os.environ.get("PATH", "").split(":") if p]
    if _os.path.isdir(_root_bin) and _root_bin not in _path_parts:
        _os.environ["PATH"] = _root_bin + (
            ":" + _os.environ["PATH"] if _os.environ.get("PATH") else ""
        )

from .tools import (
    # File Discovery
    inspect_root_data,
    # Histograms & Stats
    get_histogram_stats,
    histogram_integral,
    root_tree_to_histogram,
    # Selection & Analysis
    apply_cut_and_count,
    generate_cutflow,
    find_optimal_cut,
    compute_efficiency,
    # Statistical Tests
    compute_significance,
    histogram_significance_and_cls,
    summarize_parameter_scan,
    # Variable & Export
    define_variable,
    define_variable_and_plot,
    root_tree_to_csv,
    # Plotting
    plot,
    plot_2d,
    plot_significance_and_cls,
    # Fitting
    fit_distribution,
    fit_model,
    # Statistical Inference (RooStats)
    compute_discovery_significance,
    compute_upper_limit,
)

TOOLS = [
    # 1. File Discovery
    inspect_root_data,
    # 2. Histograms & Stats
    get_histogram_stats,
    histogram_integral,
    root_tree_to_histogram,
    # 3. Selection & Analysis
    apply_cut_and_count,
    generate_cutflow,
    find_optimal_cut,
    compute_efficiency,
    # 4. Statistical Tests
    compute_significance,
    histogram_significance_and_cls,
    summarize_parameter_scan,
    # 5. Variable & Export
    define_variable,
    define_variable_and_plot,
    root_tree_to_csv,
    # 6. Plotting
    plot,
    plot_2d,
    plot_significance_and_cls,
    # 7. Fitting
    fit_distribution,
    fit_model,
    # 8. Statistical Inference (RooStats)
    compute_discovery_significance,
    compute_upper_limit,
]

INSTRUCTIONS = """You are RooAgent -- a ROOT high-energy physics analysis assistant.

Available tools:
  Inspection:   inspect_root_data
  Counting:     apply_cut_and_count, generate_cutflow, compute_significance, compute_efficiency
  Statistics:   histogram_significance_and_cls, summarize_parameter_scan
  Histograms:   histogram_integral, get_histogram_stats, root_tree_to_histogram
  Plotting:     plot, plot_2d, plot_significance_and_cls
  Fitting:      fit_distribution, fit_model
  Variables:    define_variable, define_variable_and_plot, find_optimal_cut
  Export:       root_tree_to_csv
  Inference:    compute_discovery_significance, compute_upper_limit

Recommended Workflows:
1) File discovery and validation
   - Start with inspect_root_data(mode='summary'), then mode='branches' before writing any cuts.

2) Cut-based yield analysis
   - Use generate_cutflow to inspect cumulative cut efficiency per file.
   - Use apply_cut_and_count for individual cut-point yields.
   - Use compute_significance(cuts=[...]) -- always pass cuts as a list to avoid
     C++ operator-precedence errors when mixing && and ||.
   - Use compute_efficiency for selection acceptance or trigger efficiency.

3) Histogram statistics and mass windows
   - Build histograms with root_tree_to_histogram (name sig='sig', bkg='bkg' by convention).
   - Use get_histogram_stats, histogram_integral for basic stats.
   - Use histogram_significance_and_cls(compute_cls=False) for discovery; set compute_cls=True only for exclusion.

4) Stacked MC plots with data overlay
   - Use plot(mode='signal_background', background_files=[...], data_file=..., plot_data=True,
     weight_branch=..., cuts=[...]) for the standard stacked-MC + data plot.
   - stack_signal=True (default): signal is added to the stack on top of backgrounds.
   - stack_signal=False: backgrounds only are stacked; signal(s) are overlaid as separate colored
     lines. Use this when you want to compare signal shapes against the background stack without
     stacking signal on top. Data is never stacked -- it is always shown as markers.
   - Signal files are optional: omit signal_file/signal_files to plot backgrounds + data only.

5) Parameter scans and significance curves
   - Run per-point stat tools (histogram_significance_and_cls or compute_significance) for each
     scan point. Keep all arrays strictly index-aligned.
   - Call summarize_parameter_scan with one parameter_values array and a series dict of results.
   - Plot with plot_significance_and_cls(parameter_values=..., significance=.../cls=..., ...).
   - Passing cls=[...] auto-draws the CLs=0.05 threshold line; use draw_cls_threshold=True to
     force it for a significance/y curve instead, and cls_threshold to change the line's value.

6) Variable definition and optimisation
   - define_variable: adds a derived branch and saves a new ROOT file.
   - define_variable_and_plot: defines branches, applies cuts, and plots in one step.
   - find_optimal_cut: scans a threshold and returns the cut that maximises S/sqrt(S+B).

7) Fitting
   - fit_distribution: chi2 TF1 fit.
   - fit_model: RooFit maximum-likelihood fit; unbinned for source='tree', binned for source='hist';
     signal_shape in {gauss, crystalball, voigt}, optional background_shape in {expo, chebychev, poly}
     for an extended fit reporting nsig/nbkg.

8) Statistical inference (RooStats)
   - compute_discovery_significance: observed p0/Z.
   - compute_upper_limit: observed + median-expected CLs upper limit on nsig with +-1/+-2sigma bands.
   - Profile-likelihood analogues of workflow 2/5, built on the fit_model peak model
     (background_shape required). Same thresholds as below: Z >= 5 discovery, CLs <= 0.05 exclusion.

Defaults:
  - vector_mode: 'any' -- vector branch cuts match if any element satisfies the condition.
  - bins/xmin/xmax: 40 bins in [0, 100] for histogram building.
  - Cuts are valid C++ boolean expressions (e.g. "pt > 25 && abs(eta) < 2.4").

Discipline:
  - Always validate files with inspect_root_data before tools that open ROOT files.
  - For compute_significance with multi-condition selections, always pass cuts=[...] (list),
    never a single combined string that mixes || and && without explicit parentheses.
  - State whether results are Asimov/expected or observed (n_obs provided).
  - Discovery: Z >= 5 (p ~ 3e-7). CLs exclusion: CLs <= 0.05.
  - Run one tool at a time unless parallel execution is explicitly requested.
  - Reporting: summarize results in plain language instead of pasting raw tool output; for large
    results, give the key numbers only. Be honest: report exactly what the numbers show, and say
    so plainly if something is uncertain, missing, or failed.
"""

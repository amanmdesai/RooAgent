#!/usr/bin/env python3
"""MCP server exposing RooAgent ROOT-analysis tools to the Claude CLI.

Usage:
    rooagent-mcp          # via entry-point after `pip install -e .`
    python -m rooagent.mcp_server

Register with Claude CLI:
    claude mcp add rooagent -- rooagent-mcp
"""

import os as _os
import sys as _sys

# ---------------------------------------------------------------------------
# Bootstrap ROOT environment before importing any ROOT-dependent modules.
# ROOT must be compiled for the same Python version as this interpreter.
# Adjust ROOTSYS if your installation path differs.
# ---------------------------------------------------------------------------
_ROOTSYS = _os.environ.get(
    "ROOTSYS",
    "/home/a1905029/HEP-Softwares/ROOT/root_install",
)
_root_lib = _os.path.join(_ROOTSYS, "lib", "root")
_root_bin = _os.path.join(_ROOTSYS, "bin")

for _p in [_root_lib, _ROOTSYS + "/lib"]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

_ld = _os.environ.get("LD_LIBRARY_PATH", "")
for _p in [_root_lib, _ROOTSYS + "/lib"]:
    if _p not in _ld:
        _ld = _p + (":" + _ld if _ld else "")
_os.environ["LD_LIBRARY_PATH"] = _ld

if _root_bin not in _os.environ.get("PATH", ""):
    _os.environ["PATH"] = _root_bin + ":" + _os.environ.get("PATH", "")

_os.environ.setdefault("ROOTSYS", _ROOTSYS)
# ---------------------------------------------------------------------------

from mcp.server.fastmcp import FastMCP

from .tools import (
    inspect_root_data,
    get_histogram_stats,
    histogram_integral,
    histogram_significance_and_cls,
    summarize_parameter_scan,
    root_tree_to_histogram,
    root_tree_to_csv,
    apply_cut_and_count,
    generate_cutflow,
    compute_significance,
    compute_efficiency,
    find_optimal_cut,
    define_variable,
    define_variable_and_plot,
    plot,
    plot_2d,
    fit_distribution,
    plot_significance_and_cls,
)

_LC_TOOLS = [
    inspect_root_data,
    get_histogram_stats,
    histogram_integral,
    histogram_significance_and_cls,
    summarize_parameter_scan,
    root_tree_to_histogram,
    root_tree_to_csv,
    apply_cut_and_count,
    generate_cutflow,
    compute_significance,
    compute_efficiency,
    find_optimal_cut,
    define_variable,
    define_variable_and_plot,
    plot,
    plot_2d,
    fit_distribution,
    plot_significance_and_cls,
]


def create_mcp() -> FastMCP:
    """Build the MCP server that Claude CLI connects to."""

    server = FastMCP(
        "RooAgent",
        instructions=(
            "ROOT high-energy physics analysis assistant. "
            "Always call inspect_root_data(mode='summary') first, then mode='branches' "
            "before writing any cuts or histogramming commands."
        ),
    )

    for lc_tool in _LC_TOOLS:
        server.tool()(lc_tool)

    return server


mcp = create_mcp()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

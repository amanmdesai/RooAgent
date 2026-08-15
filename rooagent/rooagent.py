#!/usr/bin/env python3
"""MCP server for RooAgent's ROOT tools, plus the `rooagent` dispatcher
entry point (MCP vs LangGraph, see cli.py). See README for usage/registration."""

import argparse
import os
import sys

from mcp.server.fastmcp import FastMCP

from .tool_registry import TOOLS, INSTRUCTIONS


def create_mcp() -> FastMCP:
    server = FastMCP(
        "RooAgent",
        instructions=INSTRUCTIONS,
    )

    for lc_tool in TOOLS:
        server.tool()(lc_tool)

    return server


mcp = create_mcp()


def main() -> None:
    mcp.run()


def _resolve_mode(cli_mode: str | None) -> str:
    if cli_mode:
        return cli_mode
    env_mode = os.getenv("ROOAGENT_MODE", "").lower()
    if env_mode in ("mcp", "api"):
        return env_mode
    return "api"


def cli_main() -> None:
    parser = argparse.ArgumentParser(prog="rooagent")
    parser.add_argument(
        "--mode",
        choices=["mcp", "api"],
        default=None,
        help="Run as an MCP server ('mcp') or the LangGraph chatbot ('api'). "
        "Defaults to 'api' if unspecified (also overridable via ROOAGENT_MODE).",
    )
    args = parser.parse_args()

    mode = _resolve_mode(args.mode)

    if mode == "mcp":
        main()
    else:
        try:
            from .cli import main as agent_main
        except ImportError:
            print(
                "API mode requires the LangGraph dependencies:\n"
                "  pip install .",
                file=sys.stderr,
            )
            raise SystemExit(1)
        agent_main()


if __name__ == "__main__":
    main()

import asyncio

from rooagent import mcp_server


EXPECTED_TOOL_NAMES = [
    "inspect_root_data",
    "get_histogram_stats",
    "histogram_integral",
    "histogram_significance_and_cls",
    "summarize_parameter_scan",
    "root_tree_to_histogram",
    "root_tree_to_csv",
    "apply_cut_and_count",
    "generate_cutflow",
    "compute_significance",
    "compute_efficiency",
    "find_optimal_cut",
    "define_variable",
    "define_variable_and_plot",
    "plot",
    "plot_2d",
    "fit_distribution",
    "plot_significance_and_cls",
]


async def _list_tool_names(server):
    tools = await server.list_tools()
    return [tool.name for tool in tools]


def test_create_mcp_registers_all_claude_tools():
    server = mcp_server.create_mcp()

    tool_names = asyncio.run(_list_tool_names(server))

    assert tool_names == EXPECTED_TOOL_NAMES


def test_module_server_matches_expected_tool_surface():
    tool_names = asyncio.run(_list_tool_names(mcp_server.mcp))

    assert tool_names == EXPECTED_TOOL_NAMES


def test_main_calls_run(monkeypatch):
    called = {}

    def fake_run():
        called["run"] = True

    monkeypatch.setattr(mcp_server.mcp, "run", fake_run)

    mcp_server.main()

    assert called == {"run": True}

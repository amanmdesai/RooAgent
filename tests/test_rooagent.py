import asyncio

from rooagent import rooagent


EXPECTED_TOOL_NAMES = [
    "inspect_root_data",
    "get_histogram_stats",
    "histogram_integral",
    "root_tree_to_histogram",
    "apply_cut_and_count",
    "generate_cutflow",
    "find_optimal_cut",
    "compute_efficiency",
    "compute_significance",
    "histogram_significance_and_cls",
    "summarize_parameter_scan",
    "define_variable",
    "define_variable_and_plot",
    "root_tree_to_csv",
    "plot",
    "plot_2d",
    "plot_significance_and_cls",
    "fit_distribution",
    "fit_model",
    "compute_discovery_significance",
    "compute_upper_limit",
]


async def _list_tool_names(server):
    tools = await server.list_tools()
    return [tool.name for tool in tools]


def test_create_mcp_registers_all_claude_tools():
    server = rooagent.create_mcp()

    tool_names = asyncio.run(_list_tool_names(server))

    assert tool_names == EXPECTED_TOOL_NAMES


def test_module_server_matches_expected_tool_surface():
    tool_names = asyncio.run(_list_tool_names(rooagent.mcp))

    assert tool_names == EXPECTED_TOOL_NAMES


def test_main_calls_run(monkeypatch):
    called = {}

    def fake_run():
        called["run"] = True

    monkeypatch.setattr(rooagent.mcp, "run", fake_run)

    rooagent.main()

    assert called == {"run": True}


def test_resolve_mode_defaults_to_api(monkeypatch):
    monkeypatch.delenv("ROOAGENT_MODE", raising=False)

    assert rooagent._resolve_mode(None) == "api"


def test_resolve_mode_uses_env(monkeypatch):
    monkeypatch.setenv("ROOAGENT_MODE", "mcp")

    assert rooagent._resolve_mode(None) == "mcp"


def test_resolve_mode_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("ROOAGENT_MODE", "mcp")

    assert rooagent._resolve_mode("api") == "api"

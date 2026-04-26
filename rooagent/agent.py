from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator
import os
import re
# Import all tools
from .tools import *

# -----------------------------
# SYSTEM PROMPT (General)
# -----------------------------
SYSTEM_PROMPT = """
You are RooAgent — a ROOT high-energy physics analysis assistant.

Tools:
- Inspection: `inspect_root_data` — inspect ROOT files and enumerate trees, branches, and file contents.
- Counting: `apply_cut_and_count`, `generate_cutflow`, `compute_significance` — event counting and significance estimation tools.
- Statistics: `histogram_significance_and_cls`, `summarize_parameter_scan`, `compute_significance`.
- Histograms: `histogram_integral`, `histogram_significance_and_cls`, `get_histogram_stats`.
- Plotting: `plot`, `plot_2d`, `plot_significance_and_cls`.
- Fitting: `fit_distribution`.
- Variables: `define_variable`, `define_variable_and_plot`, `find_optimal_cut`, `root_tree_to_histogram`.
- Export: `root_tree_to_csv`.

Recommended Workflows:
1) ROOT-file discovery and validation
- Use `inspect_root_data(mode="summary")` first.
- If needed, follow with `inspect_root_data(mode="trees"|"branches", file_path=..., tree_name=...)`.

2) Cut-based yield workflow
- Use `generate_cutflow` to inspect cumulative cut efficiency.
- Use `apply_cut_and_count` for specific cut points.
- Use `compute_significance` for a direct S/B/Z estimate at a chosen cut.

3) Histogram-statistics workflow
    - Build or inspect histograms (`root_tree_to_histogram`, `get_histogram_stats`, `histogram_integral`).
    - Compute window-based stats with `histogram_significance_and_cls`.

4) Statistics-to-plot chaining
- For a scan over any parameter, evaluate significance/CLs per point using stat tools.
- Keep scan arrays strictly aligned by index. Use `summarize_parameter_scan` with one `parameter_values` array plus a dictionary of named result arrays.
- Aggregate arrays in the same order: `parameter_values`, and exactly one y-array (`significance` or `cls`) when plotting.
- For any "best candidate" report across scanned points, call `summarize_parameter_scan` and use its best point directly. Do not manually rebuild rankings from free-text tool outputs.
- When scan calls run in parallel, map each point using explicit identifiers reported by tools (`Signal=...` and `Center=...`), never by response order.
- Plot with `plot_significance_and_cls(parameter_values=..., significance=...|cls=...|y=..., output_png=... or output_pdf=...)`.

5) Reporting discipline
- Always state whether results are expected (Asimov/expected counts) or observed (`n_obs` provided).
- Include the cut/window definition and key inputs (S, B, and Nobs when applicable).

Defaults:
- `vector_mode`: "any" (default). Treat vector branches as matching if any element satisfies the condition; use "all" to require every element to satisfy the condition.
- `cuts`: optional selection expressions; omit to use unfiltered data.
- `bins`, `xmin`, `xmax`: default histogram binning is 40 bins in the interval [0, 100].

Execution:
- Validate input files with `inspect_root_data` before running tools that open ROOT files.
- Cuts must be valid C++ boolean expressions (for example: "m > 120 && m < 130").
- Report results with appropriate units and uncertainty estimates.
- Execute one tool at a time unless parallel execution is explicitly requested.
- When users request curves vs a scanned parameter, do not stop at scalar outputs: run the per-point stat workflow and finish with `plot_significance_and_cls`.

Notes:
- Conventionally, discovery is claimed at Z ≥ 5 (p ≈ 3×10^-7).
- Treat p-values as one-sided by default.
"""


# -----------------------------
# Initialize Model (GitHub Copilot / GitHub Models API)
# -----------------------------
DEFAULT_MODEL_NAME = "openai/gpt-4.1"
MODEL_NAME = os.getenv("MODEL", DEFAULT_MODEL_NAME)
DEFAULT_SEED = int(os.getenv("ROOAGENT_SEED", "7"))

model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=0,
    seed=DEFAULT_SEED,
    top_p=1,
)


tools = [
    inspect_root_data,
    get_histogram_stats,
    histogram_integral,
    histogram_significance_and_cls,
    summarize_parameter_scan,
    root_tree_to_csv,
    apply_cut_and_count,
    generate_cutflow,
    compute_significance,
    find_optimal_cut,
    define_variable,
    define_variable_and_plot,
    plot,
    plot_2d,
    fit_distribution,
    plot_significance_and_cls,
]


tools_by_name = {tool.name: tool for tool in tools}

# -----------------------------
# State Definition
# -----------------------------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# -----------------------------
# Tool Node
# -----------------------------
def tool_node(state: MessagesState):
    results = []

    for tool_call in state["messages"][-1].tool_calls:

        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": results}

# -----------------------------
# LLM Node
# -----------------------------
def llm_call(state: MessagesState):
    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    parallel_requested = _user_requested_parallel_tools(state)

    model_with_tools = model.bind_tools(tools, parallel_tool_calls=parallel_requested)

    response = model_with_tools.invoke(prompt_messages)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def _user_requested_parallel_tools(state: MessagesState) -> bool:
    user_text = ""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            content = getattr(message, "content", "")
            user_text = content if isinstance(content, str) else str(content)
            break

    if not user_text:
        return False

    text = user_text.lower()
    patterns = [
        r"parallel_tool_calls\s*[:=]\s*(true|yes|1)",
        r"\bparallel(?:ize|ise)?\b",
        r"\bin\s+parallel\b",
        r"\bparallel\s+tool\s+calls?\b",
    ]
    return any(re.search(pat, text) for pat in patterns)

# -----------------------------
# Routing
# -----------------------------
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END

# -----------------------------
# Build Graph
# -----------------------------
builder = StateGraph(MessagesState)

builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")

agent = builder.compile()

# -----------------------------
# CLI
# -----------------------------
def main():
    print(f"\nROOT Physics Analysis Agent using ({MODEL_NAME})")
    print("Type 'exit' to quit\n")

    state = {"messages": [], "llm_calls": 0}

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Thanks for using RooAgent!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        state = agent.invoke(state)

        reply = state["messages"][-1]
        text = getattr(reply, "content", "")
        if isinstance(text, list):
            text = " ".join(str(x) for x in text)
        print(f"Assistant: {text}\n")
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver  
import operator
import os
# Import all tools
from .tools import *

# -----------------------------
# SYSTEM PROMPT (Clean Version)
# -----------------------------
SYSTEM_PROMPT = """
ROOT-based HEP analysis assistant.

Hard rules (follow exactly):

- Always use tools. Never guess or fabricate outputs.
- Validate inputs before use: call `inspect_root_data` with the appropriate mode (`files`, `summary`, `trees`, `branches`, `contents`).
- Cuts must be expressed in C++ syntax: `true`/`false`, `&&`, `||`. If the user supplies Python-style booleans/logicals (True/False, and/or) rewrite them to C++ or call the project's rewrite utility prior to tool calls.
- Plotting calls: always supply numeric `xmin` and `xmax`. If a plotting tool complains about missing fields ("Field required"), retry the call adding the missing named arguments before aborting.
- Use `plot_1d` for all 1D plotting (`hist`, `tree`, `tree_compare`, `hist_compare`, `signal_background`) and `plot_2d` for all 2D plotting (`hist`, `tree`).
- When plotting MC, pass `weight_branch` (e.g. `weight`) to all MC histograms; data overlays must use unit weight (do not apply MC weights to data).
- For cutflow: call `generate_cutflow(file_path=..., file_paths=[...], tree_name=..., cuts=[...], weight=...)` once over all MC files (signal + backgrounds). Do NOT include `data.root` when computing the weighted MC cutflow. If a data-only cutflow is desired, call `generate_cutflow` separately for data with no weight.
- For counting events passing a single cut: call `apply_cut_and_count(file_path=..., tree_name=..., cut=...)`.
- **Significance tool selection (choose exactly one based on input type):**
  - Use `compute_significance` when inputs are **TTrees** (event-level ROOT files with a tree name and a selection cut). This computes S/sqrt(S+B) from RDataFrame yields. Always call it once with ALL background files together via `background_files=[...]`.
    - Use `histogram_significance_and_limits` when inputs are **pre-built TH1 histograms** already stored in ROOT files. It computes a counting-based discovery significance and a CLs exclusion ratio from the same counting window. Data histogram is optional — if omitted, N=S+B is assumed (expected sensitivity only).
  - Do NOT call both tools for the same analysis. Do NOT call `compute_significance` when you have histograms, and do NOT call `histogram_significance_and_limits` when you only have trees.
    - When using `histogram_significance_and_limits`: first call `inspect_root_data(mode='contents', file_path=...)` to discover available histogram names in the file. Then call `histogram_significance_and_limits` with `file_path` and explicit histogram names (`bkg_name` and `sig_name`). If a data histogram is present and requested, pass `data_name`; otherwise pass an empty string to compute expected sensitivity (assume N=S+B).
    - Always provide numeric `center` and `window` values specifying the counting window (same units as the histogram x-axis). The tool uses the stored binning as-is; no rebinning is applied.
    - The tool returns a single-line summary. The string contains discovery p-value/significance plus `CLs`, `CLs+b`, and `CLb`. After calling the tool, return the tool output and a one-line human-friendly summary (e.g. "Discovery Z=...; CLs=...; see tool output").
- For optimal cut scanning: call `find_optimal_cut(signal_file=..., background_files=[...], tree_name=..., variable=..., min_cut=..., max_cut=..., step=...)`.
- For `histogram_integral`: always supply `x_low` and `x_high`. Do not call it without those numeric bounds.
- For `plot_1d(mode='signal_background')`, provide explicit `background_labels` whenever possible to avoid generic labels.
- Return concise, factual outputs: a short cutflow summary (one line per cut), the saved PDF path, and the computed significance Z.
Always save plots to the requested path and return a final brief summary line: saved PDF path and significance Z.
"""


# -----------------------------
# Initialize Model (GitHub Copilot / GitHub Models API)
# -----------------------------
DEFAULT_MODEL_NAME = "openai/gpt-4.1"
MODEL_NAME = os.getenv("MODEL", DEFAULT_MODEL_NAME)

model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference"
)


tools = [
    inspect_root_data,
    get_histogram_stats,
    histogram_integral,
    histogram_significance_and_limits,
    root_tree_to_csv,
    apply_cut_and_count,
    generate_cutflow,
    compute_significance,
    find_optimal_cut,
    define_variable,
    define_variable_and_plot,
    plot_1d,
    plot_2d,
    fit_distribution,
]


tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

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
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

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
builder.add_node("tool_node",ToolNode(tools,handle_tool_errors=True)) 
                                      # tool_node)

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

        # FIXED BUG HERE
        state["messages"].append(HumanMessage(content=user_input))

        state = agent.invoke(state)
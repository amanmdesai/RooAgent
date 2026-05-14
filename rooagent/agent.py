from langchain_ollama import ChatOllama
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

Available tools:
  Inspection:   inspect_root_data
  Counting:     apply_cut_and_count, generate_cutflow, compute_significance, compute_efficiency
  Statistics:   histogram_significance_and_cls, summarize_parameter_scan
  Histograms:   histogram_integral, get_histogram_stats, root_tree_to_histogram
  Plotting:     plot, plot_2d, plot_significance_and_cls
  Fitting:      fit_distribution
  Variables:    define_variable, define_variable_and_plot, find_optimal_cut
  Export:       root_tree_to_csv

Recommended Workflows:
1) File discovery and validation
   - Start with inspect_root_data(mode='summary'), then mode='branches' before writing any cuts.

2) Cut-based yield analysis
   - Use generate_cutflow to inspect cumulative cut efficiency per file.
   - Use apply_cut_and_count for individual cut-point yields.
   - Use compute_significance(cuts=[...]) — always pass cuts as a list to avoid
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
     stacking signal on top. Data is never stacked — it is always shown as markers.
   - Signal files are optional: omit signal_file/signal_files to plot backgrounds + data only.

5) Parameter scans and significance curves
   - Run per-point stat tools (histogram_significance_and_cls or compute_significance) for each
     scan point. Keep all arrays strictly index-aligned.
   - Call summarize_parameter_scan with one parameter_values array and a series dict of results.
   - Plot with plot_significance_and_cls(parameter_values=..., significance=.../cls=..., ...).
   - For CLs exclusion curves set draw_cls_threshold=True, cls_threshold=0.05.

6) Variable definition and optimisation
   - define_variable: adds a derived branch and saves a new ROOT file.
   - define_variable_and_plot: defines branches, applies cuts, and plots in one step.
   - find_optimal_cut: scans a threshold and returns the cut that maximises S/sqrt(S+B).

Defaults:
  - vector_mode: 'any' — vector branch cuts match if any element satisfies the condition.
  - bins/xmin/xmax: 40 bins in [0, 100] for histogram building.
  - Cuts are valid C++ boolean expressions (e.g. "pt > 25 && abs(eta) < 2.4").

Discipline:
  - Always validate files with inspect_root_data before tools that open ROOT files.
  - For compute_significance with multi-condition selections, always pass cuts=[...] (list),
    never a single combined string that mixes || and && without explicit parentheses.
  - State whether results are Asimov/expected or observed (n_obs provided).
  - Discovery: Z ≥ 5 (p ≈ 3×10⁻⁷). CLs exclusion: CLs ≤ 0.05.
  - Run one tool at a time unless parallel execution is explicitly requested.
"""


# -----------------------------
# Initialize Model (GitHub Copilot / GitHub Models API)
# -----------------------------
DEFAULT_MODEL_NAME = "deepseek-v3.1:671b-cloud"
MODEL_NAME = os.getenv("MODEL", DEFAULT_MODEL_NAME)
DEFAULT_SEED = int(os.getenv("ROOAGENT_SEED", "7"))

# model = ChatOpenAI(
#     model=MODEL_NAME,
#     api_key=os.getenv("GITHUB_TOKEN"),
#     base_url="https://models.github.ai/inference",
#     temperature=0,
#     seed=DEFAULT_SEED,
#     top_p=1,
# )

model = ChatOllama(model=MODEL_NAME, temperature=0, seed=DEFAULT_SEED)

tools = [
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

    model_with_tools = model.bind_tools(tools)# parallel_tool_calls=parallel_requested)

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

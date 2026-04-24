from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator
import os
# Import all tools
from .tools import *

# -----------------------------
# SYSTEM PROMPT (General)
# -----------------------------
SYSTEM_PROMPT = """
You are a concise, expert assistant for ROOT-based physics analysis and RooStats.

Recommended physics workflow (concise):
- Inspect inputs first: call inspect_root_data to list TTrees, branches, and top-level histograms and note variable types/ranges.
- Prefer histogram-first counting: convert trees to TH1s via root_tree_to_histogram (or helper) before counting; always verify binning and ranges.
- Multi-file backgrounds: perform explicit accumulation upstream when needed; histogram_significance_and_limits expects a single file with explicit histogram names.
- Build observables: use define_variable / define_variable_and_plot for derived kinematics (e.g. leading jet pT, mass windows).
- Selection & yields: use apply_cut_and_count and generate_cutflow for weighted/unweighted yields.
- Statistical inference (canonical): use histogram_significance_and_limits (p0 converted to one-sided Z via RooStats PValueToSignificance) and histogram_upper_limit for mu limits.
- Optimization: use find_optimal_cut with coarse steps first and refine when promising; avoid overly fine scans by default.
- Visualization & validation: use plot, plot_2d, and plot_significance_and_cls; set apply_cuts_before_plot=True for plots used in cut studies.
- Fit as needed: use fit_distribution for resonance checks; report fitted parameters and fit quality.
- Export: use root_tree_to_csv for compact downstream summaries.

LLM & tool usage guidelines:
- Use tool calls rather than free-form code; be explicit and concise.
- Before invoking a tool, state defaults (bins, range, weights, method) and confirm them.
- Validate histograms before arithmetic: check binning, axis ranges, and sum compatibility; ask to rebin if needed.
- When summing multiple backgrounds, prefer adding TH1s with identical binning; otherwise request rebinning.
- Allow parallel tool calls only if the user explicitly requests `parallel_tool_calls: true` or asks for parallel execution.
- Keep tool docstrings short and focused; rely on tool names and arguments rather than listing other tools.
- Favor computational efficiency: avoid repeated full-file passes, excessive bins, or unnecessarily fine scans.

Work execution:
- Proceed automatically when user requests auto execution but ask clarifying questions when inputs are ambiguous.
- Return a concise summary of actions including tool names, their arguments, and short numerical results.
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
    histogram_significance_and_limits,
    histogram_upper_limit,
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
# Default to sequential tool calls. We'll bind tools per LLM invocation
# and enable parallel tool calls only when the prompt explicitly requests it.
model_with_tools = None

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

    # Inspect combined prompt/messages to decide whether to allow parallel tool calls.
    concat = " ".join(
        (getattr(m, "content", "") or "") for m in prompt_messages
    ).lower()

    parallel_requested = False
    # Accept both explicit flags and natural-language requests.
    if "parallel_tool_calls" in concat and ("true" in concat or "yes" in concat):
        parallel_requested = True
    if "parallel tool" in concat or ("in parallel" in concat and "tool" in concat):
        parallel_requested = True
    if "parallelize" in concat or "parallelise" in concat:
        parallel_requested = True

    model_with_tools = model.bind_tools(tools, parallel_tool_calls=parallel_requested)

    response = model_with_tools.invoke(prompt_messages)

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

        # FIXED BUG HERE
        state["messages"].append(HumanMessage(content=user_input))

        state = agent.invoke(state)
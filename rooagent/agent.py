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
SYSTEM_PROMPT = """You are RooAgent — a ROOT/HEP analysis agent. Convert user requests into tool calls, interpret results, and iterate until complete.


WORKFLOWS:
1. Always start: inspect_root_data(mode="summary"), then mode="branches" before writing any cuts.
2. Cut discovery: generate_cutflow → apply_cut_and_count → compute_significance
3. Histogram discovery scan: histogram_significance_and_cls(compute_cls=False) per point → summarize_parameter_scan → plot_significance_and_cls
4. Exclusion scan: histogram_significance_and_cls(compute_cls=True) per point → summarize_parameter_scan → plot_significance_and_cls(draw_cls_threshold=True)
5. Combined: compute_cls=True, parse Z and CLs, run separate plots for each.

RULES:
- Validate files/branches before use. Cuts are C++ (e.g. "pt > 25 && abs(eta) < 2.4").
- Keep scan arrays index-aligned; use summarize_parameter_scan to rank, never rerank manually.
- Output directories must exist before plotting; tool will not create them.
- Report units; label expected (Asimov) vs observed explicitly.
- vector_mode default: "any". Default histogram: 40 bins [0,100] — override for physics vars.

STATISTICS (critical — never mix these):
- DISCOVERY: compute_cls=False. Z = Φ⁻¹(1−p0), p0 = P(N≥n|B). Claim at Z≥5; evidence at Z≥3.
- EXCLUSION: compute_cls=True. CLs = CLs+b/CLb; CLs<0.05 → excluded at 95% CL. CLs is NOT discovery.
- COMBINED: compute_cls=True, report Z and CLs separately.
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
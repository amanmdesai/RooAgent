from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver  
import operator
import os
# Import all tools
from .tools import *

# SYSTEM PROMPT
SYSTEM_PROMPT = """
ROOT HEP analysis assistant. Inspect files → select trees/branches → analyze with tools.

WORKFLOW: list_root_file_contents() → list_ttrees() → analyze using plotting/cutting tools.

TOOLS: Histograms, overlays (signal/backgrounds/data), cuts, fits, variables, cutflows, CSV export.

PLOTTING: backgrounds=filled, signals=lines, data=markers. Include labels, legends, ratios when needed.

ANALYSIS: vector_mode="any" (≥1 object) or "all" (all objects). Report S, B, significance.

RULES: Always use tools; verify outputs; no assumptions; clear error messages.
"""

# -----------------------------
# Initialize Model (GitHub Copilot / GitHub Models API)
# -----------------------------
DEFAULT_MODEL_NAME = "openai/gpt-4o-mini"
MODEL_NAME = os.getenv("MODEL", DEFAULT_MODEL_NAME)

model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference"
)

# Bind tools
tools = [
    list_root_file_contents,
    list_tree_branches,
    get_histogram_stats,
    apply_cut_and_count,
    compute_significance,
    plot_tree_variable,
    compare_tree_variables,
    root_tree_to_csv,
    define_variable_and_plot,
    fit_tree_variable,
    fit_histogram,
    draw_histograms_same_canvas,
    draw_2d_histogram,
    plot_signal_vs_backgrounds,
    find_optimal_cut,
    draw_1d_histogram,
    define_variable,
    draw_2d_histogram_from_tree,
    generate_cutflow,
    discover_root_data,
    list_ttrees,
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

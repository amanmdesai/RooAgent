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
ROOT-based HEP analysis assistant.

RULES:
1. Always use tools. Never guess.
2. Verify everything (files, trees, branches) before use.
3. Workflow: inspect file -> trees -> branches -> analyze.
4. Auto-use single tree; list if multiple.
5. Handle missing data and empty results safely.
6. Compare signal vs background when relevant.
7. Output: clear plots, labeled axes, report statistics.
8. No assumptions. State uncertainty if unverified.

Prioritize correctness, efficiency, reproducibility.
"""

# -----------------------------
# Multi-model fallback (rate-limit resilience)
# If the primary model hits a 429, the next model in the list is tried.
# Set MODEL env var to override the primary. Add more models to extend fallback chain.
# -----------------------------
_PRIMARY = os.getenv("MODEL", "openai/gpt-5-mini")
FALLBACK_MODELS = list(dict.fromkeys([
    _PRIMARY,
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-5-mini",
    "microsoft/Phi-4-mini-instruct",
    "deepseek/DeepSeek-R1",
]))

# Limits to reduce token usage per call
MAX_HISTORY = 2       # keep only the last N messages (trims unbounded history growth)
MAX_TOOL_CHARS = 3000  # truncate long tool outputs before they enter the context

# Bind tools lazily to each model (built once at import time)
def _make_bound_model(name: str):
    return ChatOpenAI(
        model=name,
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.github.ai/inference"
    ).bind_tools(tools)

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

# Lazy-bound model dict. Models are bound the first time they're used to avoid
# creating external client instances at import time (which requires API keys).
_bound_models = {}

# -----------------------------
# State Definition
# -----------------------------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# -----------------------------
# Tool Node — with output truncation
# -----------------------------
def tool_node(state: MessagesState):
    results = []

    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool = {t.name: t for t in tools}[tool_name]
        observation = tool.invoke(tool_call["args"])

        content = str(observation)
        if len(content) > MAX_TOOL_CHARS:
            content = content[:MAX_TOOL_CHARS] + f"\n[...truncated {len(str(observation)) - MAX_TOOL_CHARS} chars]"

        results.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": results}


def _trim_messages(messages):
    """Keep only the last MAX_HISTORY messages; never start with a ToolMessage."""
    if len(messages) <= MAX_HISTORY:
        return messages
    trimmed = messages[-MAX_HISTORY:]
    # Shift forward past any leading ToolMessages to keep conversation structure valid
    while trimmed and isinstance(trimmed[0], ToolMessage):
        trimmed = trimmed[1:]
    return trimmed


# -----------------------------
# LLM Node — with history trimming and model fallback
# -----------------------------
def llm_call(state: MessagesState):
    trimmed = _trim_messages(state["messages"])
    prompt = [SystemMessage(content=SYSTEM_PROMPT)] + trimmed

    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            if model_name not in _bound_models:
                _bound_models[model_name] = _make_bound_model(model_name)

            response = _bound_models[model_name].invoke(prompt)
            return {
                "messages": [response],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err or "quota" in err or "ratelimit" in type(e).__name__.lower():
                print(f"[RooAgent] {model_name} rate-limited, trying next model...")
                last_error = e
                continue
            raise  # non-rate-limit errors propagate immediately

    raise RuntimeError(f"All models rate-limited.") from last_error


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
builder.add_node("tool_node", ToolNode(tools, handle_tool_errors=True))

builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")

agent = builder.compile()

# -----------------------------
# CLI
# -----------------------------
def main():
    print(f"\nROOT Physics Analysis Agent")
    print(f"Primary model: {_PRIMARY}  |  Fallbacks: {FALLBACK_MODELS[1:]}")
    print("Type 'exit' to quit\n")

    state = {"messages": [], "llm_calls": 0}

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Thanks for using RooAgent!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        state = agent.invoke(state)


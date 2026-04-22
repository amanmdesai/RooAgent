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
You are a concise, helpful assistant that supports users working with ROOT-based data and analysis tools.

Guidelines:
- When asked to execute an automated, multi-step workflow (for example: scanning a grid of mass hypotheses), drive the workflow by issuing tool calls until the entire requested work is finished. Do not return a plain natural-language progress update that ends the tool-call loop; only return a final natural-language message after all requested steps and tool calls have completed.
- ALWAYS issue tool_calls for each analysis step or a clearly-defined batch of steps. Chunk long scans into manageable batches and continue issuing tool_calls until all grid points are processed.
- Proceed without per-step confirmations when the user requested "automatic execution".
- Prefer to use the available tools for data access, computation, and plotting instead of fabricating results.
- When user-supplied parameters are missing, assume sensible defaults and proceed. Explicitly state which defaults are used (either by including them in tool_call arguments or in the tool response text), e.g. "Using default window=25.0 GeV". Do not silently apply defaults.
- Validate user inputs before calling tools; if required parameters are missing or ambiguous, ask a clarifying question (unless the user explicitly requested an automatic run).
- Return concise, factual outputs. When a tool is used, include a brief summary and the tool call (name and arguments) in the response.
- When producing visualizations or numeric summaries, ensure required numeric arguments are present; otherwise ask for them.
- For deterministic behavior, prefer a single tool call at a time unless parallel calls are explicitly needed.
- If the user asks to apply cuts to a plot, pass cuts explicitly and set apply_cuts_before_plot=True for relevant plotting tools.
- If encountering token or rate limits, prefer minimal-tool-call continuation (smaller batches) rather than returning a final message.
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
    plot_1d,
    plot_2d,
    fit_distribution,
    plot_significance_and_cls,
]


tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools, parallel_tool_calls=False)

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
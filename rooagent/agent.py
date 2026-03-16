from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver  
import operator
# Import all tools
from .tools import *

# -----------------------------
# SYSTEM PROMPT (Clean Version)
# -----------------------------
SYSTEM_PROMPT = """
You are a ROOT-based High Energy Physics (HEP) analysis assistant.

Your primary goal is to help with:
- ROOT file inspection (.root)
- TTree and TBranch analysis
- Histogram creation and comparison
- Statistical calculations
- Signal/background studies
- Cut optimization
- Significance estimation
- Data visualization
- Automation of analysis workflows

IMPORTANT RULES:

1. Always use tools when available.
   - Only use tool-based actions.
   - If a task can be done via a tool, you MUST use it.
   - Never guess file contents — inspect them using tools.

2. If information is missing:
   - Search available ROOT files in the working directory.
   - Search for TTrees inside files if not specified.
   - If multiple trees exist, list them and select the appropriate one.
   - If only one tree exists, use it automatically.

3. Be robust:
   - Handle missing branches gracefully.
   - Handle empty histograms safely.
   - Check for file existence before accessing it.
   - Validate inputs before processing.

4. Analysis workflow preference:
   - Explore file structure first.
   - Inspect branches before plotting.
   - Retrieve statistics before fitting.
   - Visualize results when useful.
   - Compare signal vs background when applicable.

5. Output quality:
   - Use clear plots with labeled axes.
   - Include legends when comparing datasets.
   - Report statistical quantities explicitly.
   - Provide concise but complete explanations. No guesses.

6. Scientific rigor:
   - Use proper statistical methods.
   - Clearly distinguish between:
       * Signal
       * Background
   - Avoid assumptions without verification.

7. Efficiency:
   - Minimize unnecessary steps.
   - If a single tool call solves the task, do not overcomplicate.
   - Automate decisions when unambiguous.
   Use evidence-based reasoning.

Follow evidence-based reasoning. Do not make claims without supporting evidence.
Always verify statements using available data, computations, or tools before drawing conclusions.
If the evidence is insufficient, clearly state the uncertainty instead of guessing.
Prioritize scientific rigor, correctness, reproducibility, and proper use of analysis tools in all responses.
"""

# -----------------------------
# Initialize Model
# -----------------------------
model = ChatOllama(
    model="qwen3.5:latest",
    temperature=0,
    checkpointer=InMemorySaver(),
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
    print("\nROOT Physics Analysis Agent")
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

        print("\n--- Conversation Trace ---\n")
        for m in state["messages"]:
            m.pretty_print()
        print("\n--------------------------\n")


if __name__ == "__main__":
    main()
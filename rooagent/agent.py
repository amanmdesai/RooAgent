from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator
import inspect
import pkgutil
import importlib

# Import all tools from the tools subpackage
from . import tools

# Initialize LLM model
model = ChatOllama(model="gpt-oss:latest", temperature=0)


def get_all_tools():
    all_tools = []
    for _, module_name, _ in pkgutil.iter_modules(tools.__path__):
        module = importlib.import_module(f"{tools.__name__}.{module_name}")
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # Only add functions that have the 'name' attribute from @tool
            if hasattr(obj, "name"):
                all_tools.append(obj)
    return all_tools

tools = get_all_tools()
tools_by_name = {tool.name: tool for tool in tools}

# Map tool names to tool objects
model_with_tools = model.bind_tools(tools)

# Define message state for LangGraph
class MessagesState(TypedDict):
    """Tracks conversation state for the agent."""
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# -----------------------------
# Tool Node
# -----------------------------
def tool_node(state: MessagesState):
    """
    Execute any tool calls from the last LLM message and return the results.

    Parameters
    ----------
    state : MessagesState
        Current conversation state.

    Returns
    -------
    dict
        Updated messages from tool calls.
    """
    results = []
    last_message = state["messages"][-1]
    for tool_call in getattr(last_message, "tool_calls", []):
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": results}

# -----------------------------
# LLM Node
# -----------------------------
def llm_call(state: MessagesState):
    """
    Call the LLM with tools bound, updating conversation state.

    Parameters
    ----------
    state : MessagesState
        Current conversation state.

    Returns
    -------
    dict
        Updated state including LLM response and incremented call count.
    """
    response = model_with_tools.invoke(
        [SystemMessage(content="You are a ROOT High Energy Physics analysis assistant. Always use tools.")] + state["messages"]
    )
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

# -----------------------------
# Routing
# -----------------------------
def should_continue(state: MessagesState):
    """
    Decide the next node in the LangGraph.

    Parameters
    ----------
    state : MessagesState
        Current conversation state.

    Returns
    -------
    str
        Next node key.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END

# -----------------------------
# Build Agent
# -----------------------------
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()

# -----------------------------
# CLI / Main Function
# -----------------------------
def main():
    """
    Run the ROOT Physics Analysis Agent in terminal mode.

    Type 'exit' or 'quit' to terminate.
    """
    print("\nROOT Physics Analysis Agent")
    print("Type 'exit' to quit\n")
    state = {"messages": [], "llm_calls": 0}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Thanks for using RooAgent!")
            break
        state["messages"].append(HumanMessage(content=user_input))
        state = agent.invoke(state)
        print("\n--- Conversation Trace ---\n")
        for m in state["messages"]:
            m.pretty_print()
        print("\n--------------------------\n")


if __name__ == "__main__":
    main()
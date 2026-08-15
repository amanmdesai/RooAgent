"""LangGraph ReAct agent for RooAgent. Provider via LLM_PROVIDER
(anthropic | openai); see README for usage."""

import operator
import os
import re

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from .tool_registry import INSTRUCTIONS, TOOLS

# -----------------------------
# Tools (wrapped for LangChain here; tool_registry.TOOLS stays plain for FastMCP)
# -----------------------------
tools = [tool(fn) for fn in TOOLS]
tools_by_name = {t.name: t for t in tools}

# -----------------------------
# SYSTEM PROMPT (derived from the shared MCP instructions)
# -----------------------------
SYSTEM_PROMPT = "You are RooAgent, a ROOT high-energy physics analysis assistant.\n\n" + INSTRUCTIONS

# -----------------------------
# Initialize Model (runtime-selected provider)
# -----------------------------
PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
DEFAULT_SEED = int(os.getenv("ROOAGENT_SEED", "7"))

if PROVIDER == "anthropic":
    from langchain_anthropic import ChatAnthropic

    MODEL_NAME = os.getenv("MODEL", "claude-sonnet-5")
    model = ChatAnthropic(model=MODEL_NAME)
# elif PROVIDER == "ollama":
#     from langchain_ollama import ChatOllama
#
#     MODEL_NAME = os.getenv("MODEL", "llama3.1")
#     model = ChatOllama(
#         model=MODEL_NAME,
#         temperature=0,
#     )
else:  # "openai" (ChatGPT)
    from langchain_openai import ChatOpenAI

    MODEL_NAME = os.getenv("MODEL", "gpt-5.5")
    model = ChatOpenAI(model=MODEL_NAME, temperature=0, seed=DEFAULT_SEED)


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
        selected_tool = tools_by_name[tool_call["name"]]
        observation = selected_tool.invoke(tool_call["args"])

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": results}


# -----------------------------
# LLM Node
# -----------------------------
def llm_call(state: MessagesState):
    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    parallel_requested = _user_requested_parallel_tools(state)

    bind_kwargs = {}
    if PROVIDER == "openai":
        bind_kwargs["parallel_tool_calls"] = parallel_requested

    model_with_tools = model.bind_tools(tools, **bind_kwargs)

    response = model_with_tools.invoke(prompt_messages)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
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
    print(f"\nROOT Physics Analysis Agent using ({PROVIDER}: {MODEL_NAME})")
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


if __name__ == "__main__":
    main()

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
You are RooAgent — an expert ROOT/HEP analysis assistant that converts natural-language analysis
requests into precise sequences of Python-accessible tools backed by ROOT (C++ HEP framework) and
scipy. You operate as an autonomous agent: plan, call tools, interpret results, and iterate until
the analysis is complete.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEP STATISTICAL FRAMEWORK — READ THIS FIRST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

There are two fundamentally different statistical analyses in HEP. Use the correct one:

DISCOVERY (searching for a signal excess)
  Metric : p0 = P(N ≥ n_obs | background-only) — upper-tail Poisson survival function.
  Output : Z = Φ⁻¹(1 − p0) — Gaussian significance (one-sided).
  Threshold: Z ≥ 5 (p0 ≈ 3×10⁻⁷) for a discovery claim; Z ≥ 3 for evidence.
  Tools  : histogram_significance_and_cls(compute_cls=False), compute_significance, plot with Do NOT pass compute_cls=True.

EXCLUSION (setting an upper limit on a signal hypothesis)
  Metric : CLs = CLs+b / CLb, where
           CLs+b = P(N ≤ n_obs | S+B) — lower-tail Poisson CDF,
           CLb   = P(N ≤ n_obs | B)   — lower-tail Poisson CDF.
           CLs ∈ [0,1]; CLs < 0.05 → signal excluded at 95 % CL.
  Tools  : histogram_significance_and_cls(compute_cls=True), plot with cls=cls_obs_arr.
  ⚠  CLs is NOT a discovery metric. Small CLs means the signal is EXCLUDED, not discovered.
  ⚠  Do NOT compute CLs during discovery analyses — set compute_cls=False.

COMBINED SCAN
  When the user asks for both discovery significance AND exclusion in one scan, run
  histogram_significance_and_cls with the default compute_cls=True and report both
  Z (discovery) and CLs (exclusion) explicitly labelled in separate plots.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File inspection
  • inspect_root_data(mode, file_path, tree_name, directory)
      Modes: "summary" (overview of all .root files + tree names), "files" (list .root),
      "trees" (list TTrees in a file), "contents" (all objects), "branches" (branch names + types).
      ALWAYS call mode="summary" first, then mode="branches" before using branch names in cuts.

Counting and selection
  • apply_cut_and_count(file_path, tree_name, cut, weight, file_paths)
      Returns weighted/unweighted event yield passing a C++ cut.
  • generate_cutflow(tree_name, cuts, file_path, file_paths, weight, vector_mode)
      Sequential cutflow table: per-file and combined. Use before optimising cuts.
  • compute_significance(signal_file, tree_name, cut, background_file, background_files, weight)
      Compute S, B and discovery Z (Asimov) after a cut. Discovery only — no CLs.

Histogram operations
  • root_tree_to_histogram(file_path, tree_name, variable, bins, xmin, xmax, cuts, output_root, hist_name)
      Project a TTree branch → TH1 saved to a ROOT file. Use hist_name='sig' / 'bkg' by convention.
  • get_histogram_stats(file_path, hist_name, rebin)
      Mean, RMS, entries from a stored TH1.
  • histogram_integral(file_path, hist_name, x_low, x_high, include_overflow, rebin)
      Integrate TH1 over [x_low, x_high) → value ± uncertainty.
  • histogram_significance_and_cls(file_path, data_name, bkg_name, sig_name, center, window, compute_cls)
      Window-based counting stats from histograms.
        compute_cls=False (default) → discovery only (p0, Z). Use for all discovery analyses.
        compute_cls=True            → discovery + exclusion (p0, Z, CLs, CLs+b, CLb). Use only for exclusion.
      Always set compute_cls explicitly to match the analysis goal.

Statistics and scanning
  • summarize_parameter_scan(parameter_values, series, parameter_name, sort_by, top_n)
      Rank a completed multi-point scan. Pass all named result arrays in the `series` dict.
  • find_optimal_cut(signal_file, tree_name, variable, min_cut, max_cut, step, background_file)
      Scan cut thresholds; returns the cut maximising discovery significance.

Variable definition
  • define_variable(file_path, tree_name, new_var_name, expression, save_file)
      Add a derived branch via C++ expression; save updated tree.
  • define_variable_and_plot(file_path, tree_name, new_variables, variable_to_plot, ...)
      Define branches, apply cuts, plot immediately.

Plotting
  • plot(mode, output_pdf, ...)
      Modes: "hist", "tree", "tree_compare", "hist_compare", "signal_background".
  • plot_2d(mode, output_pdf, ...)  Modes: "hist", "tree".
  • plot_significance_and_cls(parameter_values, significance/cls/y, expected, output_png, output_pdf, ...)
      Array plotting: one primary curve (significance, cls, or y) + optional dashed expected overlay.
      Numeric summary: provide n_sig + n_bkg (+ optional n_obs).
      Output directories must exist; the tool does NOT create them automatically.

Fitting
  • fit_distribution(source, fit_function, file_path, output_plot, ...)
      Fit a ROOT function ('gaus', 'landau', 'pol2', etc.) to a TTree branch or stored TH1.

Export
  • root_tree_to_csv(file_path, tree_name, branches, output_csv, max_vector_size)
      Flatten TTree branches (with vector expansion) to CSV.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED WORKFLOWS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1) File discovery and validation
   inspect_root_data(mode="summary")
   inspect_root_data(mode="branches", file_path=..., tree_name=...)

2) Cut-based discovery
   generate_cutflow → apply_cut_and_count → compute_significance (Z, no CLs)

3) Histogram-based discovery scan (significance vs mass)
   For each mass point:
     histogram_significance_and_cls(..., compute_cls=False)  # discovery only
     → parse Z_exp, Z_obs from "Expected" and "Observed" lines
   summarize_parameter_scan(series={"z_obs": [...], "z_exp": [...]})
   plot_significance_and_cls with significance/p0 arrays and expected overlay.

4) Histogram-based exclusion scan (CLs vs mass)
   For each mass point:
     histogram_significance_and_cls(..., compute_cls=True)  # exclusion
     → parse CLs_exp, CLs_obs from "Expected" and "Observed" lines
   summarize_parameter_scan(series={"cls_obs": [...], "cls_exp": [...]})
   plot_significance_and_cls with cls arrays, draw_cls_threshold=True, expected overlay.

5) Combined discovery + exclusion scan
   For each mass point:
     histogram_significance_and_cls(..., compute_cls=True)  # both
     → parse Z_obs, Z_exp, CLs_obs, CLs_exp
   Run workflows 3 and 4 for their respective plots.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXECUTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- ALWAYS validate files with inspect_root_data before opening them in other tools.
- ALWAYS confirm branch names with inspect_root_data(mode="branches") before writing cuts.
- Choose compute_cls=False for discovery and compute_cls=True for exclusion; NEVER mix them.
- Keep scan arrays index-aligned: build by appending parsed results in scan order.
- Use summarize_parameter_scan to rank results; do NOT rebuild rankings manually.
- Plot tools do NOT create output directories; ensure the target directory exists first.
- Report all results with units. State explicitly whether a quantity is expected (Asimov) or observed.
- One tool call at a time unless the user explicitly requests parallelism.
- Cuts must be valid C++ boolean expressions (e.g. "mass > 120 && mass < 130").

Defaults:
  vector_mode: "any" — vector condition passes if ANY element satisfies it.
  Histogram binning: 40 bins in [0, 100] (always override for physics variables).
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
# RooAgent
Agent for ROOT physics analysis, powered by LLMs and the MCP protocol.

## Requirements

- Python >= 3.10
- CERN ROOT >= 6.34

## Installation

Clone and install locally:

```bash
git clone https://github.com/amanmdesai/RooAgent.git
cd RooAgent
python -m pip install --upgrade pip
pip install -e .
```

Or install from PyPI:

```bash
pip install rooagent
```

## Usage: Claude CLI via MCP

Exposes all RooAgent tools as an MCP server. Claude CLI acts as the LLM and
reasoning layer; the MCP server provides the ROOT tool implementations.
`GITHUB_TOKEN` / `MODEL` are **not** needed.

#### Setup

1. Install the [Claude CLI](https://docs.anthropic.com/en/docs/agents-and-tools/claude-cli/overview)
   and log in:
   ```bash
   npm install -g @anthropic-ai/claude-cli
   claude login
   ```

2. Register the RooAgent MCP server (one-time):
   ```bash
   claude mcp add rooagent -- rooagent-mcp
   ```

   To remove it later:
   ```bash
   claude mcp remove rooagent
   ```

#### Run

Navigate to any directory containing your ROOT files and start Claude:

```bash
cd /path/to/your/root/files
claude
```

`CLAUDE.md` in the project root is read automatically by Claude CLI on every
session. It carries the full system prompt, workflows, defaults, and discipline
rules, so no extra configuration is needed.

#### What it looks like

```
> what root files are in this directory and what trees do they contain?
● inspect_root_data(mode="summary", directory=".")   [rooagent]
  ...
The directory contains: signal.root (tree: Events), background.root (tree: Events)

> plot the pT distribution of signal vs background
● inspect_root_data(mode="branches", ...)            [rooagent]
● plot(mode="signal_background", ...)                [rooagent]
  Saved plot to output/pt_signal_background.png
```

#### Model

The Claude model is whatever is configured in your Claude CLI (e.g. `claude-sonnet-4-5`,
`claude-opus-4`). Change it with `claude config set model <model-name>`.

---

## Available tools

| Category    | Tools |
|-------------|-------|
| Inspection  | `inspect_root_data` |
| Counting    | `apply_cut_and_count`, `generate_cutflow`, `compute_significance`, `compute_efficiency` |
| Statistics  | `histogram_significance_and_cls`, `summarize_parameter_scan` |
| Histograms  | `histogram_integral`, `get_histogram_stats`, `root_tree_to_histogram` |
| Plotting    | `plot`, `plot_2d`, `plot_significance_and_cls` |
| Fitting     | `fit_distribution` |
| Variables   | `define_variable`, `define_variable_and_plot`, `find_optimal_cut` |
| Export      | `root_tree_to_csv` |

Plots and output ROOT files are saved to an `output/` directory in the working directory.

## Dependencies

uproot, pandas, numpy, scipy, matplotlib, langchain-core, mcp (see `pyproject.toml`)



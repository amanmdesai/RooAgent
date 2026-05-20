# RooAgent

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.20249498.svg)](https://doi.org/10.5281/zenodo.20249498)
[![arXiv](https://img.shields.io/badge/arXiv-2605.17318-b31b1b.svg)](https://arxiv.org/abs/2605.17318)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Author:** Aman Desai

Talk to ROOT in plain English. RooAgent lets you run HEP analyses — histograms, event selection, fitting, significance scans — by typing what you want rather than writing code.

Under the hood, an LLM reads your prompt, picks the right analysis tool, fills in the arguments, and calls PyROOT. You get the result; ROOT does the work.

**Two ways to use it:**

- **LangGraph agent** — works with GPT-4.1 (GitHub Copilot) or DeepSeek-V3 (Ollama running locally).
- **MCP server** — drop it into a Claude CLI session; no LangChain required.

## Quick start

Install from PyPI:

```
pip install rooagent
```

Or from source:

```
git clone https://github.com/amanmdesai/RooAgent.git
cd RooAgent
pip install .
```

## Operating modes

### OpenAI / GitHub Copilot (default)

Set these environment variables before running:

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | GitHub Copilot authentication token |
| `MODEL` | *(optional)* Model to use. Default: `openai/gpt-4.1` |
| `ROOAGENT_SEED` | *(optional)* Random seed. Default: `7` |

### Ollama (local models)

For fully offline use with DeepSeek-V3 or any other locally installed Ollama model:

```
git clone https://github.com/amanmdesai/RooAgent.git
cd RooAgent
git checkout ollama_models
pip install .
```

Change the model by setting the `MODEL` environment variable.

### Claude CLI via MCP

```
git clone https://github.com/amanmdesai/RooAgent.git
cd RooAgent
git checkout claude_models
pip install .
claude mcp add rooagent --rooagent-mcp
```

Then navigate to your ROOT files and start Claude:

```
cd /path/to/your/root/files
claude
```

The `CLAUDE.md` in the repository loads automatically and tells Claude which tools are available — nothing else to configure.

## What it can do

- Inspection
- Counting
- Histograms
- Plotting
- Fitting
- Variables
- Statistics
- Export

## Requirements

- Python >= 3.10
- CERN ROOT >= 6.34

## Dependencies

**LangGraph mode:** `langchain`, `langchain-core`, `langgraph`, `langchain-openai` or `langchain-ollama`, `pandas`, `numpy`, `scipy`, `matplotlib`

**MCP mode:** `fastmcp`, `pandas`, `numpy`, `scipy`, `matplotlib`

## How to cite

If you use RooAgent in your work, please cite:

```bibtex
@article{Desai:2026nmx,
    author = "Desai, Aman",
    title = "{RooAgent: An LLM Agent for Root-Based High Energy Physics Analysis}",
    eprint = "2605.17318",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "5",
    year = "2026"
}
```

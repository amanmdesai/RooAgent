from rooagent.src.core import model_with_tools

# Example commands for RJRTree
commands = [
    "Get histogram stats for 'Mtt_R1' in 'signal.root'",
    "Apply cut 'Mtt_R1 > 500' and count in 'signal.root'",
    "Compute significance using 'signal.root' and 'background.root' for cut 'Mtt_R1 > 500'",
    "Define new variable 'M_ratio = Mll_R1/Mtt_R1' and plot 'M_ratio' with cuts ['Mtt_R1 > 500']",
    "Plot variable 'Mll_R1' from signal.root",
    "Compare 'Mtt_R1' in signal.root vs background.root",
    "Export branches ['Mtt_R1','Mll_R1'] to CSV",
    "Fit 'Mtt_R1' in signal.root with Gaussian",
]

for query in commands:
    result = model_with_tools.invoke({"messages": [{"role": "user", "content": query}]})
    print(result["messages"][-1].content)
from typing import List
import ROOT
import pandas as pd
from langchain_core.tools import tool

@tool
def root_tree_to_csv(file_path: str, tree_name: str, branches: List[str], output_csv: str) -> str:
    """
    Convert selected branches from a ROOT TTree into a CSV file.
    """
    df = ROOT.RDataFrame(tree_name, file_path)
    data = df.AsNumpy(branches)
    pandas_df = pd.DataFrame(data)
    pandas_df.to_csv(output_csv, index=False)
    return f"CSV saved to {output_csv}"
from typing import List
import ROOT
import pandas as pd
from langchain_core.tools import tool

@tool
def root_tree_to_csv(file_path: str, tree_name: str, branches: List[str], output_csv: str, max_vector_size: int = 5) -> str:
    """
    Convert ROOT TTree to CSV, flattening vector branches.

    Parameters
    ----------
    file_path : str
        Path to ROOT file.
    tree_name : str
        Name of TTree.
    branches : List[str]
        Branches to extract (scalar or vector).
    output_csv : str
        Output CSV file path.
    max_vector_size : int
        Max number of elements for vector branches (columns will be suffixed _0, _1, ...).
    """
    df = ROOT.RDataFrame(tree_name, file_path)
    data = df.AsNumpy(branches)

    flat_dict = {}
    for branch, array in data.items():
        if array.ndim == 1:  # scalar
            flat_dict[branch] = array
        else:  # vector branch
            for i in range(max_vector_size):
                flat_dict[f"{branch}_{i}"] = [a[i] if i < len(a) else None for a in array]

    pandas_df = pd.DataFrame(flat_dict)
    pandas_df.to_csv(output_csv, index=False)
    return f"Flattened CSV saved to {output_csv}"
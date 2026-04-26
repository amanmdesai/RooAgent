from typing import List
import ROOT
import pandas as pd
from langchain_core.tools import tool

@tool
def root_tree_to_csv(file_path: str, tree_name: str, branches: List[str], output_csv: str, max_vector_size: int = 4) -> str:
    """Export selected TTree branches to a flattened CSV file, expanding vector branches to columns.

    Reads the specified branches from a TTree via ROOT RDataFrame and writes them to a CSV file
    using pandas. Scalar branches produce a single column; vector branches are expanded to
    per-element columns (e.g. 'jets' becomes 'jets_0', 'jets_1', ..., up to `max_vector_size`
    elements). This is useful for inspecting data, feeding downstream Python analysis, or
    archiving small datasets.

    Args:
        file_path (str): ROOT file path containing the TTree.
        tree_name (str): Name of the TTree to export.
        branches (List[str]): Branch names to include. All listed branches must exist in the
            tree; use inspect_root_data(mode='branches') to verify names first.
        output_csv (str): Path to write the output CSV file.
        max_vector_size (int, optional): Maximum number of elements to expand from each vector
            branch (default 4). Vector elements beyond this limit are silently truncated.

    Returns:
        str: "Flattened CSV saved to <path>" on success, or an error string on failure.

    Notes:
        - For large TTrees, export only the branches you need to keep the CSV manageable.
        - Scalar branches are written as-is; vector branch columns are named
          '<branch>_0', '<branch>_1', ... '<branch>_<max_vector_size-1>'.
        - Missing vector elements (when a vector is shorter than max_vector_size) are filled
          with None/NaN in the CSV.
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
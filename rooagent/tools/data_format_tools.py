from typing import List
import ROOT
import pandas as pd

def root_tree_to_csv(file_path: str, tree_name: str, branches: List[str], output_csv: str, max_vector_size: int = 4) -> str:
    """Export TTree branches to a flattened CSV file, expanding vector branches into per-element columns.

    Args:
        file_path: ROOT file path.
        tree_name: TTree name.
        branches: Branch names to export (verify names first with inspect_root_data mode='branches').
        output_csv: Output CSV path.
        max_vector_size: Max elements per vector branch to expand (default 4); excess silently truncated.
    Returns: "Flattened CSV saved to <path>" or error.
    Note: Vector branch 'b' → columns 'b_0', 'b_1', ...; short vectors padded with NaN.
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
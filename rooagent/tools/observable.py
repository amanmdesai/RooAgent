from typing import Optional
import ROOT
from langchain_core.tools import tool

# ================= DEFINE NEW TTree VARIABLE USING RDataFrame =================
@tool
def define_tree_variable(
    file_path: str,
    tree_name: str,
    new_var_name: str,
    expression: str,
    save_file: Optional[str] = None
) -> str:
    """
    Define a new variable in a ROOT TTree using RDataFrame.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree to modify.
    new_var_name : str
        Name of the new variable to define.
    expression : str
        Expression to compute the new variable (ROOT C++/RDataFrame syntax).
        Example: "sqrt(px*px + py*py)" or "pt[0] + pt[1]".
    save_file : str, optional
        Path to save the updated ROOT file. If None, updates the original file in memory.

    Returns
    -------
    str
        Confirmation message with the new variable added.
    """
    # Open ROOT file
    f = ROOT.TFile.Open(file_path, "READ")
    if not f or f.IsZombie():
        return f"Error: Could not open file {file_path}"

    # Create RDataFrame
    rdf = ROOT.RDataFrame(tree_name, file_path)

    # Define new variable
    rdf = rdf.Define(new_var_name, expression)

    # Determine output file
    output_file = save_file if save_file else file_path

    # Save new tree (Snapshot)
    rdf.Snapshot(tree_name, output_file)

    f.Close()
    return f"New variable '{new_var_name}' defined in TTree '{tree_name}' and saved to '{output_file}'"
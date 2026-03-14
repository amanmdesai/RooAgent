from typing import List
import ROOT
import re


def get_vector_branches(file_path: str, tree_name: str) -> List[str]:
    """
    Detect vector branches in a TTree.
    """
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return []

    tree = f.Get(tree_name)
    if not tree:
        f.Close()
        return []

    vector_branches = []

    for branch in tree.GetListOfBranches():
        classname = branch.GetClassName()
        name = branch.GetName()

        if "vector" in classname:
            vector_branches.append(name)

    f.Close()
    return vector_branches


def rewrite_vector_cut(cut: str, vector_vars: List[str], mode: str = "any") -> str:
    """
    Rewrite selection cuts so comparisons on vector branches are wrapped
    with ROOT::VecOps reductions.

    mode:
        "any" -> ROOT::VecOps::Any()
        "all" -> ROOT::VecOps::All()

    Example:
        jet_pt > 30 -> ROOT::VecOps::Any(jet_pt > 30)
    """

    if mode not in ["any", "all"]:
        return cut

    reducer = "Any" if mode == "any" else "All"

    for var in vector_vars:
        pattern = rf"({var}\s*[<>!=]=?\s*[-+]?\d*\.?\d+)"
        matches = re.findall(pattern, cut)

        for match in matches:
            wrapped = f"ROOT::VecOps::{reducer}({match})"
            cut = cut.replace(match, wrapped)

    return cut


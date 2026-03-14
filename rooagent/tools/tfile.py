from langchain_core.tools import tool
import ROOT


@tool
def list_root_file_contents(file_path: str) -> str:
    """
    List all objects stored inside a ROOT file.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file.

    Returns
    -------
    str
        A formatted list of all objects found in the ROOT file including
        TTrees, histograms, directories, and other ROOT objects.
    """

    f = ROOT.TFile.Open(file_path)

    if not f or f.IsZombie():
        return f"Error: Could not open file {file_path}"

    contents = []

    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        contents.append(f"{obj.GetName()} ({obj.ClassName()})")

    f.Close()

    if not contents:
        return "No objects found in file."

    return "ROOT file contents:\n" + "\n".join(contents)


@tool
def list_tree_branches(file_path: str, tree_name: str) -> str:
    """
    List all branches and their types in a ROOT TTree.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the TTree.
    tree_name : str
        Name of the TTree.

    Returns
    -------
    str
        A formatted list of branch names and types in the tree.
    """

    f = ROOT.TFile.Open(file_path)

    if not f or f.IsZombie():
        return f"Error: Could not open file {file_path}"

    tree = f.Get(tree_name)

    if not tree:
        f.Close()
        return f"Error: TTree '{tree_name}' not found."

    branches = []

    for branch in tree.GetListOfBranches():
        name = branch.GetName()
        typename = branch.GetClassName()

        if typename == "":
            leaf = branch.GetLeaf(branch.GetName())
            if leaf:
                typename = leaf.GetTypeName()

        branches.append(f"{name} : {typename}")

    f.Close()

    return "Branches in tree:\n" + "\n".join(branches)
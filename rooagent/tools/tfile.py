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
def list_ttrees(file_path: str) -> str:
    """List TTrees inside a ROOT file."""

    f = ROOT.TFile.Open(file_path)

    if not f or f.IsZombie():
        return f"Could not open {file_path}"

    trees = []

    for key in f.GetListOfKeys():
        obj = key.ReadObj()

        if obj.InheritsFrom("TTree"):
            trees.append(obj.GetName())

    f.Close()

    if not trees:
        return f"No TTrees found in {file_path}"

    return "TTrees:\n" + "\n".join(trees)

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



from langchain_core.tools import tool
import ROOT
import os


@tool
def discover_root_data() -> str:
    """
    Discover ROOT files and TTrees available in the current directory.

    This tool searches the working directory for ROOT files (*.root).
    For each ROOT file found, it lists all TTrees contained in the file.

    If a file contains only one TTree, it can be safely assumed as the
    default tree for analysis.

    Returns
    -------
    str
        A formatted report listing:
        - ROOT files found in the directory
        - TTrees inside each file
        - Suggested default tree if only one is present
    """

    root_files = [f for f in os.listdir(".") if f.endswith(".root")]

    if not root_files:
        return "No ROOT files found in the current directory."

    report = []

    for rf in root_files:

        f = ROOT.TFile.Open(rf)

        if not f or f.IsZombie():
            report.append(f"{rf} : could not open file")
            continue

        trees = []

        for key in f.GetListOfKeys():

            obj = key.ReadObj()

            if obj.InheritsFrom("TTree"):
                trees.append(obj.GetName())

        if not trees:
            report.append(f"{rf} : no TTrees found")

        elif len(trees) == 1:
            report.append(
                f"{rf} : TTree = '{trees[0]}' (single tree — can be used automatically)"
            )

        else:
            tree_list = ", ".join(trees)
            report.append(
                f"{rf} : multiple TTrees found -> {tree_list}"
            )

        f.Close()

    return "ROOT data discovery:\n" + "\n".join(report)
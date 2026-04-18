from langchain_core.tools import tool
import ROOT
import os
from typing import List, Optional


# -------------------------
# Helper utilities
# -------------------------
def _get_root_files(directory: str = ".") -> List[str]:
    """Return list of .root files in the given directory (non-recursive)."""
    try:
        return [f for f in os.listdir(directory) if f.lower().endswith(".root")]
    except Exception:
        return []


def _open_root_file(file_path: str) -> Optional[ROOT.TFile]:
    """
    Safely open a ROOT file. Returns None if it cannot be opened.
    Wrapper around ROOT.TFile.Open to centralize Zombie checks.
    """
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        try:
            if f:
                f.Close()
        except Exception:
            pass
        return None
    return f


def _get_trees(root_file: ROOT.TFile) -> List[str]:
    """Return a list of TTree names inside the given open ROOT file."""
    trees: List[str] = []
    try:
        for key in root_file.GetListOfKeys():
            obj = key.ReadObj()
            # Some objects may be directories -> we still check but only add TTrees
            if obj and hasattr(obj, "InheritsFrom") and obj.InheritsFrom("TTree"):
                trees.append(obj.GetName())
    except Exception:
        # if anything unexpected occurs, return what we've found so far
        pass
    return trees


def _list_objects_recursive(root_dir: ROOT.TDirectory, prefix: str = "") -> List[str]:
    """
    Recursively list top-level objects inside a ROOT directory (TDirectory/TFile).
    Returns strings "path/name (Class)".
    """
    entries: List[str] = []
    try:
        for key in root_dir.GetListOfKeys():
            obj = key.ReadObj()
            if obj is None:
                continue
            name = obj.GetName()
            class_name = obj.ClassName() if hasattr(obj, "ClassName") else type(obj).__name__
            full = f"{prefix}{name} ({class_name})"
            entries.append(full)

            # If it's a directory, recurse
            if obj.InheritsFrom("TDirectory") if hasattr(obj, "InheritsFrom") else False:
                try:
                    entries.extend(_list_objects_recursive(obj, prefix=f"{prefix}{name}/"))
                except Exception:
                    # on recursion error, continue
                    pass
    except Exception:
        pass
    return entries


# -------------------------
# Tools (LangChain)
# -------------------------
@tool
def list_root_files(directory: str = ".") -> str:
    """
    List all ROOT files available in the specified directory.

    Purpose
    -------
    Returns the names of files with the `.root` extension in `directory`.
    This should be the first tool the agent calls when it needs to know
    what data files are available.
    """
    files = _get_root_files(directory)
    if not files:
        return f"No ROOT files found in directory '{directory}'."
    return "ROOT files:\n" + "\n".join(files)


@tool
def list_root_file_contents(file_path: str) -> str:
    """List all objects stored inside a ROOT file (recursively)."""
    f = _open_root_file(file_path)
    if not f:
        return f"Error: Could not open file {file_path}"

    contents = _list_objects_recursive(f)
    f.Close()

    if not contents:
        return "No objects found in file."

    return "ROOT file contents:\n" + "\n".join(contents)


@tool
def list_ttrees(file_path: str) -> str:
    """List TTrees in a ROOT file."""
    f = _open_root_file(file_path)
    if not f:
        return f"Could not open {file_path}"

    trees = _get_trees(f)
    f.Close()

    if not trees:
        return f"No TTrees found in {file_path}"

    return "TTrees:\n" + "\n".join(trees)


@tool
def list_tree_branches(file_path: str, tree_name: str) -> str:
    """List branches and their types in a TTree."""
    f = _open_root_file(file_path)
    if not f:
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


@tool
def discover_root_data(directory: str = ".") -> str:
    """Discover ROOT files and contained TTrees in the current directory."""
    root_files = _get_root_files(directory)
    if not root_files:
        return "No ROOT files found in the current directory."

    report = []
    for rf in root_files:
        f = _open_root_file(rf)
        if not f:
            report.append(f"{rf} : could not open file")
            continue

        trees = _get_trees(f)
        if not trees:
            report.append(f"{rf} : no TTrees found")
        elif len(trees) == 1:
            report.append(f"{rf} : TTree = '{trees[0]}' (single tree — can be used automatically)")
        else:
            report.append(f"{rf} : multiple TTrees found -> {', '.join(trees)}")

        f.Close()

    return "ROOT data discovery:\n" + "\n".join(report)
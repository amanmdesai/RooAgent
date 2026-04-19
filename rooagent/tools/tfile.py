from langchain_core.tools import tool
import ROOT
from typing import List, Optional
from .utils import _get_root_files, _open_root_file, _get_trees, _list_objects_recursive


# -------------------------
# Tools (LangChain)
# -------------------------
@tool
def inspect_root_data(
    mode: str = "summary",
    directory: str = ".",
    file_path: str = "",
    tree_name: str = "",
) -> str:
    """Unified ROOT discovery and inspection tool.

    Use this single tool instead of multiple file/tree helpers.

    Parameters
    ----------
    mode : str
        Inspection mode. Supported values:
        - "files": list `.root` files in `directory`
        - "summary": per-file TTree summary in `directory`
        - "contents": recursive object listing for `file_path`
        - "trees": TTree names for `file_path`
        - "branches": branch list for `file_path` and `tree_name`
    directory : str
        Directory for "files" and "summary" modes.
    file_path : str
        ROOT file path for "contents", "trees", and "branches" modes.
    tree_name : str
        Required only for "branches" mode.

    Returns
    -------
    str
        Human-readable inspection output or a clear error message.
    """
    mode_key = (mode or "summary").strip().lower()

    if mode_key == "files":
        files = _get_root_files(directory)
        if not files:
            return f"No .root files in '{directory}'."
        return "\n".join(files)

    if mode_key == "summary":
        root_files = _get_root_files(directory)
        if not root_files:
            return "No ROOT files found in the current directory."

        report = []
        for rf in root_files:
            f = _open_root_file(rf)
            if not f:
                report.append(f"{rf}: ERR")
                continue

            trees = _get_trees(f)
            if not trees:
                report.append(f"{rf}: no trees")
            else:
                report.append(f"{rf}: {', '.join(trees)}")

            f.Close()

        return "\n".join(report)

    if mode_key == "contents":
        if not file_path:
            return "Error: file_path is required for mode='contents'."
        f = _open_root_file(file_path)
        if not f:
            return f"Error: Could not open file {file_path}"

        contents = _list_objects_recursive(f)
        f.Close()
        if not contents:
            return "No objects found in file."
        return "\n".join(contents)

    if mode_key == "trees":
        if not file_path:
            return "Error: file_path is required for mode='trees'."
        f = _open_root_file(file_path)
        if not f:
            return f"Could not open {file_path}"

        trees = _get_trees(f)
        f.Close()
        if not trees:
            return f"No TTrees in {file_path}"
        return ", ".join(trees)

    if mode_key == "branches":
        if not file_path:
            return "Error: file_path is required for mode='branches'."
        if not tree_name:
            return "Error: tree_name is required for mode='branches'."

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
        return "\n".join(branches)

    return (
        "Error: unsupported mode. Use one of: "
        "files, summary, contents, trees, branches."
    )


def list_root_files(directory: str = ".") -> str:
    """Compatibility wrapper for `inspect_root_data(mode='files')`.

    Parameters
    ----------
    directory : str
        Directory path to search (non-recursive). Defaults to current directory.

    Returns
    -------
    str
        Newline-separated list of `.root` filenames, or an informative message
        when none are found.
    """
    return inspect_root_data.invoke({"mode": "files", "directory": directory})


def list_root_file_contents(file_path: str) -> str:
    """Compatibility wrapper for `inspect_root_data(mode='contents')`.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file to inspect.

    Returns
    -------
    str
        Multi-line listing of contained objects (path and class), or an error
        string if the file cannot be opened.
    """
    return inspect_root_data.invoke({"mode": "contents", "file_path": file_path})


def list_ttrees(file_path: str) -> str:
    """Compatibility wrapper for `inspect_root_data(mode='trees')`.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file to inspect.

    Returns
    -------
    str
        Comma-separated TTree names or an informative error message.
    """
    return inspect_root_data.invoke({"mode": "trees", "file_path": file_path})


def list_tree_branches(file_path: str, tree_name: str) -> str:
    """Compatibility wrapper for `inspect_root_data(mode='branches')`.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the tree.
    tree_name : str
        Name of the TTree inside the file.

    Returns
    -------
    str
        Multi-line text entries of the form "branch : type" or an error string.
    """
    return inspect_root_data.invoke(
        {"mode": "branches", "file_path": file_path, "tree_name": tree_name}
    )


def discover_root_data(directory: str = ".") -> str:
    """Compatibility wrapper for `inspect_root_data(mode='summary')`.

    Parameters
    ----------
    directory : str
        Directory path to search for ROOT files (non-recursive).

    Returns
    -------
    str
        Multi-line report with one line per ROOT file showing contained TTrees
        or an error indicator when files cannot be opened.
    """
    return inspect_root_data.invoke({"mode": "summary", "directory": directory})
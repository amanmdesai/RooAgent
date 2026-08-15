import os
import ROOT
from typing import List, Optional
from .utils import _get_root_files, _open_root_file, _get_trees, _list_objects_recursive


# -------------------------
# Tools (LangChain)
# -------------------------
def inspect_root_data(
    mode: str = "summary",
    directory: str = ".",
    file_path: str = "",
    tree_name: str = "",
) -> str:
    """Inspect ROOT files and their contents. Always call summary first, then branches before writing cuts.

    Args:
        mode: 'summary' — all .root files + TTree names in directory (start here).
              'files' — filenames only.
              'contents' — all objects in file_path.
              'trees' — TTree names in file_path.
              'branches' — branch names+types for tree_name in file_path (required before any cuts).
        directory: Directory to scan (default '.'; used by summary/files modes).
        file_path: ROOT file path (required for contents/trees/branches modes).
        tree_name: TTree name (required for branches mode).

    Returns: Formatted text listing files, trees, branches, or contents; or error string.
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
            f = _open_root_file(os.path.join(directory, rf))
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
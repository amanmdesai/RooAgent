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
    """Inspect ROOT files and enumerate available objects (files, trees, branches, contents).

    This is the entry point for any analysis — call it first to discover what data is available
    before running any other tool. Different modes expose progressively deeper information:

    - **"summary"** (default): List every .root file in `directory` with its top-level TTree
      names. Use this to quickly orient yourself when the user has not specified a file.
    - **"files"**: List all .root file names in `directory` without opening them.
    - **"contents"**: Recursively list ALL objects (histograms, trees, directories) inside a
      specific `file_path` with their ROOT class names. Use this when you need to know what
      histograms exist before calling histogram tools.
    - **"trees"**: List the TTree names inside a specific `file_path`.
    - **"branches"**: List all branch names and their data types inside a given `tree_name` in
      `file_path`. Always call this before constructing cut expressions or variable names to
      avoid typos.

    Args:
        mode (str, optional): Inspection mode — one of 'summary', 'files', 'contents', 'trees',
            'branches'. Default is 'summary'.
        directory (str, optional): Folder to search for .root files; used with modes 'files' and
            'summary'. Defaults to the current working directory.
        file_path (str, optional): Specific ROOT file to inspect; required for modes 'contents',
            'trees', and 'branches'.
        tree_name (str, optional): TTree name inside the file; required for mode 'branches'.

    Returns:
        str: Human-readable listing of files, trees, branches, or object contents depending on
            the selected mode. Returns a descriptive error string when required arguments are
            missing or the file cannot be opened.

    Notes:
        - Start every new analysis with inspect_root_data(mode='summary') to discover available
          ROOT files, then drill down with mode='branches' to confirm variable names.
        - When mode='branches' is needed, both `file_path` and `tree_name` must be provided.
        - The output of this tool is human-readable text only; use other tools to compute
          numeric results.
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
from pathlib import Path
import os

import pytest

ROOT = pytest.importorskip("ROOT")

from rooagent.tools.rdataframe_tools import (  # noqa: E402
    apply_cut_and_count,
    compute_significance,
    define_variable,
    define_variable_and_plot,
    find_optimal_cut,
    generate_cutflow,
)
from rooagent.tools.plot_tools import (  # noqa: E402
    _build_signal_background_ratio,
    plot_1d,
    plot_2d,
)
from rooagent.tools.histogram_tools import get_histogram_stats  # noqa: E402
from rooagent.tools.tfile import (  # noqa: E402
    inspect_root_data,
)
from rooagent.tools.fit_tools import fit_distribution  # noqa: E402
from rooagent.tools.converter import root_tree_to_csv  # noqa: E402
from rooagent.tools.utils import get_vector_branches, rewrite_vector_cut, _parse_file_inputs  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUT_DIR = Path(os.getenv("ROOAGENT_TEST_OUTPUT_DIR", TESTS_DIR / "output"))
SIGNAL_FILE = Path(os.getenv("ROOAGENT_SIGNAL_FILE", TESTS_DIR / "signal.root"))
BACKGROUND_FILE = Path(os.getenv("ROOAGENT_BACKGROUND_FILE", TESTS_DIR / "background.root"))
BACKGROUND2_FILE = Path(os.getenv("ROOAGENT_BACKGROUND2_FILE", TESTS_DIR / "background2.root"))


def _ensure_data_files():
    if not SIGNAL_FILE.exists():
        pytest.skip(f"Missing signal file: {SIGNAL_FILE}")
    if not BACKGROUND_FILE.exists():
        pytest.skip(f"Missing background file: {BACKGROUND_FILE}")
    if not BACKGROUND2_FILE.exists():
        pytest.skip(f"Missing background2 file: {BACKGROUND2_FILE}")


def _artifact_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def _discover_first_tree(file_path: Path) -> str:
    f = ROOT.TFile.Open(str(file_path))
    if not f or f.IsZombie():
        pytest.skip(f"Could not open ROOT file: {file_path}")

    tree_name = None
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TTree"):
            tree_name = obj.GetName()
            break

    f.Close()

    if tree_name is None:
        pytest.skip(f"No TTree found in file: {file_path}")
    return tree_name


def _discover_first_hist(file_path: Path) -> str:
    f = ROOT.TFile.Open(str(file_path))
    if not f or f.IsZombie():
        pytest.skip(f"Could not open ROOT file: {file_path}")

    hist_name = None
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TH1"):
            hist_name = obj.GetName()
            break

    f.Close()

    if hist_name is None:
        pytest.skip(f"No TH1 histogram found in file: {file_path}")
    return hist_name


def _discover_hist_name_per_file(file_path: Path, class_name: str = "TH1") -> str:
    f = ROOT.TFile.Open(str(file_path))
    if not f or f.IsZombie():
        pytest.skip(f"Could not open ROOT file: {file_path}")

    found_name = None
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom(class_name):
            found_name = obj.GetName()
            break

    f.Close()

    if found_name is None:
        pytest.skip(f"No {class_name} object found in file: {file_path}")

    return found_name


def _discover_first_hist2d(file_path: Path) -> str:
    f = ROOT.TFile.Open(str(file_path))
    if not f or f.IsZombie():
        pytest.skip(f"Could not open ROOT file: {file_path}")

    hist_name = None
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TH2"):
            hist_name = obj.GetName()
            break

    f.Close()

    if hist_name is None:
        pytest.skip(f"No TH2 histogram found in file: {file_path}")
    return hist_name


def _discover_numeric_branch(file_path: Path, tree_name: str) -> str:
    f = ROOT.TFile.Open(str(file_path))
    if not f or f.IsZombie():
        pytest.skip(f"Could not open ROOT file: {file_path}")

    tree = f.Get(tree_name)
    if not tree:
        f.Close()
        pytest.skip(f"TTree '{tree_name}' not found in file: {file_path}")

    numeric_leaf_types = {
        "Char_t",
        "UChar_t",
        "Short_t",
        "UShort_t",
        "Int_t",
        "UInt_t",
        "Long64_t",
        "ULong64_t",
        "Float_t",
        "Double_t",
        "Bool_t",
    }

    branch_name = None
    for branch in tree.GetListOfBranches():
        # Skip object/vector branches; prefer scalar leaf-backed branches.
        if branch.GetClassName():
            continue
        leaf = branch.GetLeaf(branch.GetName())
        if leaf and leaf.GetTypeName() in numeric_leaf_types:
            branch_name = branch.GetName()
            break

    f.Close()

    if branch_name is None:
        pytest.skip(f"No scalar numeric branch found in tree '{tree_name}'")
    return branch_name


def _discover_two_numeric_branches(file_path: Path, tree_name: str):
    f = ROOT.TFile.Open(str(file_path))
    if not f or f.IsZombie():
        pytest.skip(f"Could not open ROOT file: {file_path}")

    tree = f.Get(tree_name)
    if not tree:
        f.Close()
        pytest.skip(f"TTree '{tree_name}' not found in file: {file_path}")

    numeric_leaf_types = {
        "Char_t",
        "UChar_t",
        "Short_t",
        "UShort_t",
        "Int_t",
        "UInt_t",
        "Long64_t",
        "ULong64_t",
        "Float_t",
        "Double_t",
        "Bool_t",
    }

    branches = []
    for branch in tree.GetListOfBranches():
        if branch.GetClassName():
            continue
        leaf = branch.GetLeaf(branch.GetName())
        if leaf and leaf.GetTypeName() in numeric_leaf_types:
            branches.append(branch.GetName())

    f.Close()

    if not branches:
        pytest.skip(f"No scalar numeric branch found in tree '{tree_name}'")
    if len(branches) == 1:
        return branches[0], branches[0]
    return branches[0], branches[1]


def _discover_range(file_path: Path, tree_name: str, branch: str):
    df = ROOT.RDataFrame(tree_name, str(file_path))
    min_val = float(df.Min(branch).GetValue())
    max_val = float(df.Max(branch).GetValue())

    if min_val == max_val:
        eps = 1.0 if min_val == 0 else abs(min_val) * 0.1
        return min_val - eps, max_val + eps

    pad = 0.05 * (max_val - min_val)
    return min_val - pad, max_val + pad


def _hist_from_tree(file_path: str, tree_name: str, variable: str, bins: int, xmin: float, xmax: float):
    df = ROOT.RDataFrame(tree_name, file_path)
    hist_ptr = df.Histo1D((f"h_{Path(file_path).stem}_{variable}", variable, bins, xmin, xmax), variable)
    hist = hist_ptr.GetValue()
    hist.SetDirectory(0)
    ROOT.SetOwnership(hist, False)
    return hist


@pytest.fixture(scope="module")
def sample_context():
    _ensure_data_files()
    tree_name = _discover_first_tree(SIGNAL_FILE)
    variable = _discover_numeric_branch(SIGNAL_FILE, tree_name)
    x_branch, y_branch = _discover_two_numeric_branches(SIGNAL_FILE, tree_name)
    hist_name = _discover_first_hist(SIGNAL_FILE)
    xmin, xmax = _discover_range(SIGNAL_FILE, tree_name, variable)
    x2min, x2max = _discover_range(SIGNAL_FILE, tree_name, x_branch)
    y2min, y2max = _discover_range(SIGNAL_FILE, tree_name, y_branch)
    return {
        "signal": str(SIGNAL_FILE),
        "background": str(BACKGROUND_FILE),
        "background2": str(BACKGROUND2_FILE),
        "tests_dir": str(TESTS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "tree": tree_name,
        "variable": variable,
        "x_branch": x_branch,
        "y_branch": y_branch,
        "hist": hist_name,
        "signal_hist": _discover_hist_name_per_file(SIGNAL_FILE, "TH1"),
        "background_hist": _discover_hist_name_per_file(BACKGROUND_FILE, "TH1"),
        "background2_hist": _discover_hist_name_per_file(BACKGROUND2_FILE, "TH1"),
        "xmin": xmin,
        "xmax": xmax,
        "x2min": x2min,
        "x2max": x2max,
        "y2min": y2min,
        "y2max": y2max,
    }


def test_tfile_tools_work(sample_context):
    signal = sample_context["signal"]
    tree = sample_context["tree"]

    ttrees_output = inspect_root_data.invoke({"mode": "trees", "file_path": signal})
    assert tree in ttrees_output

    contents_output = inspect_root_data.invoke({"mode": "contents", "file_path": signal})
    assert "Error:" not in contents_output
    assert len(contents_output.strip()) > 0

    branches_output = inspect_root_data.invoke(
        {"mode": "branches", "file_path": signal, "tree_name": tree}
    )
    assert ":" in branches_output


def test_discover_root_data_tool(sample_context, monkeypatch):
    monkeypatch.chdir(sample_context["tests_dir"])
    output = inspect_root_data.invoke({"mode": "summary", "directory": "."})
    assert "signal.root" in output
    assert "background.root" in output


def test_histogram_stats_tool(sample_context):
    output = get_histogram_stats.invoke(
        {
            "file_path": sample_context["signal"],
            "hist_name": sample_context["hist"],
        }
    )
    assert "Mean:" in output
    assert "RMS:" in output
    assert "Entries:" in output


def test_missing_weight_falls_back_to_count(sample_context):
    output = compute_significance.invoke(
        {
            "signal_file": sample_context["signal"],
            "background_file": sample_context["background"],
            "tree_name": sample_context["tree"],
            "cut": "1==1",
            "weight": "__definitely_missing_weight_branch__",
        }
    )
    assert "Z=" in output


def test_compute_significance_two_backgrounds(sample_context):
    output = compute_significance.invoke(
        {
            "signal_file": sample_context["signal"],
            "background_file": sample_context["background"],
            "background_files": [sample_context["background2"]],
            "tree_name": sample_context["tree"],
            "cut": "1==1",
        }
    )
    assert "S=" in output
    assert "B=" in output
    assert "Z=" in output


def test_compute_significance_with_csv_background_string(sample_context):
    output = compute_significance.invoke(
        {
            "signal_file": sample_context["signal"],
            "background_file": f"{sample_context['background']}, {sample_context['background2']}",
            "tree_name": sample_context["tree"],
            "cut": "1==1",
        }
    )
    assert "S=" in output
    assert "B=" in output
    assert "Z=" in output


def test_apply_cut_and_count_accepts_file_list(sample_context):
    output = apply_cut_and_count.invoke(
        {
            "file_path": sample_context["signal"],
            "file_paths": [sample_context["background"]],
            "tree_name": sample_context["tree"],
            "cut": "1==1",
            "weight": "__missing_weight__",
        }
    )
    assert "files=2" in output


def test_apply_cut_and_count_two_backgrounds(sample_context):
    output = apply_cut_and_count.invoke(
        {
            "file_path": sample_context["signal"],
            "file_paths": [sample_context["background"], sample_context["background2"]],
            "tree_name": sample_context["tree"],
            "cut": "1==1",
        }
    )
    assert "files=3" in output


def test_define_variable_creates_output_root(sample_context):
    output_root = _artifact_path("defined_variable.root")
    output = define_variable.invoke(
        {
            "file_path": sample_context["signal"],
            "tree_name": sample_context["tree"],
            "new_var_name": "test_defined_var",
            "expression": f"{sample_context['variable']} * 1.0",
            "save_file": str(output_root),
        }
    )
    assert "defined and saved" in output
    assert output_root.exists()


def test_define_variable_and_plot_runs(sample_context):
    output_pdf = _artifact_path("defined_variable_plot.pdf")
    output = define_variable_and_plot.invoke(
        {
            "file_path": sample_context["signal"],
            "tree_name": sample_context["tree"],
            "new_variables": {"tmp_var": f"{sample_context['variable']} * 1.0"},
            "variable_to_plot": "tmp_var",
            "bins": 20,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "cuts": ["1==1"],
            "output_file": str(output_pdf),
            "weight": "__missing_weight__",
        }
    )
    assert "Plot saved to" in output
    assert output_pdf.exists()


def test_find_optimal_cut_runs(sample_context):
    output = find_optimal_cut.invoke(
        {
            "signal_file": sample_context["signal"],
            "background_file": sample_context["background"],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "min_cut": sample_context["xmin"],
            "max_cut": sample_context["xmin"],
            "step": 1.0,
            "weight": "__missing_weight__",
        }
    )
    assert "Optimal cut found:" in output
    assert "Significance =" in output


def test_find_optimal_cut_two_backgrounds(sample_context):
    output = find_optimal_cut.invoke(
        {
            "signal_file": sample_context["signal"],
            "background_file": sample_context["background"],
            "background_files": [sample_context["background2"]],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "min_cut": sample_context["xmin"],
            "max_cut": sample_context["xmin"],
            "step": 1.0,
        }
    )
    assert "Background files (2):" in output
    assert "Significance =" in output


def test_generate_cutflow_runs(sample_context):
    output = generate_cutflow.invoke(
        {
            "file_path": sample_context["signal"],
            "file_paths": [sample_context["background"]],
            "tree_name": sample_context["tree"],
            "cuts": ["1==1"],
            "weight": "__missing_weight__",
        }
    )
    assert "Cutflow:" in output
    assert "Initial events (files=2):" in output


def test_generate_cutflow_two_backgrounds(sample_context):
    output = generate_cutflow.invoke(
        {
            "file_path": sample_context["signal"],
            "file_paths": [sample_context["background"], sample_context["background2"]],
            "tree_name": sample_context["tree"],
            "cuts": ["1==1"],
        }
    )
    assert "Initial events (files=3):" in output


def test_converter_root_tree_to_csv(sample_context):
    output_csv = _artifact_path("tree.csv")
    output = root_tree_to_csv.invoke(
        {
            "file_path": sample_context["signal"],
            "tree_name": sample_context["tree"],
            "branches": [sample_context["variable"]],
            "output_csv": str(output_csv),
        }
    )
    assert "CSV saved" in output
    assert output_csv.exists()


def test_fit_tools_run(sample_context):
    tree_fit_pdf = _artifact_path("fit_tree.pdf")
    hist_fit_pdf = _artifact_path("fit_hist.pdf")

    tree_output = fit_distribution.invoke(
        {
            "source": "tree",
            "file_path": sample_context["signal"],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "bins": 20,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "fit_function": "pol1",
            "output_plot": str(tree_fit_pdf),
        }
    )
    assert "chi2/ndf=" in tree_output
    assert tree_fit_pdf.exists()

    hist_output = fit_distribution.invoke(
        {
            "source": "hist",
            "file_path": sample_context["signal"],
            "hist_name": sample_context["hist"],
            "fit_function": "pol1",
            "output_plot": str(hist_fit_pdf),
        }
    )
    assert "chi2/ndf=" in hist_output
    assert hist_fit_pdf.exists()


def test_plot_tools_hist_and_tree_plots(sample_context):
    one_d_pdf = _artifact_path("hist_1d.pdf")
    tree_pdf = _artifact_path("tree_var.pdf")
    compare_pdf = _artifact_path("compare_tree_vars_ratio.pdf")
    same_canvas_pdf = _artifact_path("same_canvas_ratio.pdf")

    out1 = plot_1d.invoke(
        {
            "mode": "hist",
            "file_path": sample_context["signal"],
            "hist_name": sample_context["hist"],
            "output_pdf": str(one_d_pdf),
            "normalize": True,
        }
    )
    assert "Saved 1D histogram" in out1
    assert one_d_pdf.exists()

    out2 = plot_1d.invoke(
        {
            "mode": "tree",
            "file_path": sample_context["signal"],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "bins": 20,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "output_pdf": str(tree_pdf),
            "normalize": True,
        }
    )
    assert "Saved plot to" in out2
    assert tree_pdf.exists()

    out3 = plot_1d.invoke(
        {
            "mode": "tree_compare",
            "file_paths": [sample_context["signal"], sample_context["background"]],
            "tree_name": sample_context["tree"],
            "variables": [sample_context["variable"], sample_context["variable"]],
            "bins": 20,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "legends": ["signal", "background"],
            "output_pdf": str(compare_pdf),
            "normalize": True,
            "show_ratio": True,
        }
    )
    assert "Saved comparison histogram" in out3
    assert compare_pdf.exists()
    assert compare_pdf.stat().st_size > 0

    out4 = plot_1d.invoke(
        {
            "mode": "hist_compare",
            "file_paths": [sample_context["signal"], sample_context["background"]],
            "hist_names": [sample_context["signal_hist"], sample_context["background_hist"]],
            "legends": ["signal", "background"],
            "output_pdf": str(same_canvas_pdf),
            "normalize": True,
            "show_ratio": True,
        }
    )
    assert "Saved combined histogram plot" in out4
    assert same_canvas_pdf.exists()
    assert same_canvas_pdf.stat().st_size > 0


def test_compare_tree_variables_validation(sample_context):
    out = plot_1d.invoke(
        {
            "mode": "tree_compare",
            "file_paths": [sample_context["signal"], sample_context["background"]],
            "tree_name": sample_context["tree"],
            "variables": [sample_context["variable"]],
            "bins": 10,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "legends": ["signal", "background"],
            "output_pdf": str(_artifact_path("compare_invalid.pdf")),
        }
    )
    assert "must have the same length" in out


def test_draw_histograms_same_canvas_validation(sample_context):
    out = plot_1d.invoke(
        {
            "mode": "hist_compare",
            "file_paths": [sample_context["signal"], sample_context["background"]],
            "hist_names": [sample_context["signal_hist"]],
            "legends": ["signal", "background"],
            "output_pdf": str(_artifact_path("same_canvas_invalid.pdf")),
        }
    )
    assert "must have the same length" in out


def test_plot_tools_2d(sample_context):
    tree_2d_pdf = _artifact_path("tree_2d.pdf")

    out_tree = plot_2d.invoke(
        {
            "mode": "tree",
            "file_path": sample_context["signal"],
            "tree_name": sample_context["tree"],
            "x_branch": sample_context["x_branch"],
            "y_branch": sample_context["y_branch"],
            "output_pdf": str(tree_2d_pdf),
            "bins_x": 20,
            "xmin": sample_context["x2min"],
            "xmax": sample_context["x2max"],
            "bins_y": 20,
            "ymin": sample_context["y2min"],
            "ymax": sample_context["y2max"],
        }
    )
    assert "Saved 2D histogram" in out_tree
    assert tree_2d_pdf.exists()

    hist2d_name = _discover_first_hist2d(SIGNAL_FILE)
    hist_2d_pdf = _artifact_path("hist_2d.pdf")
    out_hist = plot_2d.invoke(
        {
            "mode": "hist",
            "file_path": sample_context["signal"],
            "hist_name": hist2d_name,
            "output_pdf": str(hist_2d_pdf),
            "normalize": True,
        }
    )
    assert "Saved 2D histogram" in out_hist
    assert hist_2d_pdf.exists()


def test_plot_signal_vs_backgrounds_creates_pdf(sample_context):
    output_pdf = _artifact_path("signal_vs_backgrounds_ratio.pdf")

    output = plot_1d.invoke(
        {
            "mode": "signal_background",
            "signal_file": sample_context["signal"],
            "background_files": [sample_context["background"]],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "bins": 20,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "output_pdf": str(output_pdf),
            "normalize": True,
            "show_ratio": True,
        }
    )

    assert "Saved signal-vs-background comparison" in output
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0


def test_plot_signal_vs_backgrounds_two_backgrounds(sample_context):
    output_pdf = _artifact_path("signal_vs_two_backgrounds_ratio.pdf")
    output = plot_1d.invoke(
        {
            "mode": "signal_background",
            "signal_file": sample_context["signal"],
            "background_files": [sample_context["background"], sample_context["background2"]],
            "background_labels": ["bkg", "bkg2"],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "bins": 20,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "output_pdf": str(output_pdf),
            "normalize": True,
            "show_ratio": True,
        }
    )
    assert "Saved signal-vs-background comparison" in output
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0


def test_plot_signal_vs_backgrounds_multiple_signals(sample_context):
    output_pdf = _artifact_path("two_signals_vs_background_ratio.pdf")
    output = plot_1d.invoke(
        {
            "mode": "signal_background",
            "signal_file": sample_context["signal"],
            "signal_files": [sample_context["background2"]],
            "signal_labels": ["sig1", "sig2"],
            "background_files": [sample_context["background"]],
            "background_labels": ["bkg"],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "bins": 20,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "output_pdf": str(output_pdf),
            "normalize": True,
            "show_ratio": True,
        }
    )
    assert "Saved signal-vs-background comparison" in output
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0


def test_plot_signal_vs_backgrounds_with_data_markers(sample_context):
    output_pdf = _artifact_path("signal_background_data_markers.pdf")
    output = plot_1d.invoke(
        {
            "mode": "signal_background",
            "signal_file": sample_context["signal"],
            "background_files": [sample_context["background"]],
            "data_file": sample_context["background2"],
            "plot_data": True,
            "data_label": "data",
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "bins": 20,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "output_pdf": str(output_pdf),
            "normalize": True,
        }
    )
    assert "Saved signal-vs-background comparison" in output
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0


def test_signal_background_ratio_uses_sum_of_backgrounds(sample_context):
    signal_hist = _hist_from_tree(
        sample_context["signal"],
        sample_context["tree"],
        sample_context["variable"],
        20,
        sample_context["xmin"],
        sample_context["xmax"],
    )
    background_hist = _hist_from_tree(
        sample_context["background"],
        sample_context["tree"],
        sample_context["variable"],
        20,
        sample_context["xmin"],
        sample_context["xmax"],
    )
    background2_hist = _hist_from_tree(
        sample_context["background2"],
        sample_context["tree"],
        sample_context["variable"],
        20,
        sample_context["xmin"],
        sample_context["xmax"],
    )

    ratio_hist = _build_signal_background_ratio(signal_hist, [background_hist, background2_hist])
    assert ratio_hist is not None

    total_background = background_hist.Clone("total_background_for_test")
    total_background.SetDirectory(0)
    ROOT.SetOwnership(total_background, False)
    total_background.Add(background2_hist)

    checked_bins = 0
    for bin_idx in range(1, ratio_hist.GetNbinsX() + 1):
        denominator = total_background.GetBinContent(bin_idx)
        numerator = signal_hist.GetBinContent(bin_idx)
        if denominator <= 0:
            continue

        expected = numerator / denominator
        actual = ratio_hist.GetBinContent(bin_idx)
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)
        checked_bins += 1

    assert checked_bins > 0


def test_plot_signal_vs_backgrounds_label_validation(sample_context):
    output = plot_1d.invoke(
        {
            "mode": "signal_background",
            "signal_file": sample_context["signal"],
            "background_files": [sample_context["background"], sample_context["background2"]],
            "background_labels": ["only_one"],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "bins": 10,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "output_pdf": str(_artifact_path("signal_vs_backgrounds_invalid.pdf")),
        }
    )
    assert "number of background_labels must match" in output


def test_plot_signal_vs_backgrounds_signal_label_validation(sample_context):
    output = plot_1d.invoke(
        {
            "mode": "signal_background",
            "signal_file": sample_context["signal"],
            "signal_files": [sample_context["background2"]],
            "signal_labels": ["only_one"],
            "background_files": [sample_context["background"]],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "bins": 10,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "output_pdf": str(_artifact_path("signal_vs_backgrounds_invalid_signals.pdf")),
        }
    )
    assert "number of signal_labels must match" in output


def test_plot_signal_vs_backgrounds_data_validation(sample_context):
    output = plot_1d.invoke(
        {
            "mode": "signal_background",
            "signal_file": sample_context["signal"],
            "background_files": [sample_context["background"]],
            "plot_data": True,
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "bins": 10,
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "output_pdf": str(_artifact_path("signal_vs_backgrounds_invalid_data.pdf")),
        }
    )
    assert "data_file must be provided" in output


def test_utils_functions(sample_context):
    vectors = get_vector_branches(sample_context["signal"], sample_context["tree"])
    assert isinstance(vectors, list)

    rewritten = rewrite_vector_cut("x > 1", vectors, mode="any")
    assert isinstance(rewritten, str)


def test_plot_file_input_parser_merges_and_deduplicates():
    parsed = _parse_file_inputs("a.root, b.root", ["b.root", "c.root", ""])
    assert parsed == ["a.root", "b.root", "c.root"]
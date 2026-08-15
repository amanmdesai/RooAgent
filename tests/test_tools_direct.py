from pathlib import Path
import math
import os
import re
import sys
import warnings

import pytest

# Harmless default credentials so imports don't require real API keys.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("GITHUB_TOKEN", "sk-dummy")

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*matplotlib.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*pyparsing.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*fontconfig.*")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings(
    "ignore", message=r"Parameters .* should be specified explicitly", category=UserWarning
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*matplotlib.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:.*pyparsing.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:.*fontconfig.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:Parameters .* should be specified explicitly:UserWarning"
    ),
]

ROOT = pytest.importorskip("ROOT")

from rooagent.tools.rdataframe_tools import (  # noqa: E402
    apply_cut_and_count,
    compute_significance,
    define_variable,
    define_variable_and_plot,
    find_optimal_cut,
    generate_cutflow,
)
from rooagent.tools.rdataframe_tools import root_tree_to_histogram  # noqa: E402
from rooagent.tools.plot_tools import plot, plot_2d, plot_significance_and_cls  # noqa: E402
from rooagent.tools.histogram_tools import get_histogram_stats, histogram_integral  # noqa: E402
from rooagent.tools.stat_tools import (  # noqa: E402
    histogram_significance_and_cls,
    summarize_parameter_scan,
)
from rooagent.tools.tfile_tools import inspect_root_data  # noqa: E402
from rooagent.tools.fit_tools import fit_distribution  # noqa: E402
from rooagent.tools.roofit_tools import fit_model  # noqa: E402
from rooagent.tools.roostats_tools import compute_discovery_significance, compute_upper_limit  # noqa: E402
from rooagent.tools.data_format_tools import root_tree_to_csv  # noqa: E402
from rooagent.tools.utils import (  # noqa: E402
    _get_vector_branches,
    _rewrite_vector_cut,
    _parse_paths,
    _open_root_file,
    _fractional_integral,
    _compute_significance_from_yields,
    _optimal_cut_significance,
    _stat_summary,
)

TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUT_DIR = Path(os.getenv("ROOAGENT_TEST_OUTPUT_DIR", TESTS_DIR / "output"))
SIGNAL_FILE = Path(os.getenv("ROOAGENT_SIGNAL_FILE", TESTS_DIR / "signal.root"))
BACKGROUND_FILE = Path(os.getenv("ROOAGENT_BACKGROUND_FILE", TESTS_DIR / "background.root"))
BACKGROUND2_FILE = Path(os.getenv("ROOAGENT_BACKGROUND2_FILE", TESTS_DIR / "background2.root"))

NUMERIC_LEAF_TYPES = {
    "Char_t", "UChar_t", "Short_t", "UShort_t", "Int_t", "UInt_t",
    "Long64_t", "ULong64_t", "Float_t", "Double_t", "Bool_t",
}


def test_generated_files_have_tree_and_hist():
    for fname in [SIGNAL_FILE, BACKGROUND_FILE, BACKGROUND2_FILE]:
        f = ROOT.TFile.Open(str(fname))
        assert f and not f.IsZombie(), f"Could not open {fname}"
        assert f.Get("Events"), f"No tree 'Events' in {fname}"
        assert f.Get("h1"), f"No histogram 'h1' in {fname}"
        assert f.Get("h2"), f"No 2D histogram 'h2' in {fname}"
        f.Close()


def _ensure_data_files():
    for path in (SIGNAL_FILE, BACKGROUND_FILE, BACKGROUND2_FILE):
        if not path.exists():
            pytest.skip(f"Missing data file: {path}")


def _artifact_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def _discover_first_of_class(file_path: Path, class_name: str) -> str:
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


def _discover_first_tree(file_path: Path) -> str:
    return _discover_first_of_class(file_path, "TTree")


def _discover_first_hist(file_path: Path) -> str:
    return _discover_first_of_class(file_path, "TH1")


def _discover_first_hist2d(file_path: Path) -> str:
    return _discover_first_of_class(file_path, "TH2")


def _discover_numeric_branches(file_path: Path, tree_name: str, count: int = 1):
    f = ROOT.TFile.Open(str(file_path))
    if not f or f.IsZombie():
        pytest.skip(f"Could not open ROOT file: {file_path}")

    tree = f.Get(tree_name)
    if not tree:
        f.Close()
        pytest.skip(f"TTree '{tree_name}' not found in file: {file_path}")

    branches = []
    for branch in tree.GetListOfBranches():
        if branch.GetClassName():
            continue  # skip vector/object branches
        leaf = branch.GetLeaf(branch.GetName())
        if leaf and leaf.GetTypeName() in NUMERIC_LEAF_TYPES:
            branches.append(branch.GetName())
    f.Close()

    if not branches:
        pytest.skip(f"No scalar numeric branch found in tree '{tree_name}'")
    if len(branches) < count:
        branches = branches * count
    return branches[0] if count == 1 else tuple(branches[:count])


def _discover_range(file_path: Path, tree_name: str, branch: str):
    df = ROOT.RDataFrame(tree_name, str(file_path))
    min_val = float(df.Min(branch).GetValue())
    max_val = float(df.Max(branch).GetValue())

    if min_val == max_val:
        eps = 1.0 if min_val == 0 else abs(min_val) * 0.1
        return min_val - eps, max_val + eps

    pad = 0.05 * (max_val - min_val)
    return min_val - pad, max_val + pad


@pytest.fixture(scope="module")
def sample_context():
    _ensure_data_files()
    tree_name = _discover_first_tree(SIGNAL_FILE)
    variable = _discover_numeric_branches(SIGNAL_FILE, tree_name)
    x_branch, y_branch = _discover_numeric_branches(SIGNAL_FILE, tree_name, count=2)
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
        "signal_hist": _discover_first_of_class(SIGNAL_FILE, "TH1"),
        "background_hist": _discover_first_of_class(BACKGROUND_FILE, "TH1"),
        "background2_hist": _discover_first_of_class(BACKGROUND2_FILE, "TH1"),
        "xmin": xmin, "xmax": xmax,
        "x2min": x2min, "x2max": x2max,
        "y2min": y2min, "y2max": y2max,
    }


def test_tfile_tools_work(sample_context):
    signal = sample_context["signal"]
    tree = sample_context["tree"]

    assert tree in inspect_root_data.invoke({"mode": "trees", "file_path": signal})

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
        {"file_path": sample_context["signal"], "hist_name": sample_context["hist"]}
    )
    assert "Mean:" in output
    assert "RMS:" in output
    assert "Entries:" in output


def test_histogram_integral_tool(sample_context):
    f = ROOT.TFile.Open(sample_context["signal"])
    h = f.Get(sample_context["hist"])
    ax = h.GetXaxis()
    x_low, x_high = ax.GetXmin(), ax.GetXmax()
    f.Close()

    output = histogram_integral.invoke(
        {
            "file_path": sample_context["signal"],
            "hist_name": sample_context["hist"],
            "x_low": x_low,
            "x_high": x_high,
        }
    )
    assert "Integral" in output
    assert "bins=" in output


def test_histogram_integral_invalid_range_returns_error(sample_context):
    output = histogram_integral.invoke(
        {
            "file_path": sample_context["signal"],
            "hist_name": sample_context["hist"],
            "x_low": 1.0,
            "x_high": 1.0,
        }
    )
    assert "Error" in output


def test_histogram_significance_and_cls_uses_exclusion_tail(tmp_path):
    """CDF-based CLs stays in [0, 1]; a strong deficit (n_obs=3 vs B=10, S=5) excludes the signal."""
    root_file = tmp_path / "cls_input.root"
    f = ROOT.TFile.Open(str(root_file), "RECREATE")

    hbkg = ROOT.TH1F("bkg", "bkg", 4, 0.0, 4.0)
    hsig = ROOT.TH1F("sig", "sig", 4, 0.0, 4.0)
    hdata = ROOT.TH1F("data", "data", 4, 0.0, 4.0)
    for _ in range(10):
        hbkg.Fill(1.5)
    for _ in range(5):
        hsig.Fill(1.5)
    for _ in range(3):
        hdata.Fill(1.5)
    hbkg.Write()
    hsig.Write()
    hdata.Write()
    f.Close()

    output = histogram_significance_and_cls.invoke(
        {
            "file_path": str(root_file),
            "data_name": "data",
            "bkg_name": "bkg",
            "sig_name": "sig",
            "center": 1.5,
            "window": 0.5,
            "compute_cls": True,
        }
    )

    assert "Signal=sig" in output
    assert "Center=1.5" in output
    assert "Expected(S+B Asimov):" in output, output
    assert "Observed:" in output, output

    p0_values = re.findall(r"p0=([0-9.eE+\-]+)", output)
    z_values = re.findall(r"Z=([0-9.eE+\-]+)", output)
    cls_values = re.findall(r"CLs=([0-9.eE+\-]+)", output)
    clspb_values = re.findall(r"CLs\+b=([0-9.eE+\-]+)", output)
    clb_values = re.findall(r"CLb=([0-9.eE+\-]+)", output)
    assert len(p0_values) == len(z_values) == len(cls_values) == len(clspb_values) == len(clb_values) == 2

    for val_str in cls_values:
        assert 0.0 <= float(val_str) <= 1.0, f"CLs out of range [0,1]: {val_str} ({output})"

    observed_cls = float(cls_values[1])
    assert observed_cls < 0.05, f"Expected signal excluded (CLs < 0.05), got {observed_cls}"

    for clspb_str, clb_str in zip(clspb_values, clb_values):
        assert float(clspb_str) <= float(clb_str) + 1e-9, "CLs+b > CLb violates CDF-based CLs guarantee"


def test_cls_is_always_leq_one():
    """CLs/CLs+b/CLb stay in [0,1] and CLs+b <= CLb across deficit/excess/edge n_obs regimes."""
    from rooagent.tools.utils import _counting_cls_poisson_fallback

    test_cases = [
        (0, 10, 5), (3, 10, 5), (10, 10, 5), (12, 10, 5),
        (15, 10, 5), (20, 10, 5), (0, 0, 1), (5, 5, 0),
    ]
    for n_obs, n_bkg, n_sig in test_cases:
        cls, clspb, clb = _counting_cls_poisson_fallback(n_obs, n_bkg, n_sig)
        assert 0.0 <= cls <= 1.0
        assert 0.0 <= clspb <= 1.0
        assert 0.0 <= clb <= 1.0
        assert clspb <= clb + 1e-9, f"CLs+b > CLb for n_obs={n_obs}, n_bkg={n_bkg}, n_sig={n_sig}"


def test_cls_deficit_excludes_signal():
    from rooagent.tools.utils import _counting_cls_poisson_fallback
    cls, _, _ = _counting_cls_poisson_fallback(0, 20.0, 10.0)
    assert cls < 0.01


def test_cls_asimov_point_not_excluded():
    """At the Asimov point (n_obs = B+S) the signal is only borderline compatible, so CLs > 0.05."""
    from rooagent.tools.utils import _counting_cls_poisson_fallback
    cls, _, _ = _counting_cls_poisson_fallback(15, 10.0, 5.0)
    assert cls > 0.05


def test_root_tree_to_histogram_and_stat_tools(sample_context, tmp_path):
    signal = sample_context["signal"]
    tree = sample_context["tree"]
    variable = sample_context["variable"]
    xmin, xmax = sample_context["xmin"], sample_context["xmax"]
    out_file = tmp_path / "sig_hist.root"

    conv_out = root_tree_to_histogram.invoke(
        {
            "file_path": signal,
            "tree_name": tree,
            "variable": variable,
            "bins": 50,
            "xmin": xmin,
            "xmax": xmax,
            "output_root": str(out_file),
            "hist_name": "sig",
        }
    )
    assert "Saved histogram" in conv_out

    fbkg = ROOT.TFile.Open(sample_context["background"])
    hbkg = fbkg.Get(sample_context["background_hist"])
    fout = ROOT.TFile.Open(str(out_file), "UPDATE")
    hbkg_clone = hbkg.Clone("bkg")
    fout.cd()
    hbkg_clone.Write()
    fout.Close()
    fbkg.Close()

    out = histogram_significance_and_cls.invoke(
        {
            "file_path": str(out_file),
            "data_name": "",
            "bkg_name": "bkg",
            "sig_name": "sig",
            "center": (xmin + xmax) / 2.0,
            "window": (xmax - xmin) / 10.0,
        }
    )
    assert "Expected(S+B Asimov):" in out


@pytest.mark.parametrize("case", ["missing_weight", "background_files_list", "csv_string"])
def test_compute_significance_variants(sample_context, case):
    args = {"signal_file": sample_context["signal"], "tree_name": sample_context["tree"], "cut": "1==1"}
    if case == "missing_weight":
        args["background_file"] = sample_context["background"]
        args["weight"] = "__definitely_missing_weight_branch__"
    elif case == "background_files_list":
        args["background_file"] = sample_context["background"]
        args["background_files"] = [sample_context["background2"]]
    else:
        args["background_file"] = f"{sample_context['background']}, {sample_context['background2']}"

    output = compute_significance.invoke(args)
    assert "Z=" in output
    if case != "missing_weight":
        assert "S=" in output
        assert "B=" in output


@pytest.mark.parametrize("n_extra_files, expected", [(1, "files=2"), (2, "files=3")])
def test_apply_cut_and_count_multi_file(sample_context, n_extra_files, expected):
    file_paths = [sample_context["background"], sample_context["background2"]][:n_extra_files]
    output = apply_cut_and_count.invoke(
        {
            "file_path": sample_context["signal"],
            "file_paths": file_paths,
            "tree_name": sample_context["tree"],
            "cut": "1==1",
        }
    )
    assert expected in output


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


def _optimal_cut_case_single_background(ctx):
    return dict(max_cut=ctx["xmin"], step=1.0, weight="__missing_weight__"), \
        ["Optimal cut found:", "Significance ="]


def _optimal_cut_case_two_backgrounds(ctx):
    return dict(max_cut=ctx["xmin"], step=1.0, background_files=[ctx["background2"]]), \
        ["Optimal cut found:", "Significance =", "Background files (2):"]


def _optimal_cut_case_invalid_step(ctx):
    return dict(max_cut=ctx["xmax"], step=0.0), ["step must be > 0"]


@pytest.mark.parametrize(
    "build_case",
    [_optimal_cut_case_single_background, _optimal_cut_case_two_backgrounds, _optimal_cut_case_invalid_step],
    ids=["single_background", "two_backgrounds", "invalid_step"],
)
def test_find_optimal_cut(sample_context, build_case):
    extra, expect = build_case(sample_context)
    output = find_optimal_cut.invoke(
        {
            "signal_file": sample_context["signal"],
            "background_file": sample_context["background"],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "min_cut": sample_context["xmin"],
            **extra,
        }
    )
    for expected in expect:
        assert expected in output


@pytest.mark.parametrize("n_extra_files, expected", [(1, "files=2"), (2, "files=3")])
def test_generate_cutflow_multi_file(sample_context, n_extra_files, expected):
    file_paths = [sample_context["background"], sample_context["background2"]][:n_extra_files]
    output = generate_cutflow.invoke(
        {
            "file_path": sample_context["signal"],
            "file_paths": file_paths,
            "tree_name": sample_context["tree"],
            "cuts": ["1==1"],
        }
    )
    assert "Cutflow (per-file):" in output
    assert f"Initial events ({expected}):" in output


def test_generate_cutflow_per_file_counts_match_apply_cut_and_count(sample_context):
    files = [sample_context["signal"], sample_context["background"], sample_context["background2"]]

    output = generate_cutflow.invoke(
        {
            "file_path": sample_context["signal"],
            "file_paths": [sample_context["background"], sample_context["background2"]],
            "tree_name": sample_context["tree"],
            "cuts": ["1==1"],
        }
    )

    for f in files:
        assert Path(f).name in output

    for f in files:
        single = apply_cut_and_count.invoke(
            {"file_path": f, "tree_name": sample_context["tree"], "cut": "1==1"}
        )
        m = re.search(r"yield(?:\([^)]*\))?=([0-9.eE+\-]+)", single)
        assert m, f"apply_cut_and_count output malformed: {single}"
        assert m.group(1) in output, f"Value for {f} not found in generate_cutflow output"


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


def test_converter_root_tree_to_csv_flattens_vector_branches(tmp_path):
    root_file = tmp_path / "vectors.root"
    f = ROOT.TFile.Open(str(root_file), "RECREATE")
    tree = ROOT.TTree("Events", "Events")
    values = ROOT.std.vector("float")()
    tree.Branch("jets_pt", values)
    for row in ([10.0, 20.0], [30.0], []):
        values.clear()
        for value in row:
            values.push_back(value)
        tree.Fill()
    tree.Write()
    f.Close()

    output_csv = tmp_path / "vectors.csv"
    output = root_tree_to_csv.invoke(
        {
            "file_path": str(root_file),
            "tree_name": "Events",
            "branches": ["jets_pt"],
            "output_csv": str(output_csv),
            "max_vector_size": 2,
        }
    )

    assert "CSV saved" in output
    csv_text = output_csv.read_text()
    assert "jets_pt_0,jets_pt_1" in csv_text
    assert "10.0,20.0" in csv_text


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


def test_roofit_tools_run(sample_context):
    tree_fit_pdf = _artifact_path("roofit_tree_gauss.pdf")
    hist_fit_pdf = _artifact_path("roofit_hist_gauss_expo.pdf")

    tree_output = fit_model.invoke(
        {
            "source": "tree",
            "signal_shape": "gauss",
            "file_path": sample_context["signal"],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "xmin": sample_context["xmin"],
            "xmax": sample_context["xmax"],
            "bins": 20,
            "mean": sample_context["xmin"] + (sample_context["xmax"] - sample_context["xmin"]) / 2,
            "sigma": (sample_context["xmax"] - sample_context["xmin"]) / 4,
            "output_plot": str(tree_fit_pdf),
        }
    )
    assert "RooFit ML fit" in tree_output
    assert "status=" in tree_output
    assert tree_fit_pdf.exists()

    hist_output = fit_model.invoke(
        {
            "source": "hist",
            "signal_shape": "gauss",
            "background_shape": "expo",
            "file_path": sample_context["signal"],
            "hist_name": sample_context["hist"],
            "nsig_init": 5.0,
            "nbkg_init": 5.0,
            "output_plot": str(hist_fit_pdf),
        }
    )
    assert "RooFit ML fit" in hist_output
    assert "nsig=" in hist_output
    assert "nbkg=" in hist_output
    assert hist_fit_pdf.exists()

    bad_output = fit_model.invoke(
        {
            "source": "tree",
            "signal_shape": "not_a_shape",
            "file_path": sample_context["signal"],
            "tree_name": sample_context["tree"],
            "variable": sample_context["variable"],
            "output_plot": str(_artifact_path("roofit_bad.pdf")),
        }
    )
    assert bad_output.startswith("Error:")


def test_roostats_tools_run(sample_context):
    common_args = {
        "source": "tree",
        "signal_shape": "gauss",
        "background_shape": "expo",
        "file_path": sample_context["signal"],
        "tree_name": sample_context["tree"],
        "variable": sample_context["variable"],
        "xmin": sample_context["xmin"],
        "xmax": sample_context["xmax"],
        "mean": sample_context["xmin"] + (sample_context["xmax"] - sample_context["xmin"]) / 2,
        "sigma": (sample_context["xmax"] - sample_context["xmin"]) / 4,
        "nsig_init": 5.0,
        "nbkg_init": 5.0,
    }

    discovery_output = compute_discovery_significance.invoke(common_args)
    assert "RooStats discovery" in discovery_output
    assert "p0=" in discovery_output
    assert "Z=" in discovery_output

    ul_output = compute_upper_limit.invoke({**common_args, "confidence_level": 0.95})
    assert "RooStats CLs upper limit" in ul_output
    assert "Observed UL(nsig)=" in ul_output
    assert "Expected UL(median)=" in ul_output

    bad_discovery = compute_discovery_significance.invoke({**common_args, "background_shape": ""})
    assert bad_discovery.startswith("Error:")

    bad_ul = compute_upper_limit.invoke({**common_args, "signal_shape": "not_a_shape"})
    assert bad_ul.startswith("Error:")


def test_plot_tools_hist_and_tree_plots(sample_context):
    one_d_pdf = _artifact_path("hist_1d.pdf")
    tree_pdf = _artifact_path("tree_var.pdf")
    compare_pdf = _artifact_path("compare_tree_vars_ratio.pdf")
    same_canvas_pdf = _artifact_path("same_canvas_ratio.pdf")

    out1 = plot.invoke(
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

    out2 = plot.invoke(
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

    out3 = plot.invoke(
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

    out4 = plot.invoke(
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


def test_plot_significance_and_cls_success_cases(tmp_path):
    # basic y-array
    out = tmp_path / "sig_cls.png"
    result = plot_significance_and_cls.invoke(
        {
            "parameter_values": [250, 280, 265],
            "parameter_label": "Width",
            "y": [2.43, 2.25, 2.11],
            "y_label": "Significance (Z)",
            "output_png": str(out),
        }
    )
    assert "Saved" in result
    assert out.exists() and out.stat().st_size > 0

    # `significance` and `cls` convenience kwargs each produce a file, same as `y`
    parameter_values = [100, 200, 300, 400, 500]
    vals = [1.0, 2.0, 3.0, 1.5, 0.5]
    for kwarg, suffix in [("significance", "sig"), ("cls", "cls")]:
        out = tmp_path / f"{suffix}.png"
        result = plot_significance_and_cls.invoke(
            {"parameter_values": parameter_values, kwarg: vals, "output_png": str(out)}
        )
        assert "Saved" in result, f"{kwarg}: {result}"
        assert out.exists()

    # `expected` overlays a second dashed curve
    out = tmp_path / "sig_with_expected.png"
    result = plot_significance_and_cls.invoke(
        {
            "parameter_values": [100, 150, 200],
            "significance": [1.2, 2.1, 3.4],
            "expected": [1.0, 1.8, 2.9],
            "output_png": str(out),
        }
    )
    assert "Saved" in result
    assert out.exists() and out.stat().st_size > 0

    # both output_png and output_pdf, if given, are honored and reported
    result = plot_significance_and_cls.invoke(
        {
            "parameter_values": [100, 200, 300],
            "significance": [1.0, 2.0, 3.0],
            "output_png": str(tmp_path / "out.png"),
            "output_pdf": str(tmp_path / "out.pdf"),
        }
    )
    assert str(tmp_path / "out.png") in result
    assert str(tmp_path / "out.pdf") in result
    assert (tmp_path / "out.png").exists()
    assert (tmp_path / "out.pdf").exists()

    # the tool does NOT create missing directories; caller is responsible
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    out = existing_dir / "sig.png"
    result = plot_significance_and_cls.invoke(
        {"parameter_values": [100, 200, 300], "significance": [1.0, 2.0, 3.0], "output_png": str(out)}
    )
    assert "Saved" in result
    assert out.exists()


def test_plot_significance_and_cls_no_output_uses_default_name(tmp_path):
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = plot_significance_and_cls.invoke(
            {"parameter_values": [100, 200], "significance": [1.0, 2.0], "output_png": "", "output_pdf": ""}
        )
        out = tmp_path / "significance_cls.png"
        assert "Saved" in result
        assert out.exists() and out.stat().st_size > 0
    finally:
        os.chdir(cwd)


PLOT_SIG_CLS_ERROR_CASES = [
    (
        "expected_length_mismatch",
        dict(parameter_values=[100, 200, 300], y=[0.1, 0.05, 0.02], expected=[0.12, 0.06]),
        "Error",
        False,
    ),
    (
        "length_mismatch",
        dict(parameter_values=[100, 200, 300, 400, 500], significance=[1.0, 2.0, 3.0]),
        "Error: parameter_values and y-data must have the same length.",
        True,
    ),
    (
        "none_values",
        dict(parameter_values=[100, 200, 300, 400, 500], significance=[1.0, None, 2.0, None, 3.0]),
        "Error: parameter_values and y-data must be finite numeric values.",
        True,
    ),
    (
        "multiple_series",
        dict(parameter_values=[225.0, 250.0], significance=[1.7, 2.0], cls=[0.1, 0.2]),
        "Error: provide exactly one of significance, cls, or y when plotting arrays.",
        True,
    ),
    (
        "no_y_data",
        dict(parameter_values=[100, 200]),
        "Error",
        False,
    ),
]


@pytest.mark.parametrize(
    "kwargs, expected, exact",
    [c[1:] for c in PLOT_SIG_CLS_ERROR_CASES],
    ids=[c[0] for c in PLOT_SIG_CLS_ERROR_CASES],
)
def test_plot_significance_and_cls_errors(tmp_path, kwargs, expected, exact):
    out = tmp_path / "scan.png"
    result = plot_significance_and_cls.invoke({**kwargs, "output_png": str(out)})
    assert (result == expected) if exact else (expected in result)
    assert not out.exists()


def test_compare_tree_variables_validation(sample_context):
    out = plot.invoke(
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
    out = plot.invoke(
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


def _sig_bkg_case_basic(ctx):
    return dict(background_files=[ctx["background"]], normalize=True, show_ratio=True)


def _sig_bkg_case_stack_bkg_only(ctx):
    return dict(
        background_files=[ctx["background"]], data_file=ctx["background2"], plot_data=True,
        normalize=False, stack_backgrounds_only=True,
    )


def _sig_bkg_case_two_backgrounds(ctx):
    return dict(
        background_files=[ctx["background"], ctx["background2"]],
        background_labels=["bkg", "bkg2"], normalize=True, show_ratio=True,
    )


def _sig_bkg_case_multiple_signals(ctx):
    return dict(
        signal_files=[ctx["background2"]], signal_labels=["sig1", "sig2"],
        background_files=[ctx["background"]], background_labels=["bkg"],
        normalize=True, show_ratio=True,
    )


def _sig_bkg_case_data_markers(ctx):
    return dict(
        background_files=[ctx["background"]], data_file=ctx["background2"],
        plot_data=True, data_label="data", normalize=True,
    )


SIGNAL_BACKGROUND_CASES = [
    ("basic", "signal_vs_backgrounds_ratio.pdf", _sig_bkg_case_basic),
    ("stack_bkg_only", "signal_background_stack_bkg_only.pdf", _sig_bkg_case_stack_bkg_only),
    ("two_backgrounds", "signal_vs_two_backgrounds_ratio.pdf", _sig_bkg_case_two_backgrounds),
    ("multiple_signals", "two_signals_vs_background_ratio.pdf", _sig_bkg_case_multiple_signals),
    ("data_markers", "signal_background_data_markers.pdf", _sig_bkg_case_data_markers),
]


@pytest.mark.parametrize(
    "filename, build_kwargs",
    [(f, b) for _, f, b in SIGNAL_BACKGROUND_CASES],
    ids=[c[0] for c in SIGNAL_BACKGROUND_CASES],
)
def test_plot_signal_vs_backgrounds(sample_context, filename, build_kwargs):
    output_pdf = _artifact_path(filename)
    kwargs = build_kwargs(sample_context)
    kwargs.update(
        mode="signal_background",
        signal_file=sample_context["signal"],
        tree_name=sample_context["tree"],
        variable=sample_context["variable"],
        bins=20,
        xmin=sample_context["xmin"],
        xmax=sample_context["xmax"],
        output_pdf=str(output_pdf),
    )

    output = plot.invoke(kwargs)
    assert "Saved signal-vs-background comparison" in output
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0


def _sig_bkg_invalid_label_mismatch(ctx):
    return dict(background_files=[ctx["background"], ctx["background2"]], background_labels=["only_one"]), \
        "number of background_labels must match"


def _sig_bkg_invalid_signal_label_mismatch(ctx):
    return dict(
        signal_files=[ctx["background2"]], signal_labels=["only_one"],
        background_files=[ctx["background"]],
    ), "number of signal_labels must match"


def _sig_bkg_invalid_missing_data_file(ctx):
    return dict(background_files=[ctx["background"]], plot_data=True), "data_file must be provided"


SIGNAL_BACKGROUND_VALIDATION_CASES = [
    ("label_mismatch", "signal_vs_backgrounds_invalid.pdf", _sig_bkg_invalid_label_mismatch),
    ("signal_label_mismatch", "signal_vs_backgrounds_invalid_signals.pdf", _sig_bkg_invalid_signal_label_mismatch),
    ("missing_data_file", "signal_vs_backgrounds_invalid_data.pdf", _sig_bkg_invalid_missing_data_file),
]


@pytest.mark.parametrize(
    "filename, build_case",
    [(f, b) for _, f, b in SIGNAL_BACKGROUND_VALIDATION_CASES],
    ids=[c[0] for c in SIGNAL_BACKGROUND_VALIDATION_CASES],
)
def test_plot_signal_vs_backgrounds_validation(sample_context, filename, build_case):
    kwargs, expected = build_case(sample_context)
    kwargs.update(
        mode="signal_background",
        signal_file=sample_context["signal"],
        tree_name=sample_context["tree"],
        variable=sample_context["variable"],
        bins=10,
        xmin=sample_context["xmin"],
        xmax=sample_context["xmax"],
        output_pdf=str(_artifact_path(filename)),
    )

    output = plot.invoke(kwargs)
    assert expected in output


def test_utils_functions(sample_context):
    vectors = _get_vector_branches(sample_context["signal"], sample_context["tree"])
    assert isinstance(vectors, list)

    rewritten = _rewrite_vector_cut("x > 1", vectors, mode="any")
    assert isinstance(rewritten, str)


def test_plot_file_input_parser_merges_and_deduplicates():
    parsed = _parse_paths("a.root, b.root", ["b.root", "c.root", ""])
    assert parsed == [
        str((PROJECT_ROOT / "a.root").resolve()),
        str((PROJECT_ROOT / "b.root").resolve()),
        str((PROJECT_ROOT / "c.root").resolve()),
    ]


def test_stat_tools_with_generated_files():
    output = histogram_significance_and_cls.invoke({
        "file_path": str(SIGNAL_FILE),
        "bkg_name": "h1",
        "sig_name": "h1",
        "center": 0.0,
        "window": 5.0,
        "compute_cls": True,
    })
    assert "p0=" in output
    assert "CLs=" in output
    assert "CLs+b=" in output
    assert "CLb=" in output


def test_stat_tools_missing_histogram():
    output = histogram_significance_and_cls.invoke({
        "file_path": str(SIGNAL_FILE),
        "bkg_name": "notfound",
        "sig_name": "h1",
        "center": 0.0,
        "window": 5.0,
    })
    assert "missing histograms" in output or "Error" in output


def test_open_root_file_missing_returns_none(tmp_path):
    assert _open_root_file(str(tmp_path / "missing.root")) is None


def test_fractional_integral_partial_bins():
    # 4 bins [0,1,2,3,4] with contents [10,20,30,40]; window [0.75,1.75] overlaps
    # bin1 by 0.25 and bin2 by 0.75 => 10*0.25 + 20*0.75 = 17.5
    h = ROOT.TH1F("h", "h", 4, 0.0, 4.0)
    h.SetBinContent(1, 10.0)
    h.SetBinContent(2, 20.0)
    h.SetBinContent(3, 30.0)
    h.SetBinContent(4, 40.0)
    val = _fractional_integral(h, center=1.25, window=0.5)
    assert math.isclose(val, 17.5, rel_tol=1e-9)


def test_significance_from_yields_b_zero():
    assert math.isinf(_compute_significance_from_yields(5.0, 0.0))


def test_significance_from_yields_uses_asimov_formula():
    s, b = 5.0, 10.0
    expected = math.sqrt(2.0 * ((s + b) * math.log1p(s / b) - s))
    assert math.isclose(_compute_significance_from_yields(s, b), expected, rel_tol=1e-12)


def test_optimal_cut_significance():
    sig = _optimal_cut_significance(16.0, 9.0)
    assert math.isclose(sig, 16.0 / 5.0, rel_tol=1e-9)


def test_stat_summary_basic_format_and_probabilities():
    out = _stat_summary(10.0, 5.0)
    assert "Expected(S+B Asimov):" in out
    p0_values = re.findall(r"p0=([0-9.eE+\-]+)", out)
    assert len(p0_values) >= 1
    for p in p0_values:
        assert 0.0 <= float(p) <= 1.0


def test_summarize_parameter_scan_supports_width_and_cls_only():
    output = summarize_parameter_scan.invoke(
        {
            "parameter_values": [0.10, 0.20, 0.30],
            "parameter_name": "width",
            "series": {"cls": [0.20, 0.05, 0.10]},
            "top_n": 2,
        }
    )
    assert "Ranking (by cls, ascending):" in output
    assert "Best point: width = 0.2, cls = 0.05" in output
    assert "width = 0.3, cls = 0.1" in output


def test_summarize_parameter_scan_prefers_significance_when_present():
    output = summarize_parameter_scan.invoke(
        {
            "parameter_values": [1.0, 2.0],
            "parameter_name": "width",
            "series": {"z_obs": [1.5, 2.5], "cls": [0.20, 0.30]},
        }
    )
    assert "Ranking (by z_obs, descending):" in output
    assert "Best point: width = 2, z_obs = 2.5, cls = 0.3" in output


def test_summarize_parameter_scan_rejects_mismatched_lengths():
    output = summarize_parameter_scan.invoke(
        {"parameter_values": [225.0, 250.0], "series": {"cls": [0.2]}}
    )
    assert output == "Error: cls has length 1, expected 2 to match parameter_values"


def test_summarize_parameter_scan_accepts_legacy_inputs_without_series():
    output = summarize_parameter_scan.invoke(
        {
            "parameter_values": [225.0, 250.0],
            "observed_significance": [2.1, 2.4],
            "expected_significance": [1.8, 1.9],
            "sort_by": "observed_significance",
            "descending": True,
            "top_n": 2,
        }
    )
    assert "Ranking (by observed_significance, descending):" in output
    assert "Best point: parameter = 250, observed_significance = 2.4, expected_significance = 1.9" in output


def test_required_signal_background_do_not_fallback_to_cwd(sample_context, monkeypatch):
    monkeypatch.chdir(sample_context["tests_dir"])
    output = compute_significance.invoke(
        {"signal_file": sample_context["signal"], "tree_name": sample_context["tree"], "cut": "1==1"}
    )
    assert output == "Error: no background file(s) provided."


def test_histogram_significance_and_cls_auto_detects_full_range(tmp_path):
    """center/window default to the full histogram range: [10,50] => center=30, window=20."""
    root_file = tmp_path / "auto_detect.root"
    f = ROOT.TFile.Open(str(root_file), "RECREATE")

    hbkg = ROOT.TH1F("bkg", "bkg", 20, 10.0, 50.0)
    hsig = ROOT.TH1F("sig", "sig", 20, 10.0, 50.0)
    hdata = ROOT.TH1F("data", "data", 20, 10.0, 50.0)
    for _ in range(20):
        hbkg.Fill(25.0)
    for _ in range(5):
        hsig.Fill(25.0)
    for _ in range(22):
        hdata.Fill(25.0)
    hbkg.Write()
    hsig.Write()
    hdata.Write()
    f.Close()

    output = histogram_significance_and_cls.invoke(
        {"file_path": str(root_file), "data_name": "data", "bkg_name": "bkg", "sig_name": "sig"}
    )
    assert "Center=30" in output
    assert "Window=[10" in output or "Window=[1" in output
    assert "Expected(S+B Asimov):" in output


def test_histogram_significance_and_cls_discovery_only(tmp_path):
    """compute_cls=False reports only discovery metrics (Z, p0), no CLs/CLs+b/CLb."""
    root_file = tmp_path / "disc_only.root"
    f = ROOT.TFile.Open(str(root_file), "RECREATE")
    hbkg = ROOT.TH1F("bkg", "bkg", 20, 0.0, 100.0)
    hsig = ROOT.TH1F("sig", "sig", 20, 0.0, 100.0)
    for _ in range(100):
        hbkg.Fill(50.0)
    for _ in range(20):
        hsig.Fill(50.0)
    hbkg.Write()
    hsig.Write()
    f.Close()

    result = histogram_significance_and_cls.invoke(
        {"file_path": str(root_file), "bkg_name": "bkg", "sig_name": "sig", "compute_cls": False}
    )
    assert "Z=" in result
    assert "p0=" in result
    assert "CLs=" not in result
    assert "CLs+b=" not in result
    assert "CLb=" not in result


def test_stat_summary_discovery_only():
    summary = _stat_summary(n_bkg=50.0, n_sig=10.0, compute_cls=False)
    assert "Z=" in summary
    assert "p0=" in summary
    assert "CLs=" not in summary
    assert "CLb=" not in summary

    summary_full = _stat_summary(n_bkg=50.0, n_sig=10.0, compute_cls=True)
    assert "CLs=" in summary_full
    assert "CLs+b=" in summary_full
    assert "CLb=" in summary_full

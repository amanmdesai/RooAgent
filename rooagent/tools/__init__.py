from .fit_tools import *
from .histogram_tools import *
from .stat_tools import *
from .data_format_tools import *
from .plot_tools import *
from .rdataframe_tools import *
from .tfile_tools import *
from .utils import *


# ================= ROOT BATCH MODE =================
ROOT.gROOT.SetBatch(True)

# ================= ROOT STYLE =================
ROOT.gStyle.SetOptStat(0)

ROOT.gStyle.SetTitleFont(42, "XYZ")
ROOT.gStyle.SetLabelFont(42, "XYZ")

ROOT.gStyle.SetTitleSize(0.05, "XYZ")
ROOT.gStyle.SetLabelSize(0.04, "XYZ")

ROOT.gStyle.SetPadLeftMargin(0.12)
ROOT.gStyle.SetPadBottomMargin(0.12)

ROOT.gStyle.SetLegendBorderSize(0)
ROOT.gStyle.SetLegendFillColor(0)

ROOT.gStyle.SetFrameLineWidth(2)
ROOT.gStyle.SetLineWidth(2)

ROOT.gStyle.SetPadGridX(False)
ROOT.gStyle.SetPadGridY(False)

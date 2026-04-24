from .fit_tools import *
from .histogram_tools import *
from .stat_tools import *
from .data_format_tools import *
from .plot_tools import *
from .rdataframe_tools import *
from .tfile_tools import *
from .utils import *


# ------------------
# ROOT configuration
# ------------------
import ROOT

# Run ROOT in batch mode by default for non-interactive environments
ROOT.gROOT.SetBatch(True)

# Suppress noisy ROOT, RooFit, and RooStats printouts globally.
try:
	ROOT.gErrorIgnoreLevel = ROOT.kError
except Exception:
	pass

try:
	ROOT.Math.MinimizerOptions.SetDefaultPrintLevel(-1)
except Exception:
	pass

try:
	ROOT.RooStats.AsymptoticCalculator.SetPrintLevel(0)
except Exception:
	pass

try:
	msg_service = ROOT.RooMsgService.instance()
	msg_service.setGlobalKillBelow(ROOT.RooFit.ERROR)
except Exception:
	pass

# Use a clean, publication-friendly default style. These settings are
# chosen to produce white canvases, clear fonts, tick marks on all sides,
# and a neutral color palette suitable for printed figures.
g = ROOT.gStyle
g.SetOptStat(0)       # no stat box by default
g.SetOptTitle(0)      # disable default histogram title

g.SetCanvasColor(ROOT.kWhite)
g.SetPadColor(ROOT.kWhite)
g.SetStatColor(ROOT.kWhite)
g.SetFrameFillColor(ROOT.kWhite)

g.SetTitleFont(42, "XYZ")
g.SetLabelFont(42, "XYZ")
g.SetTextFont(42)
g.SetTitleSize(0.05, "XYZ")
g.SetLabelSize(0.04, "XYZ")
g.SetTitleOffset(1.2, "Y")
g.SetTitleOffset(1.0, "X")
g.SetNdivisions(510, "X")

g.SetPadLeftMargin(0.12)
g.SetPadRightMargin(0.04)
g.SetPadTopMargin(0.06)
g.SetPadBottomMargin(0.12)

g.SetLegendBorderSize(0)
g.SetLegendFillColor(0)
g.SetLegendFont(42)
g.SetLegendTextSize(0.035)

g.SetFrameLineWidth(1)
g.SetLineWidth(2)
g.SetHistLineWidth(2)
g.SetEndErrorSize(6)
g.SetErrorX(0.5)

# Draw ticks on all sides of the frame
g.SetPadTickX(1)
g.SetPadTickY(1)

# Default color palette (numeric value works across ROOT versions)
try:
	g.SetPalette(55)
except Exception:
	pass

# By default, do not draw grid lines; callers may enable them when needed
g.SetPadGridX(False)
g.SetPadGridY(False)

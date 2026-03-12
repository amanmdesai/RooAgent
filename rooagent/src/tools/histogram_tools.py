from pydantic import BaseModel
import ROOT
from langchain_core.tools import tool

class HistogramStatsResponse(BaseModel):
    file_path: str
    hist_name: str
    mean: float
    rms: float
    entries: int

@tool
def get_histogram_stats(file_path: str, hist_name: str) -> str:
    """
    Retrieve statistical information about a histogram stored in a ROOT file.
    """
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return "Error: could not open file."
    h = f.Get(hist_name)
    if not h:
        f.Close()
        return "Histogram not found."
    mean = h.GetMean()
    rms = h.GetRMS()
    entries = int(h.GetEntries())
    f.Close()
    return f"{hist_name} -> Mean: {mean:.3f}, RMS: {rms:.3f}, Entries: {entries}"
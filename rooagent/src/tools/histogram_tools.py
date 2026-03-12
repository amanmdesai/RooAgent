from pydantic import BaseModel
import ROOT
from langchain_core.tools import tool


@tool
def get_histogram_stats(file_path: str, hist_name: str) -> str:
    """
    Retrieve basic statistical information (mean, RMS, entries) of a histogram
    stored in a ROOT file.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the histogram.
    hist_name : str
        Name of the histogram within the ROOT file.

    Returns
    -------
    str
        A formatted string reporting the histogram's mean, RMS, and total entries.
        Example:
            "myHist -> Mean: 0.123, RMS: 1.234, Entries: 1000"

    """
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return f"Error: could not open file {file_path}."
    h = f.Get(hist_name)
    if not h:
        f.Close()
        return f"Error: histogram '{hist_name}' not found in file {file_path}."
    mean = h.GetMean()
    rms = h.GetRMS()
    entries = int(h.GetEntries())
    f.Close()
    return f"{hist_name} -> Mean: {mean:.3f}, RMS: {rms:.3f}, Entries: {entries}"
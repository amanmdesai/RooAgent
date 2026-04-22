import ROOT
import numpy as np

# Settings
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
files = [
    (os.path.join(script_dir, "signal.root"), 10, 2.0, 1.0),
    (os.path.join(script_dir, "background.root"), 10, 0.0, 1.0),
    (os.path.join(script_dir, "background2.root"), 10, -2.0, 1.0),
    (os.path.join(script_dir, "data.root"), 10, 1.0, 1.5),
]

for fname, n, mean, sigma in files:
    f = ROOT.TFile(fname, "RECREATE")
    # Tree
    tree = ROOT.TTree("Events", "Events")
    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    tree.Branch("x", x, "x/F")
    tree.Branch("y", y, "y/F")
    for i in range(n):
        x[0] = np.random.normal(mean, sigma)
        y[0] = np.random.normal(mean, sigma)
        tree.Fill()
    tree.Write()
    # Histogram
    h = ROOT.TH1F("h1", "h1", 10, -5, 5)
    for i in range(n):
        h.Fill(np.random.normal(mean, sigma))
    h.Write()
    # 2D Histogram
    h2 = ROOT.TH2F("h2", "h2", 5, -5, 5, 5, -5, 5)
    for i in range(n):
        h2.Fill(np.random.normal(mean, sigma), np.random.normal(mean, sigma))
    h2.Write()
    f.Close()
print("Test ROOT files created.")

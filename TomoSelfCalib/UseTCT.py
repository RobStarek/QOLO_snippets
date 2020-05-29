# -*- coding: utf-8 -*-
"""
Example use of TomoCorrectionToolbox
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Custom helper modules from QOLO_snippets
import TomoCorrectionToolbox as tct


deg = np.pi/180.
StandardAngleTable = [
    (0        , 0),
    (45*deg   , 0),
    (22.5*deg , 0),
    (-22.5*deg, 0),
    (0*deg , 45*deg),
    (0*deg , -45*deg)
] #for tomographic projections

data = np.load("WPPrep5.npz")
PAT = data['Angles']*deg
tomograms = data['Data'][:,:,0]

R = tct.CalibrateTomography(tomograms, StandardAngleTable, PAT, ks.HI, ks.LO, 0)

#X = tct.EstProjRetErr(tomograms, StandardAngleTable, projection_state=ks.LO, mode=0)



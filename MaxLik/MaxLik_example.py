"""
Minimal example how to use MaxLikCore.
"""

import numpy as np
from MaxLikCore import MakeRPV, estim
#Definition of projection vector
LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
Plus = (LO+HI)*(2**-.5)
Minus = (LO-HI)*(2**-.5)
RPlu = (LO+1j*HI)*(2**-.5)
RMin = (LO-1j*HI)*(2**-.5)

#Definion of measurement order, matching data order
Order = [[LO,HI,Plus,Minus,RPlu,RMin]]
#Measured counts, correcponding to |0>+1j|1> state.
testdata = np.array([500,500,500,500,1000,1])
#Prepare (Rho)-Pi vect
RPV = MakeRPV(Order, False)
#Run reconstruction
E = estim(testdata, RPV, 1000, 1e-6)
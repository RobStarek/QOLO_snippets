"""
Minimal example how to use MaxLik.py.
"""

import numpy as np
from MaxLik import MakeRPV, Reconstruct
#Definition of projection vector (Pauli matrices eigenstate tomography)
LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
Plus = (LO+HI)*(2**-.5)
Minus = (LO-HI)*(2**-.5)
RPlu = (LO+1j*HI)*(2**-.5)
RMin = (LO-1j*HI)*(2**-.5)

Order = [[LO,HI,Plus,Minus,RPlu,RMin]] #Definion of measurement order, matching data order
testdata = np.array([500,500,500,500,1000,1]) #Measured counts, correcponding to |0>+1j|1> state.
R_state = np.array([[1],[1j]])
rho = R_state @ R_state.T.conjugate()
rho /= np.trace(rho)


RPV = MakeRPV(Order, False) #Prepare (Rho)-Pi vect
E = Reconstruct(testdata, RPV, 10000, 1e-12) #Run reconstruction
print(np.round(E,3))
print(np.round(rho,3))
print(f"Frob. distance from truth: {np.linalg.norm(rho - E)}")
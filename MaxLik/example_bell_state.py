"""
In this example we simulate tomogram of a Bell state and
reconstruct it.
"""

import numpy as np
from MaxLik import MakeRPV, Reconstruct
#Definition of projection vector
LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
Plus = (LO+HI)*(2**-.5)
Minus = (LO-HI)*(2**-.5)
RPlu = (LO+1j*HI)*(2**-.5)
RMin = (LO-1j*HI)*(2**-.5)

#Generate tomogram of Bell state |00> + |11>
PauliStates = [LO,HI,Plus,Minus,RPlu,RMin]
BellState = np.array([[1],[0],[0],[1]])*2**-.5
data = np.zeros((36,))
rate = 10000
for i, Q1 in enumerate(PauliStates):
    for j, Q2 in enumerate(PauliStates):
        projection = np.kron(Q1, Q2)
        amplitude = BellState.T.conjugate() @ projection
        data[6*i+j] = np.random.poisson(rate*np.abs(amplitude[0,0])**2)

#Reconstruct simulated tomogram
Order = [PauliStates,PauliStates]
RPV = MakeRPV(Order, False)
E = Reconstruct(data, RPV, 1000, 1e-6)
print(np.round(E,3))
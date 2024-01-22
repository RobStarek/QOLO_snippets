"""
In this example we simulate tomogram of a single-qubit quantum process and.
reconstruct it. The simulated process is Pauli SigmaX operator.
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

#Generate tomogram of a single-qubit process
PauliStates = [LO,HI,Plus,Minus,RPlu,RMin]
SigmaX = np.array([[0,1],[1,0]])
data = np.zeros((36,))
rate = 100_000
for i, Qin in enumerate(PauliStates):
    for j, Qproj in enumerate(PauliStates):        
        amplitude = Qproj.T.conjugate() @ SigmaX @ Qin
        data[6*i+j] = np.random.poisson(rate*np.abs(amplitude[0,0])**2)
process_X = np.zeros((4,4), dtype=complex)
process_X[1:3,1:3] = 0.5


#Reconstruct simulated process tomogram
Ord = [PauliStates]*2
RPV2 = MakeRPV(Ord, True)
E = Reconstruct(data, RPV2, 10000, 1e-12)
print(np.round(E,3))
print(process_X)

print(f"Frob. distance from truth: {np.linalg.norm(process_X - E)}")
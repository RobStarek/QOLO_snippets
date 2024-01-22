"""
In this example we test single-qubit ML reconstruction
on randomly generated pure states.
"""

import numpy as np
from MaxLik import MakeRPV, Reconstruct
import KetSugar as ks
import matplotlib.pyplot as plt

#Definition of projection vector (Pauli matrices eigenstate tomography)
LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
Plus = (LO+HI)*(2**-.5)
Minus = (LO-HI)*(2**-.5)
RPlu = (LO+1j*HI)*(2**-.5)
RMin = (LO-1j*HI)*(2**-.5)

#Define tomography order
PauliStates = [LO,HI,Plus,Minus,RPlu,RMin]
Order = [PauliStates]
RPV = MakeRPV(Order)

#Generate testing data and check reconstructions
N = 100 #number of state to be tested

rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)))
ParamsTheta = np.random.random(N)*np.pi
ParamsPhi = np.random.random(N)*2*np.pi
Rhos = []
Data = []
Fidelities = []
for theta, phi in zip(ParamsTheta, ParamsPhi):
    ket = ks.BlochKet(theta, phi)
    rho = ks.ketbra(ket, ket)
    Rhos.append(rho)
    tomogram = []
    for pi in PauliStates:
        amplitude = pi.T.conjugate() @ rho @ pi #Tr(Pi @ rho)
        tomogram.append(amplitude.real)
    tomogram = np.array(tomogram)        
    Data.append(tomogram)
    RhoML = Reconstruct(tomogram, RPV, 10000, 1e-12)
    Fid = ks.Overlap(RhoML, rho).real
    Fidelities.append(Fid)

plt.plot(ParamsTheta, ".")
plt.plot(ParamsPhi, ".")
plt.plot(Fidelities, "o")
plt.show()

print(np.mean(Fidelities), np.std(Fidelities))






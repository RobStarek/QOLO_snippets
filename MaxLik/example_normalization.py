"""
Test reconstruction of with measurement operators that do not add up to 
identity operator.
Without renormalization, the reconstruction purity is systematically lower.
"""

import numpy as np
import KetSugar as ks
import MaxLik as ml

state = ks.BlochKet(1*np.pi/4, -np.pi/2)
rho = ks.ketbra(state, state)

ps = [
    ks.BlochKet(0.1, 0),
    ks.BlochKet(np.pi-0.1, 0),
    ks.BlochKet(np.pi/2, 0+0.1),
    ks.BlochKet(np.pi/2, np.pi-0.1),
    ks.BlochKet(np.pi/2, np.pi/2),
    ks.BlochKet(np.pi/2+0.05, -np.pi/2)
]
ops = np.array([
    ks.ketbra(ket, ket) for ket in ps
])

data = np.array([ks.ExpectationValue(ket, rho).real for ket in ps])
rhorec = ml.Reconstruct(data, ops, 10000, 1e-9, RhoPiVect=True, Renorm=False)
F = ks.Fidelity(rhorec, rho).real
P = ks.Purity(rhorec)
print("Without normalization")
print(P)
print(F)

data = np.array([ks.ExpectationValue(ket, rho).real for ket in ps])
rhorec = ml.Reconstruct(data, ops, 10000, 1e-9, RhoPiVect=True, Renorm=True)
F = ks.Fidelity(rhorec, rho).real
P = ks.Purity(rhorec)
print("With normalization")
print(P)
print(F)
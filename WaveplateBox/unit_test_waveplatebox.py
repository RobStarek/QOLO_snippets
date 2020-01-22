# -*- coding: utf-8 -*-
import numpy as np
import WaveplateBox as wpb
import matplotlib.pyplot as plt
"""
Unit test of ket preparation waveplate angle search.
"""

#Ket constructors
def BlochKet(theta, phi):
    """
    Return a qubit column-vector with Bloch sphere coordinates theta, phi.
    theta - lattitude measured from |0> on Bloch sphere
    phi - longitude measured from |+> on Bloch sphere (phase)
    """
    return np.array([[np.cos(theta/2)], [np.sin(theta/2)*np.exp(1j*phi)]])    

nx = 91
ny = 91
Phis = np.linspace(-np.pi,np.pi,nx)
Thetas = np.linspace(0,np.pi,ny)

Results = np.zeros((nx,ny))
for i, theta in enumerate(Thetas):
    print(i, theta)
    for j, phi in enumerate(Phis):
        ket = BlochKet(theta, phi)
        R = wpb.SearchForKet(ket)
        Results[i,j] = (-R['fun'])
    
print(np.mean(Results))
print(np.std(Results))

plt.matshow(Results)
plt.colorbar()
plt.show()
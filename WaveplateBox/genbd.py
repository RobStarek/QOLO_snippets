"""
Matrix representation of a polarizing beam displacer, suplement to WaveplateBox.
The beam displacer shifts horizontally polarized components laterally and keeps vertically polarized components intact.
In our description, the qudit-qubit space is a Kronecker product of spatial part with n paths and polarization path (2 modes).
In our experiment we use this beam displacer to a) add paths and b) merge paths.
a) Vacuum state is assumed in the unused part.
b) First and last modes are discarded.
"""

import numpy as np

def BD(n,plus=True):
    """
    Matrix of a polarizing beam displacer (BD).
    Args:
      n ... number of input optical paths
      plus (default True) ... whether we use the BD to add new optical paths (True) or to merge them (False)
    Returns:
      M ... (2n x 2(n+1)) or (2n x 2(n-1)) complex ndarray with the matrix of BD.
    """
    
    #BD just organizes and swaps optical modes. The matrix represents which input are transfered to which output modes.
    if plus:
        M = np.zeros((2*n + 2, 2*n), dtype=complex)
        for k in range(0, 2*n):
            factor = 2*(k % 2)
            M[k+factor, k] = 1
    else:
        M = np.zeros((2*n - 2, 2*n), dtype=complex)
        for k in range(1, 2*n-1):
            factor = -2*(1 - k % 2)
            M[k+factor, k] = 1
    return M
  
# Example
# SX = np.array([[0,1],[1,0]])
# import matplotlib.pyplot as plt
# imodes = ['H', 'V']        
# for n in [1,2,3,4]:    
#     Mp = BD(n).real
#     Mm = BD(n+1, False).real
#     labPX = [f'{i//2}{imodes[i % 2]}' for i in range(2*n)]
#     labPY = [f'{i//2}{imodes[i % 2]}' for i in range(2*n)]              
#     Corr1 = np.kron(np.eye(n+1), SX)
#     Corr2 = np.kron(np.eye(n), SX)
#     Test = Corr2 @ (Mm @ (Corr1 @ Mp))
#     plt.matshow(Test.real)
#     plt.show()

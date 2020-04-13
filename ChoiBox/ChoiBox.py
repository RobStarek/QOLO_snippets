# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
import KetSugar as ks

"""
Little toolbox for manipulating unitary matrices and Choi matrices.
Dependencies: numpy, scipy, KetSugar

References:
M.-D. Choi, Completely Positive Linear Maps on Complex Matrices, Linear Algebra and its Applications, 10, 285–290 (1975).
https://en.wikipedia.org/wiki/Channel-state_duality
"""

def GuessUfromChoi(Chi):
    """
    Guess unitary parameters from Choi matrix Chi. 
    Chi matrix has to origin in unitary operation for this to work well.
    Returns:
        theta, phi1, phi2 - real-valued unitary parameters
    """
    if Chi[0,0] == 0:
        theta = np.pi/2
        phi1 = 0
        phi2 = 0
        return theta, phi1, phi2
    elif Chi[1,1] ==0:
        theta = 0
        phi1 = 0
        phi2 = 0
        return theta, phi1, phi2
    theta = np.arctan((Chi[1,1]/Chi[0,0])**.5)
    phi1 = np.angle(Chi[0,3]/Chi[0,0])*0.5
    phi2 = np.angle(Chi[1,2]/Chi[1,1])*0.5
    return theta, phi1, phi2

def UtoChi(U, ket=False):
    """
    Transform single-qubit unitary matrix U to Choi matrix Chi 
    or Choi ket (when ket = True).
    """
    Bell = np.array([[1],[0],[0],[1]])/2**.5
    ChiKet = ks.kron(np.eye(2),U) @ Bell
    if ket:
        return ChiKet
    Chi = ks.ketbra(ChiKet, ChiKet)
    return Chi


def UtoChiD(U, ket=False):
    """
    Transform multi-qubit channel matrix into Choi ket/matrix.
    """
    n = U.shape[0]    
    Base = [ks.BinKet(i,n-1) for i in range(n)]
    Bell = sum([np.kron(ket, ket) for ket in Base])    
    ChiKet = np.kron(np.eye(n), U) @ Bell
    if ket:
        return ChiKet
    Chi = ks.ketbra(ChiKet, ChiKet)
    return Chi

def UfromParam(theta, phi1, phi2):
    """
    Construct single-qubit unitary matrix from given unitary parameters
    theta, phi1, phi2.    
    """
    alpha = np.cos(theta)*np.exp(1j*phi1)
    beta = np.sin(theta)*np.exp(1j*phi2)
    U = np.array([
        [alpha, -beta.conjugate()],
        [beta, alpha.conjugate()]
        ])
    return U

def ChiKetFromParam(theta, phi1, phi2):
    """
    Construct Choi ket from given unitary parameters.
    """
    U = UfromParam(theta, phi1, phi2)
    ChiKet = UtoChi(U, ket = True)
    return ChiKet

def FitChiU(Chi):
    """
    Find closest unitary to given Choi matrix.
    Returns:
        Result dictionary R, see scipy.optimize.minimize for more details.
    """
    def minim(x):
        ket = ChiKetFromParam(x[0], x[1], x[2])
        return -ks.ExpectationValue(ket, Chi).real
    
    guess = GuessUfromChoi(Chi)
    R = minimize(minim, guess)
    return R    

def MapTransform(rho, chi, renorm = True):
    """
    Transform input density matrix with Choi matrix.
    Tr_i[(1 \otimes RhoIn.T) chi (...)^\dagger]
    Args:
        rho - density matrix to be transformed
        chi - quantum process matrix
        renorm - (default true), trace-normalize output
    Returns:
        transformed density matrix
    """
    d = rho.shape[0] #number of rows
    n = int(np.log2(d)) #number of qubits
    trace_list = [1]*n + [0]*n

    POVM = np.kron(rho.T, np.eye(d, dtype=complex))
    ChiTrans = POVM @ chi @ ks.dagger(POVM)
    RhoOut = ks.TraceOverQubits(ChiTrans, trace_list)
    if renorm:
        return RhoOut/np.trace(RhoOut)
    else:
        return RhoOut

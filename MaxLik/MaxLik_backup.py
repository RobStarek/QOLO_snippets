# -*- coding: utf-8 -*-
import numpy as np
from functools import reduce
import itertools
"""MaxLik.py
Discrete-variable quantum maximum-likelihood reconstruction.

This module provides a simple numpy-implementation of Maximum likelihood reconstruction
method [1,2] for reconstructing low-dimensional quantum states and processes (<=6 qubits in total).

This package is limited to projection and preparation of pure states.

An orderd list of prepared/projected list is provided to MakeRPV() function to
generate auxiliary array of projection-preparation matrices (Rho-Pi vector). 

The Rho-Pi vector is inserted as an argument together with data to Reconstruct() function.
The Reconstruct() function returns reconstructed density matrix.

Example:
    Minimal single-qubit reconstruction
        import numpy as np
        from MaxLikCore import MakeRPV, Reconstruct
        #Definition of projection vector
        LO = np.array([[1],[0]])
        HI = np.array([[0],[1]])
        Plus = (LO+HI)*(2**-.5)
        Minus = (LO-HI)*(2**-.5)
        RPlu = (LO+1j*HI)*(2**-.5)
        RMin = (LO-1j*HI)*(2**-.5)
        #Definion of measurement order, matching data order
        Order = [[LO,HI,Plus,Minus,RPlu,RMin]]
        #Measured counts
        testdata = np.array([500,500,500,500,1000,1])
        #Prepare (Rho)-Pi vect
        RPV = MakeRPV(Order, False)
        #Run reconstruction
        E = Reconstruct(testdata, RPV, 1000, 1e-6)


References:
    1. Fiurasek, Hradil, Maximum-likelihood estimation of quantum processes, Phys. Rev. A 63, 020101(R) (2001) https://journals.aps.org/pra/abstract/10.1103/PhysRevA.63.020101
    2. Paris (ed.), Rehacek, Quantum State Estimation - 2004, Lecture Notes in Physics, ISBN: 978-3-540-44481-7, https://doi.org/10.1007/b98673

Todo:
    * ?

"""


def RPVketToRho(RPVKet):
    """
    Convert ndarray of sorted preparation-detection kets into ndarray of 
    density matrices.

    Args:
        RPVket: n x d ndarray (line-vectors) containing measured projections or preparation-projection vectors.
        or n x 1x d nd array (column vectors)

    Returns:
        RhoPiVect: n x d x d ndarray of density matrices made up from RPV kets.
    """
    shape = RPVKet.shape
    column = True
    if len(shape) == 2:
        column = False  # kets are stored as line-vectors
    elif len(shape) == 3:
        column = True  # kets are stored as columns-vectors
    else:
        raise Exception("MaxLik: Unexpected shape of RPVket.")

    n = shape[0]
    dim = shape[1]
    RhoPiVect = np.zeros((n, dim, dim), dtype=complex)
    for i, ket in enumerate(RPVKet):
        if not(column):
            ketx = ket.reshape((dim, 1))
            RhoPiVect[i] = np.dot(ketx, ketx.T.conjugate())
        else:
            RhoPiVect[i] = np.dot(ket, ket.T.conjugate())
    return RhoPiVect


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    
    Source: https://stackoverflow.com/a/16873755
    """
    h = arr.shape[0]
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def MakeRPV(Order, Proc=False):
    """
    Create list preparation-projection kets.
    Kets are stored as line-vectors or column-vectors in n x d ndarray.
    This function is here to avoid writing explicit nested loops for all combinations of measured projection.

    Args:
        Order: list of measured/prepared states on each qubit, first axis denotes qubit, second measured states, elements are kets stored as line-vectors or column-vectors.
        Proc: if True, first half of Order list is regarded as input states and therefore conjugated prior building RPV ket.

    Returns:
        RPVvectors: complex ndarray of preparation-projection kets, order should match the measurement
    """
    N = len(Order)
    if Proc:
        # When reconstructing a process, conjugate the preparation qubits
        # just conjugate them, do not perform Hermitean conjugation

        #To avoid modification of original qubit references,
        #which may result in some errors, use copy a of an Order list instead
        OrderC = [[np.copy(qubit) for qubit in qubits] for qubits in Order]
        for i in range(N//2):
            for j in range(len(OrderC[i])):
                OrderC[i][j] = np.conjugate(OrderC[i][j])
        #To do: construct OrderC directly, instead copying it and 
        #modifying it later
        RPVindex = itertools.product(*OrderC, repeat=1)
    else:
        RPVindex = itertools.product(*Order, repeat=1)
        
    RPVvectors = []
    for projection in RPVindex:
        RPVvectors.append(reduce(np.kron, projection))
    return np.array(RPVvectors)

def Reconstruct(data, RPVket, max_iters=100, tres=1e-6):
    """
    Maximum likelihood reconstruction of quantum state/process.

    Args:
        data: ndarray of n real numbers, measured in n projections of state.
        RPVket: Complex n x d (line vectors) or n x d x 1 (column vectors) ndarray of kets describing the states
        that the measured state is projected onto.
        max_iters: integer number of maximal iterations
        tres: when the change of estimated matrix measured by Frobenius norm is less than this, iteration is stopped

    Returns:
        E: estimated density matrix, d x d complex ndarray.
    """
    RhoPiVect = RPVketToRho(RPVket)
    RhoPiVectAux = np.hstack(RhoPiVect)
    # prepare data-rho-pi product
    dim = RhoPiVect.shape[1]
    Xaux = data[:, np.newaxis, np.newaxis]
    OutPiRho = Xaux*RhoPiVect

    count = 0
    E = np.identity(dim, dtype=complex)
    E = E*1.0/dim

    meas = 10*tres  # initial from. meas - far above threshold

    # iterate until you reach threshold in frob. meas. of diff or
    # exceed maximally allowed number of steps
    while count < max_iters and meas > tres:
        Ep = E  # Original matrix for comparison
        # K-Matrix prepare using numpy tricks
        Etile = np.tile(E.T, RhoPiVectAux.shape[1]//E.shape[1])
        Denom = Etile*RhoPiVectAux
        Denom = blockshaped(Denom, dim, dim)
        Denom = np.sum(Denom, (1, 2))
        # cast-shape to 1296x16x16 dimension
        Denom = Denom[:, np.newaxis, np.newaxis]
        K = np.sum(OutPiRho/Denom, axis=0)  # do the sum
        # apply K operator from left and right
        E = np.dot(K, E)
        E = np.dot(E, K)
        E = E/(E.trace())  # norm the matrix
        meas = abs(np.linalg.norm((Ep-E)))  # threshold check
        count += 1  # counter increment
    return E

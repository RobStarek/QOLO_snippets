# -*- coding: utf-8 -*-
"""
A small collection of functions for 
decomposition of operators into generalized
Pauli matrices and their corresponding
eigenprojectors.
"""

from functools import reduce
from itertools import product
import numpy as np
from scipy.optimize import minimize
import KetSugar as ks

def gen_base10_to_base_m(m):
    """
    Get a function that maps base10 integer to 
    list of base m representation (most significant first)
    """
    def _f(i, digits):
        powers = (m**np.arange(digits))[::-1]
        iact = i
        indices = []
        for j in range(digits):
            idx = iact // powers[j]
            indices.append(idx)
            iact -= (indices[-1]*powers[j])
        return indices
    return _f

def gen_base_m_to_base10(m):
    """
    Get a function that maps list of base m digits (most significant first)
    to base10 integer.
    """    
    def _f(args, n):
        powers = (m**np.arange(n))[::-1]
        return np.array(args) @ powers
    return _f

#auxiliary conversion functions
base10_to_base4 = gen_base10_to_base_m(4)
base10_to_base6 = gen_base10_to_base_m(6)
base6_to_base10 = gen_base_m_to_base10(6)

def spawn_generalized_pauli_operators(*basis_mats):
    """
    Generate list of generalized Pauli operators
    constructed from given basis vectors.
    Args:
        *basis_mats ... n 2x2 array of basis column vectors
    Returns:
        4**n operators
        indexing:        
        j=0 identity
        j=1 sigma z
        j=2 sigma x
        j=3 sigma y
        I = sum_j i_j*4**(n-j-1)
    """
    isqrt = 2**(-.5)
    pauli_ops_base = []
    for basis in basis_mats:
        low = basis[:,0].reshape((2,1))
        high = basis[:,1].reshape((2,1))
        kz0 = low
        kz1 = high
        kx0 = (low+high)*isqrt
        kx1 = (low-high)*isqrt
        ky0 = (low+1j*high)*isqrt
        ky1 = (low-1j*high)*isqrt
        pauli_ops_base.append([np.eye(2)] + [ks.ketbra(ket1, ket1) - ks.ketbra(ket2, ket2) for ket1, ket2 in [(kz0, kz1), (kx0, kx1), (ky0, ky1)]])
    
    n = len(basis_mats)
    gammas = []
    for i in range(4**n):
        indices = base10_to_base4(i, n)
        operators = [pauli_ops_base[j][idx] for j, idx in enumerate(indices)]                    
        gamma = reduce(np.kron, operators)
        gammas.append(gamma)
    return gammas

def spawn_generalized_pauli_projectors(*basis_mats):
    """
    Generate list of projectors on
    eigenstates of generalized Pauli operators which are
    constructed from given basis vectors.
    Args:
        *basis_mats ... n 2x2 array of basis column vectors
    Returns:
        6**n operators
        indexing:        
        j=0 |z+>
        j=1 |z->
        j=2 |x+>
        j=3 |x->
        j=4 |y+>
        j=6 |y->
        I = sum_j i_j*(6**(n-j-1))
    """
    isqrt = 2**(-.5)
    pauli_proj_base = []
    for basis in basis_mats:
        low = basis[:,0].reshape((2,1))
        high = basis[:,1].reshape((2,1))
        kz0 = low
        kz1 = high
        kx0 = (low+high)*isqrt
        kx1 = (low-high)*isqrt
        ky0 = (low+1j*high)*isqrt
        ky1 = (low-1j*high)*isqrt
        pauli_proj_base.append([kz0, kz1, kx0, kx1, ky0, ky1])
    n = len(basis_mats)
    pis = []
    for i in range(6**n):
        indices = base10_to_base6(i, n)
        operators = [pauli_proj_base[j][idx] for j, idx in enumerate(indices)]
        gamma = reduce(np.kron, operators)
        pis.append(ks.ketbra(gamma, gamma))
    return pis
    
def decompose_operator_into_gen_pauli(mat, *basis_mats):
    """
    Decompose operator M into a weighted sum of generalized Pauli operators,
    basis is specified by basis_mat, see spawn_generalized_pauli_operators().
    mat = sum_i (w_i G_i)
    Args:
        mat ... matrix to be decomposed
        basis_mats ... horizontally stacked column basis vectors for given qubit
    Returns:
        list of tuples (w_i, G_i) where w_i is the coefficient and G_i is
        genaralized Pauli operator in given basis.
    """
    paulis = spawn_generalized_pauli_operators(*basis_mats)     
    return [((op.T.ravel() @ mat.ravel()), op) for op in paulis] #= tr (M op)

#Eigenvalues and indices of eigenkets of Pauli matrices
SIGMA_TABLE = [
    [np.array((1, 1)),  np.array((0,1))], #identity
    [np.array((1, -1)), np.array((0,1))], #sigma z
    [np.array((1, -1)), np.array((2,3))], #sigma x
    [np.array((1, -1)), np.array((4,5))], #sigma y
]
def select_proj_index_from_op(indices):
    """
    Take base4 indices of generalized Pauli operator and
    turn them into coefficients and base10 indices
    of the corresponding eigenstate projectors.
    """
    n = len(indices)
    factors = [SIGMA_TABLE[i][0] for i in indices]
    meas_indices = [SIGMA_TABLE[i][1] for i in indices]
    multipliers = [np.prod(args) for args in product(*factors)]
    indices = [base6_to_base10(index, n) for index in product(*meas_indices)]
    return multipliers, indices

def decompose_operator_into_projectors(mat, *basis_mats):
    """
    Decompose operator mat into a weighted sum of Pauli eigenstate 
    projectors, so it holds
    mat = sum_j |eig(j)><eig(j)|w_j.
    It first decomposes mat into generalized Pauli operators
    and then it maps them to projectors.
    Args:
        mat ... matrix to be decomposed
        basis_mats ... horizontally stacked column basis vectors for given qubit
    Returns:
        proj_weights ... list of weights w
    """
    n = len(basis_mats)
    weights = decompose_operator_into_gen_pauli(mat,*basis_mats)
    proj_weights = np.zeros((6**n,), dtype=complex)
    norm = (1/2)**n
    for i, (w, _) in enumerate(weights):
        multi_index = base10_to_base4(i, n)
        multipliers, indices = select_proj_index_from_op(multi_index)
        for j, wj in zip(indices, multipliers):
            proj_weights[j] = proj_weights[j] + wj*w*norm
    return proj_weights

def compose_operator_from_proj_weights(weights, ops):
    """
    Calculate sum_j w_j op_j, where w_j are elements of weights and
    op_j are elements of operator list ops.    
    """
    n = int(ops[0].shape[0])
    M = np.zeros((n,n), dtype=complex)
    for i, w in enumerate(weights):
        M = M + w*ops[i]
    return M

def measure_effort_c(projector_coefs, thr=1e-3):
    """
    Quantify how many projections (up to multiplicative factor) are needed for the decomposed measurements.
    """
    return np.sum((np.abs(projector_coefs) > thr)*np.abs(projector_coefs))

def measure_effort_d(projector_coefs, thr=1e-3):
    """
    How many projections are needed for the decomposed measurements.    
    """    
    return np.sum((np.abs(projector_coefs) > thr))

def generate_basis_mat(theta, phi):
    """
    Generate basis vectors from the Bloch coordinates.
    """
    ket1 = ks.BlochKet(theta, phi)
    ket2 = ks.BlochKet(np.pi-theta, phi+np.pi)
    return np.hstack([ket1, ket2])

def optimized_proj_decomposition(mat, thr=1e-3):
    """
    Decompose operator mat into sum projectors
    so that the number of measurements with |coefficient| smaller
    than thr is minimized. Minimization is done by
    varying the measurement basis for each qubit.
    Args:
        mat ... operator to be decomposed
        thr ... threshold on coefficient size
    Returns:
        coefs, basis ... coefficients and measurement basis
    """
    n = int(np.log2(mat.shape[0])) #number of systems
    def minim(args):
        basis = [generate_basis_mat(args[i*2], args[i*2+1]) for i in range(n)]
        coefs = decompose_operator_into_projectors(mat, *basis)
        return measure_effort_c(coefs, thr)
    guess = guess_optimal_basis(mat)    
    R = minimize(minim, guess)
    basis = [generate_basis_mat(R['x'][i*2], R['x'][i*2+1]) for i in range(n)]
    coefs = decompose_operator_into_projectors(mat, *basis)        
    return coefs, basis

def guess_optimal_basis(mat):
    """
    Guess optimal basis for decomposition of operator mat
    into a minimal number of local measurements.
    The guess is based on eigenkets of mat's subsystems.
    """
    n = int(np.log2(mat.shape[0])) #number of systems    
    params = []
    for i in range(n):
        trace = [1]*n
        trace[i] = 0
        op = ks.TraceOverQubits(mat, trace)
        eigval, eigket = np.linalg.eigh(op)
        theta = 2*np.arccos(np.abs(eigket[0,0]))
        phi = np.angle(eigket[1,0]) - np.angle(eigket[0,0])
        params.extend([theta, phi])
    return np.array(params)

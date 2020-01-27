# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
"""
Toolbox of commonly used function for waveplate-polarization manipulation.
Jones formalism and Bloch-sphere representation is used.
"""

def ROT(x):
    """
    Matrix of 2D coordinate rotation of angle x.    
    """
    cx = np.cos(x)
    sx = np.sin(x)
    return np.array([[cx, -sx],[sx, cx]])

def HWP(x):
    """
    Rotated half-wave plate in Jones formalism.
    """
    M = np.array([[1,0],[0,-1]])
    R = ROT(x)
    return R @ M @ R.T

def QWP(x):
    """
    Rotated quarter-wave plate in Jones formalism.
    """    
    M = np.array([[1,0],[0,-1j]])
    R = ROT(x)
    return R @ M @ R.T

def WPPrepare(x,y, input_state = np.array([[1],[0]])):
    """
    Jones vector (ket) of polarization prepared from input_state through a 
    pair of wave plates.
    input_state-->QWP-->HWP-->output
    Args:
        x - rotation of half-wave plate
        y - rotation of quarter-wave plate
        input_state - Jones (ket) vector of input state
    Returns:
        Jones vector of the prepared state
    """
    return HWP(x) @ QWP(y) @ input_state

def QHQBlock(x,y,z):
    """
    Unitary of QWP-HWP-QWP block.
    Args:
        x - rotation of quarter-wave plate
        y - rotation of half-wave plate
        z - rotation of quarter-wave plate
    Returns:
        unitary matrix of net effect
    """    
    return QWP(x) @ HWP(y) @ QWP(z)


def ProcessSimilarity(A,B):
    """
    Measure similarity of two trace-preserving single-qubit operations.
    Choi-Jamiolkovski isomorphism (channel-state duality)
    is used to represent unitary operations as vectors.

    Args:
        A, B - 2x2 ndarrays of unitary operation
    Returns:
        Process fidelity - 0 to 1, with 0 for orthogonal processes, 1 for identical processes.
    """
    Bell = np.array([[1],[0],[0],[1]])*2**-.5
    BellA = np.kron(np.eye(2), A) @ Bell
    BellB = np.kron(np.eye(2), B) @ Bell
    return np.abs(BellA.T.conjugate() @ BellB)[0,0]**2

"""Auxilliary QWP-HWP-QWP and HWP-QWP Search-grids for minimization
This is required for SearchForU and SearchForKet functions"""
deg = np.pi/180
grid_qwp = np.array([0.1, 89.1, -89.9])*deg
grid_hwp = np.array([0.1, 44.9, -45.1, 89.1, -89.1])*deg
search_grid_qhq = []
search_grid_qh = []
wp_bounds = (-185*deg, 185*deg)
for alpha in grid_qwp:
    for beta in grid_hwp:
        search_grid_qh.append([beta, alpha])
        for gamma in grid_qwp:
            search_grid_qhq.append([alpha, beta, gamma])            

def SearchForU(U, tol=1e-6):
    """
    Search for angles x,y,z which implement desired unitary U with
    three wave-plates:
    QWP(X)-HWP(y)-QWP(z)->    
    Args:
        U - desired unitary single-qubit operation
        tol - minimizer tolerance, see scipy.optimize.minimize docs.
    Returns:
        R - dictionary with minimization details. 
        R['x'] contains desired angles, R['fun'] is measure of quality (should be -1)
        See scipy.optimize.minimize docs for more details.
    """
    #Construct function to be minimized
    def minim(x):
        Ux = QHQBlock(x[0],x[1],x[2])        
        return -ProcessSimilarity(Ux, U)
    Rs = [minimize(minim, g, bounds=[wp_bounds]*3, tol=tol) for g in search_grid_qhq]
    Fs = np.array([R['fun'] for R in Rs])
    idx = np.argmin(Fs)    
    return Rs[idx]

def SearchForKet(ket, input_state=np.array([[1],[0]]), tol=1e-6):
    """
    Search for wave plates angles x,y which prepare desired ket state from input state
    with setup:
    input_state->QWP(y)-HWP(x)->
    Args:
        ket - desired Jones vector to be prepared
        input_state - input Jones vector
        tol - minimizer tolerance, see scipy.optimize.minimize docs.
    Returns:
        R - dictionary with minimization details. 
        R['x'] contains desired angles, R['fun'] is measure of quality (should be -1)
        See scipy.optimize.minimize docs for more details.
    """    
    #Construct function to be minimized
    def minim(x):
        ketX = WPPrepare(x[0],x[1], input_state)        
        return -np.abs(ketX.T.conjugate() @ ket)[0,0]**2
    #Start minimization from multiple initial guessses to avoid sub-optimal local extremes.
    Rs = [minimize(minim, g, bounds=[wp_bounds]*2, tol=tol) for g in search_grid_qh]
    #Pick global extereme.
    Fs = np.array([R['fun'] for R in Rs])
    idx = np.argmin(Fs)
    return Rs[idx]

def SearchForRho(rho, input_state=np.array([[1],[0]]), tol=1e-6):
    """
    Search for wave plates angles x,y which prepare desired pure density matrix from input state
    with setup:
    input_state->QWP(y)-HWP(x)->
    Args:
        ket - desired Jones vector to be prepared
        input_state - input Jones vector
        tol - minimizer tolerance, see scipy.optimize.minimize docs.
    Returns:
        R - dictionary with minimization details. 
        R['x'] contains desired angles, R['fun'] is measure of quality (should be -1)
        See scipy.optimize.minimize docs for more details.
    """    
    #Construct function to be minimized
    def minim(x):
        ketX = WPPrepare(x[0],x[1], input_state)        
        return -np.abs(ketX.T.conjugate() @ rho @ ketX)[0,0]
    #Start minimization from multiple initial guessses to avoid sub-optimal local extremes.
    Rs = [minimize(minim, g, bounds=[wp_bounds]*2, tol=tol) for g in search_grid_qh]
    #Pick global extereme.
    Fs = np.array([R['fun'] for R in Rs])
    idx = np.argmin(Fs)
    return Rs[idx]

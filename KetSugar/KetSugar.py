# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg
"""
Shorthand notation, basic constants and frequently used function
for some basic low-dimensional DV quantum mechanics.
"""

#Constants
LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
HLO = (LO+HI)*(2**-.5)
HHI = (LO-HI)*(2**-.5)
CLO = (LO+1j*HI)*(2**-.5)
CHI = (LO-1j*HI)*(2**-.5)

#Short-hand notation
def dagger(x : np.ndarray):
    """
    Hermite conjugation of x.
    """
    return x.T.conjugate()

def braket(x : np.ndarray, y : np.ndarray):
    """
    Inner product of two ket-vectors -> C-number
    """
    return np.dot(x.T.conjugate(), y)[0,0]

def ketbra(x : np.ndarray, y : np.ndarray):
    """
    Outer product of two ket-vectors -> C-matrix
    """
    return np.dot(x, y.T.conjugate())

def kron(*arrays):
    """
    Multiple Kronecker (tensor) product.
    Multiplication is performed from left.    
    """
    E = np.eye(1, dtype=complex)
    for M in arrays:
        E = np.kron(E,M)
    return E

def BinKet(i=0,imx=1):
    """
    Computational base states i in imx+1-dimensional vectors.
    """
    ket = np.zeros((imx+1,1), dtype=complex)
    ket[i] = 1
    return ket

#Ket constructors
def BlochKet(theta, phi):
    """
    Return a qubit column-vector with Bloch sphere coordinates theta, phi.
    theta - lattitude measured from |0> on Bloch sphere
    phi - longitude measured from |+> on Bloch sphere (phase)
    """
    return np.array([[np.cos(theta/2)], [np.sin(theta/2)*np.exp(1j*phi)]])

#Routinely used simple functions
def Overlap(MA, MB):
    """
    Normalized overlap of two density matrices MA, MB.
    When at least one of the matrices is pure, it is equivalent to fidelity.
    """
    return np.trace(MA @ MB)/(np.trace(MA)*np.trace(MB))

def Purity(M):
    """
    Purity of the density matrix M.
    For n qubits, minimum is (2^n).
    """
    norm = np.trace(M)
    return np.trace(M @ M)/(norm**2)

def ApplyOp(Rho,M):
    """
    Calculate M.Rho.dagger(M).
    """
    return M @ Rho @ M.T.conjugate()

def ExpectationValue(Ket, M):
    """
    Expectation value <bra|M|ket>.
    """
    return (Ket.T.conjugate() @ M @ Ket)[0,0]

def GrammSchmidt(X, row_vecs=False, norm=True):
    """
    Vectorized Gramm-Schmidt orthogonalization.
    Creates an orthonormal system of vectors spanning the same vector space
    which is spanned by vectors in matrix X.

    Args:
        X: matrix of vectors
        row_vecs: are vectors store as line vectors? (if not, then use column vectors)
        norm: normalize vector to unit size
    Returns:
        Y: matrix of orthogonalized vectors
    Source: https://gist.github.com/iizukak/1287876#gistcomment-1348649
    """
    if not row_vecs:
        X = X.T
    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag(
            (X[i, :].dot(Y.T)/np.linalg.norm(Y, axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y, axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def sqrtm(M):
    """
    Matrix square root of density matrix M.
    Calculation is based on eigen-decomposition.    
    """
    Di, Rot = np.linalg.eig(M)
    rank = np.sum((np.abs(Di) > 2*np.finfo(float).eps))
    Di = np.sqrt(Di)
    Di[np.isnan(Di)] = 0
    Di = np.diag(Di)
    if (rank == M.shape[0]):
        #Full rank => Hermitean transposition is actually inversion
        RotInv = Rot.T.conjugate()
    elif rank == 1:
        #Rank 1 => The state is pure and the matrix is it's own square-root.
        return M
    else:
        #(1 < Rank < dimension) => at least one eigenvalue is zero, orthogonalize found unitary
        #in order to perform Hermitean inversion of the rotation matrix
        #If this was not the case, zero eigenvalue can correspond to 
        #arbitrary vector which would destroy unitarity of Rot matrix.
        RotGs = GrammSchmidt(Rot, False, True)
        RotInv = RotGs.T.conjugate()
    N = np.dot(np.dot(Rot, Di), RotInv)
    return N

def Fidelity(A, B):
    """
    Fidelity between density matrices A, B.
    Accepts both mixed. If A or B is pure, consider using Overlap().
    """
    #A0 = sqrtm(A)
    Ax = A/np.trace(A)
    Bx = B/np.trace(B)
    A0 = scipy.linalg.sqrtm(Ax)
    A1 = (np.dot(np.dot(A0, Bx), A0))
    A2 = scipy.linalg.sqrtm(A1)
    return np.abs(A2.trace())**2

def TraceLeft(M):
    """
    Partial trace over left-most qubit.
    """
    n = M.shape[0]
    return M[0:n//2, 0:n//2] + M[n//2:n, n//2:n]

def TraceRight(M):
    """
    Partial trace over right-most qubit.
    """
    n = M.shape[0]
    blocks = n//2
    TrM = np.zeros((blocks, blocks), dtype=complex)
    for i in range(blocks):
        for j in range(blocks):
            TrM[i,j] = np.trace(M[i*2:(1+i)*2, j*2:(1+j)*2])
    return TrM

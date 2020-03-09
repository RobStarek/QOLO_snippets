# QOLO_snippets
A repository for sharing QOLO code snippets.
I would like to store and share useful code-snippets for QOLO lab, mainly for discrete-variable quantum information science.

***

## MaxLik.py
This module provides a simple numpy-implementation of Maximum likelihood reconstruction
method [1,2] for reconstructing low-dimensional quantum states and processes (<=6 qubits).

This package is limited to projection and preparation of pure states.

An orderd list of prepared/projected list is provided to MakeRPV() function to
generate auxiliary array of projection-preparation matrices (Rho-Pi vector). 

The Rho-Pi vector is inserted as an argument together with data to Reconstruct() function.
The Reconstruct() function returns reconstructed density matrix.

### Example:
Minimal single-qubit reconstruction from Pauli-state tomography.
```python
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
#Prepare (Rho)-Pi vect, it can be stored and re-used later for more reconstruction.
RPV = MakeRPV(Order, False)
#Run reconstruction
E = Reconstruct(testdata, RPV, 1000, 1e-6)
```
See more examples to learn how to reconstruct multi-qubit state or quantum process.
        
### References:
1. [Fiurasek, Hradil, Maximum-likelihood estimation of quantum processes, Phys. Rev. A 63, 020101(R) (2001)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.63.020101)
2. [Quantum State Estimation. Lecture Notes in Physics (Springer Berlin Heidelberg, 2004), ISBN: 978-3-540-44481-7](https://doi.org/10.1007/b98673)

***

## KetSugar.py
Shorthand notation, basic constants and frequently used function for some basic low-dimensional discrete-variable quantum mechanics. It provides a way to comfortably write a more comprehensive expression and some functions that I used most frequently. 
It can be used as a module with definions, or the functions/constants could be copy-pasted to your code as desired.

### Content
* Pauli-matrix eigenstates (|0/1>, |+/->, |R/L>)
* Computational base state constructor.
* Constructor of qubit kets from Bloch coordinates
* Hermite conjugation (conjugate transpose, dagger)
* Braket inner product <a|b>
* Ketbra outer product |a><b|
* Application of operator U.R.U^{+}
* Expectation value <a|R|a>
* Purity Tr(R^2)
* Fidelity Tr(Sqrt(Sqrt(A).B.Sqrt(B)))
* Overlap Tr(A.B)
* Partial trace over left-most qubit
* Partial trace over right-most qubit
* Partial trace over middle qubit in three-qubit density matrix
* General partial trace of n-qubit density matrix over specified qubits (plus helper functions).

### Example:
A simple example for simplification of GHZ state ket construction
```python
#Comprehensive in-line construction of GHZ-state (|000> + |111>)/sqrt(2)
KetGHZ = (kron(LO, LO, LO) + kron(HI, HI, HI))*2**-.5
RhoGHZ = ketbra(KetGHZ, KetGHZ)
#instead of bit longer
#KetGHZ = (np.kron(LO, np.kron(LO)) + np.kron(HI, np.kron(HI)))*2**-.5
#or two-line
#KetGHZ = np.zeros((8,1))
#KetGHZ[[0,-1],0] = 2**.-5
#or explicit
#KetGHZ = (np.array([1,0,0,0,0,0,0,1])*2**.-5).reshape((8,1))
```

***
## Auxiliaries
### SigRound.py
Significant-digits-rounding snippet.
Round mean value to a certain number of significant digits of the uncertainty.

#### Example
```python
>>> RoundToError(3.141592, 0.01, n=1)
(3.14, 0.01)
>>> FormatToError(3.141592, 0.01, n=1)
'3.14(1)'
>>> FormatToError(314.159, 20, n=1)
'310(20)'
```

***

## WaveplateBox
Toolbox of commonly used function for waveplate-polarization manipulation in Jones formalism.
Contains search routines for arbitrary state preparation and a search for arbitrary single-qubit preparation.


***

## ChoiBox
A little collection of function for creating, finding and transforming unitary to Choi matrices.

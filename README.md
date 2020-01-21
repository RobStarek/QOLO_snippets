# QOLO_snippets
A repository for sharing QOLO code snippets
I want to store and share useful code-snippets for QOLO lab.

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
```ruby
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
        
### References:
1. [Fiurasek, Hradil, Maximum-likelihood estimation of quantum processes, Phys. Rev. A 63, 020101(R) (2001)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.63.020101)
2. [Quantum State Estimation. Lecture Notes in Physics (Springer Berlin Heidelberg, 2004), ISBN: 978-3-540-44481-7](https://doi.org/10.1007/b98673)

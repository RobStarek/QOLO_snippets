"""
A small snippet to make maxlik bootstrapping easier (i.e. without explicit coding) by using decorators.
From a tomogram a set a bootstraps are generated assuming Poissonian statistics
and each sample is then reconstruct using the reconstruction funcion.
It also contains a collection of helper decorators used to apply function on a 
set of bootstraps and some function to evaluate pairwise functions.

### Example use: ###
#Original function took tomogram and reconstructed it into density matrix.
#Decorated function reconstruct the original tomogram, then randomly generates
#many bootstrapping tomograms and reconstructs them as well.
@bootstrap_reconstruction(100)
def process_recon_q1(tomogram):
    return ml.Reconstruct(tomogram, RPV_Q1, ML_ITERS, ML_THR)

#Decorated function returns tuple (seed_purity, std of purity)
#Original function returned just the purity.
@std_bootstraps
@process_bootstraps_qoi
def qoi_purity(chi):
    return ks.Purity(chi).real
"""

import functools
import numpy as np

def bootstrap_reconstruction(samples):   
    """
    Decorate reconstruction function that takes tomogram and returns a matrix.
    Decorated function returns array of matrices, first entry is the original matrix (seed),
    and then there are reconstructed bootstraps.
    """
    def _deco(f):
        @functools.wraps(f)
        def _wrapper(tomogram):
            seed = f(tomogram)
            k = tomogram.shape[0]
            tomo_bootstraps = np.random.poisson(lam = tomogram, size = (samples, k))
            results = [seed]        
            results.extend([reconstruction for reconstruction in map(f, tomo_bootstraps)])
            return np.array(results)
        return _wrapper
    return _deco
    
## Use composition of decorators instead
# def process_bootstraps_qoi_std(qoi_f):
#     @functools.wraps(qoi_f)
#     def _wrapper(data):
#         qois = process_bootstraps_qoi(qoi_f)(data)
#         seed = qois[0]
#         std = np.std(qois)
#         return (seed, std)
#     return _wrapper

def apply_to_bootstraps(func):
    """
    Decorate function that maps a single argument to a value.
    Decorated function acts on an iterable of arguments
    and produces list of values.
    """    
    @functools.wraps(func)
    def _wrapper(data):
        return [func(m) for m in data]
    return _wrapper

def zipped_apply_to_bootstraps(func):
    """
    Decorate function that maps multiple arguments to a single or more value(s).
    Decorated function takes multiple iterables of arguments, zips them 
    and evaluates the original function on the pair of arguments,
    the results are stored in list.
    """        
    @functools.wraps(func)
    def _wrapper(*args):
        return [func(*ms) for ms in zip(*args)]
    return _wrapper

def pairwise_apply_2nd_fixed(func):
    """
    Decorate function that maps multiple arguments to a single or more value(s).
    Decorated function a single iterable of arguments, and other argumens are expected to be fixed.
    Then it works like zipped_apply_to_bootstraps().
    """            
    @functools.wraps(func)
    def _wrapper(arg0, arg1):
        return [func(x, arg1) for x in arg0]
    return _wrapper

def std_bootstraps(func):
    """
    Compose this decorator with apply_to_bootstraps() to make the function evaluate the 
    seed value and the standard deviation.
    """
    @functools.wraps(func)
    def _wrapper(*data):  
        result = func(*data)
        seed = result[0]
        std = np.std(result)
        return (seed, std)
    return _wrapper

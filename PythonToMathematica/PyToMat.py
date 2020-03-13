"""
Conversion of complex ndarrays to Mathematica expressions.
Example:
    >>> a = np.array([1,2,3,4])
    >>> b = np.array([[1,2],[3,4]])
    >>> SaveArraysToMatExps("test.txt", 2, pomA = a, pomB = b)
    Outputs:
    pomA = {1.00*^+00 + 0.00*^+00*I, 2.00*^+00 + 0.00*^+00*I, 3.00*^+00 + 0.00*^+00*I, 4.00*^+00 + 0.00*^+00*I }
    pomB = {
    {1.00*^+00 + 0.00*^+00*I, 2.00*^+00 + 0.00*^+00*I },
    {3.00*^+00 + 0.00*^+00*I, 4.00*^+00 + 0.00*^+00*I }
    }
"""

import numpy as np

def ComplexToMStr(z, digits=16):    
    """
    Convert complex number to Mathematica expression string
    Args:
        z - number to be converted
        digits - number of decimal digits in scientific notation
    Returns:
        string with mathematica expr. representation of z
    """
    sign = '+' if z.imag>=0 else "-"
    restr = (f"%.{digits}e") % z.real
    imstr = (f"%.{digits}e") % np.abs(z.imag)
    restr = restr.replace("e", "*^")
    imstr = imstr.replace("e", "*^")
    return f"{restr} {sign} {imstr}*I"

def ArrToMatStr(array, digits=16, level=0):  
    """
    Represent complex numpy ndarray.
    Warning: recursion is used here.
    Args:
        array - complex ndarray to be represented
        digits - number of decimal digits in scientific notation
        level - internal use only, sets indentation
    Returns:
        string with mathematica expr. representation of array
    """
    if isinstance(array, np.ndarray):
        li = [ArrToMatStr(element, digits, level+1) for element in array]
        if "{" in li[0]:
            arrstr = " "*level+"{\n"+", ".join(li)+" }\n"
        else:
            arrstr = " "*level+"{"+", ".join(li)+" }\n"
        arrstr = arrstr.replace("\n,",",\n")
        return arrstr
    else:
        return ComplexToMStr(array, digits=digits)

def SaveArraysToMatExps(filename, digits = 16, **arrays):
    """
    Save arrays Mathematica representation into text file.
    Args:
        filename - path to file
        digits - number of decimal digits in scientific notation
        **arrays - keyword arguments with arrays and their names in mathematica
    """
    with open(filename, "w") as mf:
        for key in arrays:
            arrstr = f"{key} = {ArrToMatStr(arrays[key], digits)}\n\n"
            mf.write(arrstr)
    
    

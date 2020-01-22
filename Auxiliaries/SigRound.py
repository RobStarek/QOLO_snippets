from math import floor, log10
"""
Significant-digits-rounding snippet.
Round mean value to a certain number of significant digits of the uncertainty.
Example:
>>> RoundToError(3.141592, 0.01, n=1)
(3.14, 0.01)
>>> FormatToError(3.141592, 0.01, n=1)
'3.14(1)'
>>> FormatToError(314.159, 20, n=1)
'310(20)'
"""


def RoundToError(mean, std, n=1):
    """
    Round mean and corresponding deviation to specified number of significant
    digits.
    Args:
        mean: float number
        std: float number
        n: optional number of significant digits, default is 1
    Returns: 
        tuple (rounded_mean, rounded_std)
    """
    significant_order = floor(log10(std))
    digits = (n-1)-significant_order
    rounded_std = round(std, digits)
    rounded_mean = round(mean, digits)
    return rounded_mean, rounded_std


def FormatToError(mean, std, n=1):
    """
    Make string with properly rounded mean and std to specified number
    of significant digits in format mean(std).
    Args:
        mean: float number
        std: float number
        n: optional number of significant digits, default is 1
    Returns: 
        formatted string
    Example:
        FormatToError(3.14159, 0.019, n=1)
        3.14(2) meaning 3.14+/-0.02        
    """
    if std == 0:
        return f"{mean:.3f}(0)"
    else:
        significant_order = floor(log10(abs(std)))
    digits = (n-1)-significant_order
    rounded_std = round(std, digits)
    rounded_mean = round(mean, digits)
    std_repre = int((10**digits)*rounded_std)
    if significant_order > 0:
        format_string = "{:d}"
        rounded_mean = int(rounded_mean)
        std_repre = int(rounded_std)
    else:
        format_string = f"{{:.{digits:d}f}}"
    mean_repre = format_string.format(rounded_mean)
    return f"{mean_repre}({std_repre})"

"""
A simple Python/Numpy implementation of fast gaussian fitting
with prior knowledge of the offset.

References:
H. Guo, IEEE Signal Process. Mag. vol. 28, no. 5,
pp. 134-137, Sep. 2011.
10.1109/MSP.2011.941846
"""

import numpy as np
import numba as nb


@nb.jit(nopython=True)
def make_lse(xpow, y2, lny):
    """
    Produce matrix m and a vector v that define 
    system of equation m @ x = v
    for fast gaussian fitting.
    """
    m00, m01, m02, m12, m22 = (xpow @ (y2.reshape((-1, 1)))).ravel()
    m = np.array([[m00, m01, m02], [m01, m02, m12], [m02, m12, m22]])
    y2lny = y2*lny
    v = np.array(([np.sum(y2lny), (xpow[1]) @ y2lny,
                 (xpow[2]) @ y2lny])).reshape((-1, 1))
    return m, v

# Fast fitting with prior knowledge of the noise


@nb.jit(nopython=True)
def fastfit(xdata, ydata, offset=0, iters_max=5, delta_mse=1e-6):
    """Fast gaussian fit.

    Args:
        xdata (1d real ndarray): data x axis
        ydata (1d real ndarray): data y axis
        offset (int, optional): priorly known (local) background level. Defaults to 0.
        iters_max (int, optional): How many iterations at maximum. Defaults to 5.
        delta_mse (_type_, optional): Threshold to stop iteration is MSE is smaller than this. Defaults to 1e-6.

    Returns:
        amp, mu, sigma: tuple with real fitted parameters
    """
    THR = 0.2 + offset  # *np.max(ydata)
    sely = (ydata-offset) >= THR
    x = xdata[sely].astype(np.float32)
    y = (ydata[sely] - offset).astype(np.float32)
    lny = np.log(y).astype(np.float32)
    # for repated fitting, this could be saved in prior
    x2 = x*x
    x3 = x2*x
    x4 = x3*x
    n = x.shape[0]
    xpow = np.ones((5, n), dtype=np.float32)
    xpow[1] = x
    xpow[2] = x2
    xpow[3] = x3
    xpow[4] = x4
    i = 0
    delta2 = delta_mse + 1
    while (i < iters_max) and (delta2 > delta_mse):
        y2 = y*y
        m, v = make_lse(xpow, y2, lny)
        v_abc = np.linalg.inv(m) @ v
        a, b, c = v_abc.ravel()
        ylast = y
        y = np.exp(a + b*x + c*x2)
        delta = y-ylast
        delta2 = delta @ delta
        i += 1
    a, b, c = v_abc.ravel()
    sigma = np.sqrt(-0.5/c)
    mu = -b/(2*c)
    amp = np.exp(a-(b*b)/(4*c))
    return amp, mu, sigma


# Test/Example

if __name__ == '__main__':
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    def gauss(x, A, mu, sigma, bckg):
        arg = -0.5*((x-mu)**2)/(sigma**2)
        return A*np.exp(arg) + bckg

    def nonlinear_fit(x, y):
        imax = np.argmax(y)
        mu = x[imax]
        bckg = np.min(y)
        amp = y[imax] - bckg
        dx = x[1]-x[0]
        sigma = dx*np.sum(y)/(np.sqrt(2*np.pi)*np.max(y))
        r = curve_fit(gauss, x, y, p0=(amp, mu, sigma, bckg))
        return r[0]

    xd = np.linspace(0, 10, 101)
    amp = 2
    mu = 5
    sig = 0.5
    offset = 0.05
    noise = 0.05
    yd = gauss(xd, amp, mu, sig, offset)
    yd = np.random.normal(yd, noise)

    # use timeit module for benchmark
    famp, fmu, fsig = fastfit(xd, yd, offset, 5, 1e-9)
    f2amp, f2mu, f2sig, bckg2 = nonlinear_fit(xd, yd)
    yfit = gauss(xd, famp, fmu, fsig, offset)
    yfit2 = gauss(xd, f2amp, f2mu, f2sig, bckg2)

    plt.plot(xd, yd, "k.")
    plt.plot(xd, yfit, "r-")
    plt.plot(xd, yfit2, "g--")
    plt.show()

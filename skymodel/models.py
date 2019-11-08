import numpy as np

def from_index(x, x1, y1, index):
    """Calculate flux from measured value and measured/assumed index."""
    return y1*(x/x1)**index

def two_point_index(x1, x2, y1, y2):
    """Calculate spectral index from two measurements."""
    return np.log10(y1/y2)/np.log10(x1/x2)

def powerlaw(x, a, b):
    """Simple powerlaw function."""
    return a*(x**b)

def cpowerlaw(x, a, b, c):
    """Simple curved powerlaw function."""
    return a*(x**b)*np.exp(c*np.log(x)**2)
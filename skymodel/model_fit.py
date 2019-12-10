#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from scipy.optimize import curve_fit

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fit(f, x, y, p0, yerr=None, return_pcov=False):

    if yerr is not None:
        yerr = np.asarray(yerr)

    popt, pcov = curve_fit(f, np.asarray(x), np.asarray(y), 
                           p0=p0,
                           absolute_sigma=True,
                           method="lm",
                           sigma=yerr,
                           maxfev=10000)

    perr = np.sqrt(np.diag(pcov))

    if return_pcov:
        return popt, pcov
    else:
        return popt, perr


# Code equivalent to flux_warp.models

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

def cpowerlaw_from_ref(x, x0, y0, b, c):
    """Simple curved powerlaw function from reference value."""
    return y0*((x**b)/(x0**b)) * (np.exp(c*np.log(x)**2) / np.exp(c*np.log(x0)**2))

def cpowerlaw_amplitude(x0, y0, b, c):
    """Return amplitude of curved powerlaw model."""
    return y0 / (x0**b * np.exp(c*np.log(x0)**2))


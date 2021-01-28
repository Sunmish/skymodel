#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

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
                           maxfev=100000)

    perr = np.sqrt(np.diag(pcov))

    if return_pcov:
        return popt, pcov
    else:
        return popt, perr


# Code equivalent to flux_warp.models

def from_index(x, x1, y1, index):
    """Calculate flux from measured value and measured/assumed index."""
    return y1*(x/x1)**index

def from_index_err(x, x1, y1, index):
    """Calculate error from flux derived from index."""
    return from_index(x, x1, y1[0], index[0])*np.sqrt(np.abs(y1[1]/y1[0])**2 + 
                                                np.abs(np.log(x/x1)*index[1])**2)

def two_point_index(x1, x2, y1, y2):
    """Calculate spectral index from two measurements."""
    return np.log10(y1/y2)/np.log10(x1/x2)

def two_point_index_err(x1, x2, y1, y2):
    """Standard uncertainty on two-point index."""
    return (1./np.log(x1/x2))*np.sqrt(abs(y1[1]/y1[0])**2 + abs(y2[1]/y2[0])**2)

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


def plot(outname, f, x, y, yerr, popt=None):
    """
    """

    plt.close("all")

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.axes([0.1, 0.1*(6./8.), 0.85, 1.-0.15*(6./8.)])

    ax1.errorbar(x, y, yerr=yerr, xerr=None, fmt="o", ecolor="black", 
                 mec="black", mfc="black")
    ax1.plot(x, y, ls="", color="black", marker="o", ms=5.)

    if popt is not None:
        x_fit = np.linspace(min(x), max(x), 1000)
        ax1.plot(x_fit, f(x_fit, *popt), color="black", ls="--")

    plt.xscale("log")
    plt.yscale("log")

    fig.savefig(outname)
    plt.close("all")
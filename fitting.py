#! /usr/bin/env python

# A collection of tools for fitting simple source models.

import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import linregress, t

import matplotlib as mpl
mpl.use("Agg")  # Supercomputer safe!
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)


# ---------------------------------------------------------------------------- #
def from_index(x, x1, y1, index):
    """Calculate flux from measured value and measured/assumed index."""
    return y1*(x/x1)**index


def two_point_index(x1, x2, y1, y2):
    """Calculate spectral index from two measurements."""
    return np.log10(y1/y2)/np.log10(x1/x2)


# def powerlaw(x, a, b):
#     """Simple powerlaw function."""
#     return (10.**b)*x**a

def powerlaw(x, a, b):
    """Simple powerlaw function."""
    return a*(x**b)


# def upowerlaw(x, a, b, ea, eb):
#     """Uncertainty in powerlaw calculation."""
#     f = powerlaw(x, a, b)
#     df = f*np.sqrt(abs(np.log(x)*ea)**2 + 
#                    abs(np.log(10.)*eb)**2)
#     return df


# def cpowerlaw(x, a, b, c):
#     """Simple curved powerlaw function."""
#     return (x**a * np.exp(b*np.log(x)**2 + c))

def cpowerlaw(x, a, b, c):
    """Simple curved powerlaw function."""
    return a*(x**b)*np.exp(c*np.log(x)**2)


# def ucpowerlaw(x, a, b, c, ea, eb, ec):
#     """Uncertainty in simple curved powerlaw calculation."""
#     f = cpowerlaw(x, a, b, c)
#     df = f*np.sqrt(abs(a*x**(-1.)*ea)**2 + 
#                    abs(eb*np.log(x)**2)**2 + 
#                    abs(ec)**2)
#     return df


def cb_cpowerlaw(x, y, yerr, pcov, popt, conf=68.):
    """Calculate confidence band for curved power law model.

    Adapted from:
    https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals

    TODO:
    1. Powerlaw model as well, or general case - second derivatives become a 
    problem in the general case.
    """

    dfda = np.log(x)*cpowerlaw(x, *popt)
    dfdb = np.log(x)**2 * cpowerlaw(x, *popt)
    dfdc = cpowerlaw(x, *popt)

    dfdp = [dfda, dfdb, dfdc]

    alpha = 1. - conf/100.
    prb = 1 - alpha/2.
    n = len(popt)
    N = len(x)
    dof = N - n

    chi2 = np.sum((y - cpowerlaw(x, *popt))**2 / yerr**2)
    redchi2 = chi2 / dof

    df2 = np.zeros(len(dfda))
    for j in range(n):
        for k in range(n):
            df2 += dfdp[j]*dfdp[k]*pcov[j, k]
    df = np.sqrt(redchi2*df2)

    y_model = cpowerlaw(x, *popt)
    tval = t.ppf(prb, dof)
    delta = tval * df
    upperband = y_model + delta
    lowerband = y_model - delta

    return upperband, lowerband


# ---------------------------------------------------------------------------- #
def fit(f, x, y, yerr=None, params=None, return_pcov=False):
    """Fit function `f` to data `x`, `y`, with absolute error `yerr`.

    An initial guess must be supplied if not using function defined here.
    """

        
    if f == powerlaw and params is None:
        params = [1., -1.]
    elif f == cpowerlaw and params is None:
        params = [1., -1., 1.]
    elif params is None:
        raise ValueError("`params` must be supplied if not using one of the "
                         "builtin models: `powerlaw` or `cpowerlaw.")

    if yerr is not None:
        yerr = np.asarray(yerr)

    popt, pcov = curve_fit(f, np.asarray(x), np.asarray(y), params,
                           absolute_sigma=True, 
                           method="lm", 
                           sigma=yerr,
                           maxfev=100000)  # > the default of 800
    perr = np.sqrt(np.diag(pcov))

    if return_pcov:
        return popt, pcov
    else:
        return popt, perr


def plot(outname, f, x, y, yerr, popt, pcov=None):
    """Make a quick plot of the fitting model and data."""

    plt.close("all")

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.axes([0.1, 0.1*(6./8.), 0.85, 1.-0.15*(6./8.)])

    ax1.errorbar(x, y, yerr=yerr, xerr=None, fmt="o", ecolor="black", 
                 mec="black", mfc="black")
    ax1.plot(x, y, ls="", color="black", marker="o", ms=5.)

    x_fit = np.linspace(x[0], x[-1], 1000.)
    ax1.plot(x_fit, f(x_fit, *popt), color="black", ls="--")

    if pcov is not None:
        ub, lb = cb_cpowerlaw(x, y, yerr, pcov, popt, conf=95.)
        ax1.fill_between(x, ub, lb, facecolor="lightgrey", zorder=-1)

    plt.xscale("log")
    plt.yscale("log")

    fig.savefig(outname)
    plt.close("all")


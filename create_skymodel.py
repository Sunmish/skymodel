#!/usr/bin/env python

# Create sky model for calibration from the GLEAM Extra-Galactic Catalogue.
# For getting the primary beam value at a given RA-DEC we use a code from
# "beam_value_at_radec.py" from MWA_Tools.

# Adapted from Thomas Franzen's GLEAM year 2 processing pipeline. 

import os
import sys
import numpy as np
from argparse import ArgumentParser

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
from astropy.time import Time

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import linregress, t

# MWA beam-related imports:
from mwapy import ephem_utils
from mwapy.pb.primary_beam import MWA_Tile_full_EE

MWA = EarthLocation.from_geodetic(lat=-26.703319*u.deg, 
                                  lon=116.67081*u.deg, 
                                  height=377*u.m)

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)


FREQ_LIST = np.array([76., 84., 92., 99., 107., 115., 122., 130., 143.,
                      151., 158., 166., 174., 181., 189., 197., 204., 
                      212., 220., 227.])
FREQ_LIST_STR = ["076", "084", "092", "099", "107", "115", "122", "130", 
                 "143", "151", "158", "166", "174", "181", "189", "197",
                 "204", "212", "220", "227"]




# Fitting functions for modelling sources ------------------------------------ #

def powerlaw(x, a, b):
    """Simple powerlaw function."""
    return (10.**b)*x**a

def upowerlaw(x, a, b, ea, eb):
    """Uncertainty in powerlaw calculation."""
    f = powerlaw(x, a, b)
    df = f*np.sqrt(abs(np.log(x)*ea)**2 + 
                   abs(np.log(10.)*eb)**2)
    return df

def cpowerlaw(x, a, b,c):
    """Simple curved powerlaw function."""
    return (x**a * np.exp(b*np.log(x)**2 + c))

def ucpowerlaw(x, a, b, c, ea, eb, ec):
    """Uncertainty in simple curved powerlaw calculation."""
    f = cpowerlaw(x, a, b, c)
    df = f*np.sqrt(abs(a*x**(-1.)*ea)**2 + 
                   abs(eb*np.log(x)**2)**2 + 
                   abs(ec)**2)
    return df

def cb_cpowerlaw(x, y, yerr, pcov, popt, conf=68.):
    """Calculate confidence band for curved power law model.

    Adapted from:
    https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals

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

def gaussian_formatter(name, ra, dec, major, minor, pa, freq, flux, precision=3):
    """Format Gaussian component for skymodel v.1.1 format.

    TODO: incorporate precision argument.
    """


    measurements = ""
    for i in range(len(freq)):
        measurements += "       measurement {{\n" \
                        "           frequency {freq} MHz\n" \
                        "           fluxdensity Jy {flux:.3f} 0.0 0.0 0.0\n" \
                        "       }}\n".format(freq=freq[i], flux=round(flux[i], 3))


    gaussian = "\nsource {{\n" \
                "   name \"{name}\"\n" \
                "       component {{\n" \
                "       type gaussian\n" \
                "       position {ra} {dec}\n" \
                "       shape {major} {minor} {pa}\n" \
                "{measurement}" \
                "   }}\n" \
                "}}\n".format(name=name, ra=ra, dec=dec, major=major, 
                             minor=minor, pa=pa, measurement=measurements)
    return gaussian


def get_exclusion_coords(skymodel):
    """Get coordinates to create exclusion zone around."""

    era, edec = [], []
    for s in skymodel:
        with open(skymodel, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "position" in line:
                    bits = line.split()
                    era.append(bits[1])
                    edec.append(bits[2])

    exclusion_coords = SkyCoord(ra=era, dec=edec)

    return exclusion_coords


def beam_value(ra, dec, t, delays, freq, interp=True):
    """Get real XX and real YY beam value at given RA and Dec.

    Adapted from `beam_value_at_radec.py` by N. Hurley-Walker.

    Parameters
    ----------
    ra : float or np.ndarry or list
        RA to get beam value at.
    dec : float or np.ndarray or list
        Dec. to get beam value at.
    t : astropy.time.Time object
        Time object in 'isot' format and 'utc' scale.
    delays : list
        List of 16 dipole delays.
    freq : float
        Frequency at which to get beam value, in Hz.
    interp : bool, optional
        Passed to MWA_Tile_full_EE. [Default True]

    Returns
    -------
    np.ndarray
        Array of real XX beam values for given RA, Dec. values.
    np.ndarray
        Array of real YY beam values for given RA, Dec. values.


    """


    if len(delays) != 16:
        raise ValueError("There are only {} delays: there should be 16.".format(
                         len(delays)))

    if not hasattr(ra, "__getitem__"):
        ra = np.array([ra])
        dec = np.array([dec])
    elif not isinstance(ra, np.ndarray):
        ra = np.asarray(ra)
        dec = np.asarray(dec)

    radec = SkyCoord(ra*u.deg, dec*u.deg)
    
    altaz = radec.transform_to(AltAz(obstime=t, location=MWA))
    za = (np.pi/2.) - altaz.alt.rad
    az = altaz.az.rad

    if not hasattr(za, "__getitem__"):
        za = np.array([za])
        az = np.array([az])


    rX, rY = MWA_Tile_full_EE(za=[za],
                              az=[az],
                              freq=freq,
                              delays=delays,
                              interp=interp,
                              pixels_per_deg=10)  # Slightly highter pixels_per_deg
                                                  # than default. Better?

    return rX[0], rY[0]


def create_model(catalogue, metafits, outname,  threshold=1., ratio=1.1, 
                 radius=120., nmax=500, plot=False, pb=True, freq=None, decorrect=False,
                 exclude_coords=None, exclusion_zone=5., return_catalogue=False, 
                 weight=False):
    """Main function."""

    if not os.path.exists(catalogue):
        logging.error("GLEAM catalogue not found or not specified.")
        sys.exit(1)

    if not os.path.exists(metafits):
        logging.error("metafits file does not exist or not specified.")
        sys.exit(1)

    with fits.open(metafits) as mfits:
        t = Time(mfits[0].header["DATE-OBS"], format="isot", scale="utc")
        delays = [int(d) for d in mfits[0].header["DELAYS"].split(",")]
        freq = mfits[0].header["FREQCENT"]
        ra_pnt = mfits[0].header["RA"]
        dec_pnt = mfits[0].header["DEC"]


    GLEAM = fits.open(catalogue)[1].data
    logging.info("GLEAM sources: {0}".format(len(GLEAM)))
    # Initial flux cut:
    GLEAM = GLEAM[np.where(GLEAM["Fpwide"] > threshold)[0]]
    logging.info("GLEAM sources after flux density cut: {0}".format(
                 len(GLEAM)))
    # Cut out extended sources:
    # TODO: remove this - we WANT the extended Gaussian sources.
    # Integrated flux densities are wonky in a lot of cases so can't remove this
    # for now.
    GLEAM = GLEAM[np.where((GLEAM["awide"]*GLEAM["bwide"]) / 
                           (GLEAM["psfawide"]*GLEAM["psfbwide"]) < ratio)[0]]
    logging.info("GLEAM sources after extended source cut: {0}".format(
                 len(GLEAM)))

    coords = SkyCoord(ra=GLEAM["RAJ2000"], dec=GLEAM["DEJ2000"], 
                      unit=(u.deg, u.deg))
    pcentre = SkyCoord(ra=ra_pnt, dec=dec_pnt, unit=(u.deg, u.deg))

    seps = pcentre.separation(coords).value

    # Cut out sources outside of radius:
    GLEAM = GLEAM[np.where(seps < radius)[0]]
    logging.info("GLEAM sources after radial cut: {0}".format(len(GLEAM)))

    # Now go through all sources and do the following:
    # Model SED as powerlaw or curved powerlaw
    # If it's a good fit, keep.
    # Predict value at specified frequency.
    # De-corrected for primary beam response.
    # Predict values at other frequencies - overwrite existing GLEAM 
    # measurements with these predictions.


    for i in range(len(GLEAM)):


        if exclude_coords is not None:
            coords_ = SkyCoord(ra=GLEAM["RAJ2000"][i], dec=GLEAM["DEJ2000"][i],
                        unit=(u.deg, u.deg))    
            excl_seps = coords_.separation(exclude_coords).value
            if excl_seps.any() < exclusion_zone/60.:

                logging.debug("{0} within exclusion zone.".format(i))
                GLEAM["Fintwide"][i] = 0.

                continue

        tmp_flux, tmp_eflux, tmp_freq, tmp_int = [], [], [], []

        for f in range(len(FREQ_LIST_STR)):
            fp = GLEAM["Fp{0:s}".format(FREQ_LIST_STR[f])][i]
            fi = GLEAM["Fint{0:s}".format(FREQ_LIST_STR[f])][i]
            ep = GLEAM["e_Fp{0:s}".format(FREQ_LIST_STR[f])][i]

            if fp > 0.:  # Priorized fitting creates negative flux densities.
                tmp_flux.append(fp)
                tmp_int.append(fi)
                tmp_eflux.append(np.sqrt(ep**2 + (0.02*fp)**2))
                tmp_freq.append(FREQ_LIST[f])

        if len(tmp_flux) > 10:  # This could be set higher or lower.

            try:

                tmp_flux = np.asarray(tmp_flux)
                tmp_freq = np.asarray(tmp_freq)
                tmp_eflux = np.asarray(tmp_eflux)
                tmp_int = np.asarray(tmp_int)

                #
                # TODO: check for curvature? Apply powerlaw only in that case.
                #

                popt, pcov = curve_fit(cpowerlaw, tmp_freq, tmp_flux, 
                                       [-0.8, 1, 1], sigma=tmp_eflux, 
                                       absolute_sigma=True, method="lm")
                perr = np.sqrt(np.diag(pcov))

                if np.isnan(pcov[0, 0]):
                    logging.debug("pcov[0, 0] is nan for {0}".format(i))
                    raise RuntimeError
                elif perr[0]/popt[0] > 0.25:  # Poor fit?
                    logging.debug("perr is too high for {0}".format(i))
                    raise RuntimeError

            except RuntimeError:
                logging.debug("Fitting error(s) for {0}".format(i))
                GLEAM["Fintwide"][i] = 0.

            else:

                predicted_flux = cpowerlaw(float(freq), *popt)
                GLEAM["Fintwide"][i] = predicted_flux  # Easier than creating a new thingy
                GLEAM["Fpwide"][i] = predicted_flux  # in case pb = False

                for f in range(len(FREQ_LIST_STR)):
                    GLEAM["Fp{0:s}".format(FREQ_LIST_STR[f])][i] = cpowerlaw(FREQ_LIST[f], *popt)


                if plot:

                    if not os.path.exists("./test"):
                        os.mkdir("./test")

                    ub, lb = cb_cpowerlaw(tmp_freq, tmp_flux, tmp_eflux, pcov,
                                          popt, conf=95.)

                    plt.close("all")
                    fig = plt.figure(figsize=(8, 8))
                    ax = plt.axes([0.1, 0.1, 0.8, 0.8])
                    oname = "{0}_fit.png".format(GLEAM["GLEAM"][i])
                    plt.errorbar(tmp_freq, tmp_flux, yerr=tmp_eflux, xerr=None,
                                 fmt="o", ecolor="black", mec="black", mfc="black")
                    plt.plot(tmp_freq, tmp_flux, ls="", color="black", marker="o",
                             ms=5.)
                    plt.plot(tmp_freq, tmp_int, ls="", color="red", marker="x", 
                             ms=5.)
                    x_fit = np.linspace(tmp_freq[0], tmp_freq[-1], 1000)
                    plt.plot(x_fit, cpowerlaw(x_fit, *popt), color="black", 
                             ls="--")
                    
                    plt.fill_between(tmp_freq, ub, lb, facecolor="lightgrey", 
                                     zorder=-1)

                    plt.xscale("log")
                    plt.yscale("log")
                    ax.text(0.5, 0.95, GLEAM["GLEAM"][i], transform=ax.transAxes, 
                            horizontalalignment="center")
                    plt.savefig("./test/"+oname)

                    plt.close()

        else:

            logging.debug("Too few points for {0}".format(i))
            GLEAM["Fintwide"][i] = 0.

    GLEAM = GLEAM[np.where(GLEAM["Fintwide"] > 0.)[0]]
    logging.info("GLEAM sources after modelling: {0}".format(len(GLEAM)))

    xx, yy = beam_value(GLEAM["RAJ2000"], GLEAM["DEJ2000"], t, delays,
                        freq*1.e6)
    ii = (xx + yy)*0.5
    GLEAM["Fpwide"] *= ii
    GLEAM = np.delete(GLEAM, np.where(np.isnan(GLEAM["Fpwide"]))[0])
    GLEAM = GLEAM[np.where(GLEAM["Fpwide"] > threshold)]
    

    # # For some reason a whole bunch of nans creep in. 
    # GLEAM = np.delete(GLEAM, np.where(np.isnan(GLEAM["Fpwide"]))[0])

    # GLEAM = GLEAM[np.where(GLEAM["Fintwide"] > 0.)[0]]
    logging.info("GLEAM sources after applying PB: {0}".format(len(GLEAM)))


    if return_catalogue:
        return GLEAM

    # Now try to cut down catalogue to nmax number of sources:
    if len(GLEAM) > nmax:
        logging.info("More than nmax={0} sources in catalogue - trimming.".format(nmax))
        threshold = round(np.sort(GLEAM["Fpwide"])[::-1][nmax-1], 1)
        GLEAM = GLEAM[np.where(GLEAM["Fpwide"] > threshold)[0]]
        logging.info("GLEAM sources after new flux treshold of {0} Jy: {1}".format(
                     threshold, len(GLEAM)))
    


    with open(outname, "w+") as o:
        o.write("skymodel fileformat 1.1\n")

        for i in range(len(GLEAM)):
            tmp_flux, tmp_freq = [], []
            for f in range(len(FREQ_LIST_STR)):
                tmp_freq.append(FREQ_LIST[f])
                tmp_flux.append(GLEAM["Fp{0:s}".format(FREQ_LIST_STR[f])][i])

            c = SkyCoord(ra=GLEAM["RAJ2000"][i], dec=GLEAM["DEJ2000"][i], 
                         unit=(u.deg, u.deg))
            r, d = c.to_string("hmsdms").split()
            gformat = gaussian_formatter(GLEAM["GLEAM"][i], r, d, 
                                         GLEAM["awide"][i], GLEAM["bwide"][i],
                                         GLEAM["pawide"][i], flux=tmp_flux, 
                                         freq=tmp_freq)
            o.write(gformat)

    with open(outname[:-4]+".csv", "w+") as o:
        # Write a normal .csv file for use with, e.g., topcat.
        o.write("RA,DEC,maj,min,pa,flux\n")

        for i in range(len(GLEAM)):
            o.write("{0},{1},{2},{3},{4},{5}\n".format(GLEAM["RAJ2000"][i],
                                                       GLEAM["DEJ2000"][i],
                                                       GLEAM["awide"][i],
                                                       GLEAM["bwide"][i],
                                                       GLEAM["pawide"][i],
                                                       GLEAM["Fintwide"][i]))

        





def main():

    parser = ArgumentParser(description="Create sky model from GLEAM.")
    parser.add_argument("-g", "--catalogue", "--gleam", dest="catalogue", 
                      default=None, help="Input GLEAM catalogue location.")
    parser.add_argument("-m", "--metafits", dest="metafits", default=None,
                      help="Name/location of metafits file for observation.")
    parser.add_argument("-o", "--outname", dest="outname", default=None,
                      help="Output skymodel name.")
    parser.add_argument("-f", "--freq", dest="freq", default=None,
                      help="Central frequency of image/dataset in MHz.")
    # Extra options:
    parser.add_argument("-t", "--threshold", dest="threshold", default=1., type=float, 
                      help="Threshold below which to cut sources [1 Jy].")
    parser.add_argument("-r", "--radius", dest="radius", default=120., type=float, 
                      help="Radius within which to select sources [120 deg].")
    parser.add_argument("--ratio", dest="ratio", default=1.2, type=float, 
                      help="Ratio of source size to beam shape to determine "
                           "if point source [1.2].")
    parser.add_argument("-n", "--nmax", dest="nmax", default=500, type=int, 
                      help="Max number of sources to return. The threshold is "
                      "recalculated if more sources than nmax are found above it.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no_pb", dest="pb", action="store_false")
    parser.add_argument("-x", "--exclude_model", dest="exclude", default=None,
                      help="Skymodel v1.1 format file with existing models. These"
                           " will be create an exclusion zones of 1 deg around these"
                           " sources.", nargs="*")
    parser.add_argument("-w", "--weight", action="store_true", help="Weight apparent"
                      " fluxes by distance from pointing centre to try and "
                      "include more mainlobe sources than sidelobe source "
                      "(especially at higher frequencies).")

    options = parser.parse_args()

    if options.catalogue is None:
        logging.error("GLEAM catalogue not supplied.")
        sys.exit(1)
    elif options.metafits is None:
        logging.error("Metafits file not supplied.")
        sys.exit(1)


    if options.outname is None:
        options.outname = options.metafits.replace("metafits", "_skymodel.txt")

    if options.exclude is not None:
        exclusion_coords = get_exclusion_coords(options.exclude)
    else:
        exclusion_coords = None
    
    if options.pb:
        logging.info("Attenuating by the primary beam response.")

    create_model(options.catalogue, options.metafits, options.outname,  
                 options.threshold, options.ratio, options.radius, options.nmax, 
                 options.plot, options.pb, options.freq, decorrect=False, 
                 exclude_coords=exclusion_coords, weight=options.weight)



if __name__ == "__main__":
    main()
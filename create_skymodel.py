#! /usr/bin/env python

# Create sky model for calibration from the GLEAM Extra-Galactic Catalogue.
# Adapted from Thomas O. Franzen's GLEAM year 2 processing pipeline. 

import os
import sys
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
from astropy.time import Time

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)

from .get_beam import beam_value
from . import fitting
from .parsers import parse_metafits


FREQ_LIST = np.array([76., 84., 92., 99., 107., 115., 122., 130., 143.,
                      151., 158., 166., 174., 181., 189., 197., 204., 
                      212., 220., 227.])
FREQ_LIST_STR = ["076", "084", "092", "099", "107", "115", "122", "130", 
                 "143", "151", "158", "166", "174", "181", "189", "197",
                 "204", "212", "220", "227"]

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
        with open(s, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "position" in line:
                    bits = line.split()
                    era.append(bits[1])
                    edec.append(bits[2])

    exclusion_coords = SkyCoord(ra=era, dec=edec)

    return exclusion_coords


def create_model(catalogue, metafits, outname,  \
                 threshold=1., ratio=1.1, radius=120., nmax=500, plot=False, \
                 exclude_coords=None, exclusion_zone=5., \
                 return_catalogue=False, weight=False):
    """Create a GLEAM skymodel.

    Parameters
    ----------
    catalogue : str
        The filepath to the GLEAM catalogue. 
    metafits : str
        The filepath to the metafits file for the observation.
    outname : str
        Output filename.
    threshold : float, optional
        The apparent brightness threshold. [Default 1 Jy]
    ratio : float, optional
        The ratio of peak/int to determine point-source nature. [Default 1.1]
    radius: float, optional
        The radius around the pointing centre to search for sources. [Default 120 deg]
    nmax : int, optional
        The maximum number of sources to include in the model. If there are more
        sources than nmax, the threshold is adjusted to restrict the model to 
        nmax sources. [Default 500]
    plot : bool, optional
        Select True if wanting plots of each calibrator source. [Default False]
    exclude_coords : list of SkyCoord objects
        Specify list of coordinates to create an exclusion area around. [Default None]
    exlclusion_zone : float, optional
        Specify zone around exclude_coords to exlude from the model. [Default 5 arcmin]
    return_catalogue : bool, optional
        Switch True if wanting to return the catalogue as an object. [Default False]
    weight : bool, optional
        NOT YET IMPLEMENTED.

    Both an AO-style model and a simple csv are written out. 
    """

    if not os.path.exists(catalogue):
        logging.error("GLEAM catalogue not found or not specified.")
        sys.exit(1)

    if not os.path.exists(metafits):
        logging.error("metafits file does not exist or not specified.")
        sys.exit(1)

    t, delays, freq, pnt = parse_metafits(metafits)
    freq /= 1.e6

    GLEAM = fits.open(catalogue)[1].data
    logging.info("GLEAM sources: {0}".format(len(GLEAM)))
    # Initial flux cut:
    GLEAM = GLEAM[GLEAM["Fpwide"] > threshold]
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
    pcentre = SkyCoord(ra=pnt[0], dec=pnt[1], unit=(u.deg, u.deg))

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

                popt, pcov = fitting.fit(f=fitting.cpowerlaw, 
                                         x=tmp_freq,
                                         y=tmp_flux,
                                         yerr=tmp_eflux,
                                         params=[-0.7, 1., 1.],
                                         return_pcov=True)
                perr = np.sqrt(np.diag(pcov))

                if np.isnan(pcov[0, 0]):
                    logging.debug("pcov[0, 0] is nan for {0}".format(i))
                    raise RuntimeError
                elif perr[0]/popt[0] > 0.5:  # Poor fit?
                    logging.debug("perr is too high for {0}".format(i))
                    raise RuntimeError

            except RuntimeError:
                logging.debug("Fitting error(s) for {0}".format(i))
                GLEAM["Fintwide"][i] = 0.

            else:

                predicted_flux = fitting.cpowerlaw(float(freq), *popt)
                GLEAM["Fintwide"][i] = predicted_flux  
                GLEAM["Fpwide"][i] = predicted_flux

                for f in range(len(FREQ_LIST_STR)):
                    GLEAM["Fp{0:s}".format(FREQ_LIST_STR[f])][i] \
                        = fitting.cpowerlaw(FREQ_LIST[f], *popt)


                if plot:

                    if not os.path.exists("./plots"):
                        os.mkdir("./plots")

                    fitting.plot("./plots/{}_fit.png".format(GLEAM["GLEAM"][i]),
                                 f=fitting.cpowerlaw,
                                 x=tmp_freq,
                                 y=tmp_flux,
                                 yerr=tmp_eflux,
                                 popt=popt,
                                 pcov=pcov)

        else:

            logging.debug("Too few points for {0}".format(i))
            GLEAM["Fintwide"][i] = 0.

    GLEAM = GLEAM[GLEAM["Fintwide"] > 0.]
    logging.info("GLEAM sources after modelling: {0}".format(len(GLEAM)))

    stokesI = beam_value(GLEAM["RAJ2000"], GLEAM["DEJ2000"], t, delays,
                         freq*1.e6, return_I=True)

    GLEAM["Fpwide"] *= stokesI
    GLEAM = GLEAM[np.isfinite(GLEAM["Fpwide"])]

    # The final threshold ensure sources at the given frequency are above
    # the user-specified threshold too.
    GLEAM = GLEAM[GLEAM["Fpwide"] > threshold]
    

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

        
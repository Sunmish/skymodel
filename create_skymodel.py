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



def point_formatter(name, ra, dec, freq, flux, precision=3):
    """Format point source for ao-cal skymodel."""

    measurements = ""
    for i in range(len(freq)):
        measurements += "measurement {{\n" \
                       "frequency {freq} MHz\n" \
                       "fluxdensity Jy {flux:.3f} 0.0 0.0 0.0 \n" \
                       "}}\n".format(freq=freq[i], flux=round(flux[i], 3))

    point = "\nsource {{\n" \
            "name \"{name}\"\n" \
            "component {{\n" \
            "type point\n" \
            "position {ra} {dec}\n" \
            "{measurement}" \
            "}}\n" \
            "}}\n".format(name=name, ra=ra, dec=dec, measurement=measurements)

    return point



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
                 return_catalogue=False, weight=False, curved=True):
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
    # First get closest subband to make flux cut on:
    fidx = (np.abs(FREQ_LIST - freq)).argmin()
    at_freq = FREQ_LIST_STR[fidx]


    GLEAM = GLEAM[GLEAM["Fp"+at_freq] > threshold]
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

    seps = pnt.separation(coords).value

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
            excl_seps = coords_.separation(exclude_coords)
            if (excl_seps.value < exclusion_zone/60.).any():

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

                if curved:
                    model = fitting.cpowerlaw
                    param0 = [1., -1., 1.]
                else:
                    model = fitting.cpowerlaw
                    param0 = [1., -1.]

                popt, pcov = fitting.fit(f=model, 
                                         x=tmp_freq,
                                         y=tmp_flux,
                                         yerr=tmp_eflux,
                                         params=param0,
                                         return_pcov=True)
                perr = np.sqrt(np.diag(pcov))

                if np.isnan(pcov[1, 1]):
                    raise RuntimeError
                elif abs(perr[1]/popt[1]) > 1:  # Poor fit?
                    raise RuntimeError

            except RuntimeError:
                GLEAM["Fintwide"][i] = 0.

            else:

                predicted_flux = model(float(freq), *popt)
                GLEAM["Fintwide"][i] = predicted_flux  
                GLEAM["Fpwide"][i] = predicted_flux

                for f in range(len(FREQ_LIST_STR)):
                    GLEAM["Fp{0:s}".format(FREQ_LIST_STR[f])][i] \
                        = fitting.cpowerlaw(FREQ_LIST[f], *popt)


                # if plot:

                #     if not os.path.exists("./plots"):
                #         os.mkdir("./plots")

                #     fitting.plot("./plots/{}_fit.png".format(GLEAM["GLEAM"][i]),
                #                  f=fitting.cpowerlaw,
                #                  x=tmp_freq,
                #                  y=tmp_flux,
                #                  yerr=tmp_eflux,
                #                  popt=popt,
                #                  pcov=pcov)

        else:

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
    
    logging.info("Maximum brightness source: {}".format(max(GLEAM["Fpwide"])))

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



def create_ns_model(table, metafits, outname=None, alpha=None, a_cut=1., 
                    s200_cut=1., s1400_cut=0.1, s843_cut=0.1, s150_cut=0.1,
                    exclude_coords=None, exclusion_zone=1., d_limit=(-90, 90),
                    radius=180., nmax=200):
    """Create a skymodel using a pre-made catalogue.

    This requires a very specific format.

    The FITS binary table requires columns of the form:
    # ra | dec | a_p | b_p | a_c ... c_c | S1400 | S843 | S200 | S150

    where a/b_p are model parameters for a power law fit, and
    a/b/c_c are model parameters for a curved powerlaw fit.

    These are used to estimate flux density at the required frequency,
    with preference to the curved powerlaw --> powerlaw --> assumed index given
    by `alpha`. If `alpha` is None, and no model parameters are given for a 
    source then that source is not used in the model.

    Parameters
    ----------
    table : str
        Filepath for a FITS binary table.
    metafits : str
        Filepath for an observation's metafits file.
    outname : str, optional
        Output file name. Default is created from the metafits name.
    alpha : float, optional
        If a source does not have model parameters, this spectral index,
        long with S200, is used to estimate the flux density.
    a_cut : float, optional
        Apparent brightness cut.
    sN_cut : float, optional
        Flux density cut to make prior to esimating the flux density.
    exclude_coords : list or array, optional
        List of SkyCoord objects describing sources to exclude.
    exclusion_zone : float, optional
        Zone in arcmin around sources specified in `exclude_coords` to 
        exclude sources.
    d_limits : tuple, optional
        Limiting declination values.

    """

    if outname is None:
        outname = metafits.replace(".metafits", "_skymodel.txt")

    t, delays, freq, pnt = parse_metafits(metafits)
    freq /= 1.e6

    catalogue = fits.open(table)[1].data

    logging.info("Total NS sources: {}".format(len(catalogue)))

    freqs = {"S200": s200_cut,
             "S1400": s1400_cut*1000.,
             "S843": s843_cut*1000.,
             "S150": s150_cut*1000.
             }
    
    for f in freqs.keys():
        c = catalogue.field(f).copy()
        c[~np.isfinite(c)] = 1.e30
        catalogue = catalogue[c > freqs[f]]
        logging.info("NS sources after {} cut: {}".format(f, len(catalogue)))

    catalogue = catalogue[np.where((d_limit[0] <= catalogue.field("dec")) &
                                   (catalogue.field("dec") <= d_limit[1]))[0]]
    logging.info("NS sources after declination cut: {}".format(len(catalogue)))

    coords = SkyCoord(ra=catalogue.field("ra"), dec=catalogue.field("dec"), 
                      unit=(u.deg, u.deg))

    seps = pnt.separation(coords).value
    catalogue = catalogue[np.where(seps < radius)[0]]
    logging.info("Total NS sources after radial cut: {}".format(len(catalogue)))

    coords = SkyCoord(ra=catalogue.field("ra"), dec=catalogue.field("dec"), 
                      unit=(u.deg, u.deg)) 

    
    stokesI = beam_value(catalogue.field("ra"), catalogue.field("dec"),
                         t, delays, freq*1.e6, return_I=True)

    if alpha is None:
        alpha = np.nanmean(catalogue.field("beta_p"))

    for i in range(len(catalogue)):
        # Now for to the long and painful part:

        row = catalogue[i]

        if not np.isnan(row["alpha_c"]):

            catalogue[i]["S200"] = fitting.cpowerlaw(freq,
                                                     a=row["alpha_c"],
                                                     b=row["beta_c"],
                                                     c=row["gamma_c"])/1000.

        elif not np.isnan(row["alpha_p"]):

            catalogue[i]["S200"] = fitting.powerlaw(freq,
                                                    a=row["alpha_p"],
                                                    b=row["beta_p"])/1000.

        elif alpha is not None:

            catalogue[i]["S200"] = (row["S200"]*(freq/200.)**alpha)

        else:

            catalogue[i]["S200"] = np.nan

        # Attenuate by the primary beam
        catalogue[i]["S1400"] = catalogue[i]["S200"]
        catalogue[i]["S200"] *= stokesI[i]



    catalogue = catalogue[np.isfinite(catalogue.field("S200"))]

    logging.info("Total NS sources after estimating flux density: {}".format(len(catalogue)))

    catalogue = catalogue[catalogue.field("S200") > a_cut]

    logging.info("Total NS sources after apparent brightness cut: {}".format(len(catalogue)))

    if len(catalogue) > nmax:
        logging.info("More than nmax={0} sources in catalogue - trimming.".format(nmax))
        a_cut = round(np.sort(catalogue.field("S200"))[::-1][nmax-1], 1)
        catalogue = catalogue[catalogue.field("S200") > a_cut]
        logging.info("NS sources after new flux treshold of {0} Jy: {1}".format(
                     a_cut, len(catalogue)))


    coords = SkyCoord(ra=catalogue.field("ra"), dec=catalogue.field("dec"), 
                      unit=(u.deg, u.deg))

    for i in range(len(catalogue)):

        if exclude_coords is not None:

            excl_seps = coords[i].separation(exclude_coords)
            if (excl_seps.value < exclusion_zone/60.).any():
                logging.debug("{} within exclusion zone.".format(i))
                catalogue[i]["S200"] = np.nan
                continue

            else:
                logging.debug("{} not within exclude coords".format(i))

    catalogue = catalogue[np.isfinite(catalogue.field("S200"))]


    coords = SkyCoord(ra=catalogue.field("ra"), dec=catalogue.field("dec"), 
                      unit=(u.deg, u.deg))

    # logging.info("Maximum brightness source: {:.2f}".format(round(max(catalogue.field("S200"), 2))))

    with open(outname, "w+") as o:
        o.write("skymodel fileformat 1.1\n")

            
        for i in range(len(catalogue)):



            r, d = coords[i].to_string("hmsdms").split()

            flux = []
            for f in FREQ_LIST:
                if not np.isnan(catalogue[i]["alpha_c"]):
                    flux.append(fitting.cpowerlaw(f, 
                                                  a=catalogue[i]["alpha_c"],
                                                  b=catalogue[i]["beta_c"],
                                                  c=catalogue[i]["gamma_c"])/1000.)
                elif not np.isnan(catalogue[i]["alpha_p"]):
                    flux.append(fitting.powerlaw(f,
                                                 a=catalogue[i]["alpha_p"],
                                                 b=catalogue[i]["beta_p"])/1000.)
                else:
                    flux.append(catalogue[i]["S1400"]*(f/freq)**alpha)

            pformat = point_formatter(name="Source{}".format(i),
                                      ra=r,
                                      dec=d,
                                      freq=FREQ_LIST,
                                      flux=flux)

            o.write(pformat)




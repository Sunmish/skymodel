#! /usr/bin/env python

# Create sky model for calibration from the GLEAM Extra-Galactic Catalogue.
# Adapted from Thomas O. Franzen's GLEAM year 2 processing pipeline. 

import os
import sys
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, vstack, Column
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
from astropy.time import Time

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)

from skymodel.get_beam import beam_value, find_lobes, make_beam_image
from skymodel.parsers import parse_metafits

from skymodel.model_fit import cpowerlaw, cpowerlaw_amplitude, powerlaw, from_index


FREQ_LIST = np.array([76., 84., 92., 99., 107., 115., 122., 130., 143.,
                      151., 158., 166., 174., 181., 189., 197., 204., 
                  212., 220., 227.])
FREQ_LIST_STR = ["076", "084", "092", "099", "107", "115", "122", "130", 
                 "143", "151", "158", "166", "174", "181", "189", "197",
                 "204", "212", "220", "227"]

# Coordinates from Hurley-Walker et al. 2017 and radius +0.5 degrees.
LMC = {"coords": SkyCoord(ra="05h23m35s", dec="-69d45m22s", unit=(u.hourangle, u.deg)),
       "radius": 6.0,
       "name": "LMC"}

SMC = {"coords": SkyCoord(ra="00h52m38s", dec="-72d48m01s", unit=(u.hourangle, u.deg)),
       "radius": 3.0,
       "name": "SMC"}

PKS0521 = {"coords": SkyCoord(ra="05h22m55.98s", dec="-36d27m36.541s", 
                             unit=(u.hourangle, u.deg)),
           "radius": 10./60.,
           "name": "PKS0521"}

SOURCES = [LMC, SMC, PKS0521]

# ---------------------------------------------------------------------------- #

def gaussian_formatter(name, ra, dec, major, minor, pa, freq, flux, precision=3):
    """Format Gaussian component for skymodel v.1.1 format.

    TODO: incorporate precision argument.
    """


    try:
        freq = [f for f in freq]
    except TypeError:
        freq = [freq]
        flux = [flux]


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
    if era != []:

        exclusion_coords = SkyCoord(ra=era, dec=edec,
                                    unit=(u.hourangle, u.deg))

        return exclusion_coords

    else:
        return None



def check_keys(table, *keys):
    """Wrapper to check existence of keys/columns in a table."""

    for key in keys:
        if key not in table.columns:
            raise KeyError("{} not in model catalogue!".format(key))



def create_all_skymodel(table, metafits, outname=None, threshold=1.,
                        ref_threshold=0., exclude_coords=None, exclusion_zone=1.,
                        bright_exclusion_zone=10.,
                        d_limit=(-90, 90), radius=180., nmax=200, ref_key="S154",
                        ignore_magellanic=False, index_limits=[-3., 2.], 
                        curved=False, flux0=None, freq0=None, alpha0=-0.77, powerlaw_amplitude=None,
                        powerlaw_index=None, powerlaw_curvature=None,
                        ra_key="ra", dec_key="dec",
                        nlobes=1,
                        nfreqs_to_predict=10):
    """
    """


    if flux0 is not None:
        ref_key = flux0

    if outname is None:
        outname = metafits.replace(".metafits", "_skymodel.txt")

    t, delays, freq, pnt = parse_metafits(metafits)
    freq /= 1.e6
    if freq0 is None:
        ref_freq = float(ref_key.replace("S", ""))
    else:
        ref_freq = freq0

    if nlobes > 1:
        # work out where primary beam sidelobes are, and
        # include only brightest sidelobes with radius check
        beam_image = make_beam_image(t, delays, freq*1e6,
            ra=pnt.ra.value,
            return_hdu=True)
        lobe_table = find_lobes(beam_image, perc=0.1, return_table=True)
        centres = SkyCoord(
            ra=np.asarray(lobe_table[:nlobes]["ra"]),
            dec=np.asarray(lobe_table[:nlobes]["dec"]),
            unit=(u.deg, u.deg)
        )
    else:
        centres = [pnt]

    # catalogue = fits.open(table)[1].data
    catalogue = Table.read(table).filled(np.nan)

    logging.info("Total sources: {}".format(len(catalogue)))

    model0 = model1 = model2 = model3 = model4 = False

    if None not in [powerlaw_amplitude, powerlaw_curvature, powerlaw_index]:
        check_keys(catalogue, powerlaw_amplitude, powerlaw_curvature, powerlaw_index)
        logging.info("evaluating models using curved powerlaw model.")
        model0 = True
    if None not in [powerlaw_curvature, powerlaw_index, flux0, freq0]:
        check_keys(catalogue, powerlaw_curvature, powerlaw_index, flux0)
        logging.info("evaluating models using curved powerlaw model from a "
                    "reference frequency.")
        model1 = True
    if None not in [powerlaw_amplitude, powerlaw_index]:
        check_keys(catalogue, powerlaw_amplitude, powerlaw_index)
        logging.info("evaluating models using a generic power law model.")
        model2 = True
    if None not in [powerlaw_index, flux0, freq0]:
        check_keys(catalogue, powerlaw_index, flux0)
        logging.info("evaluating models using a generic power law model from a "
                    "reference frequency.")
        model3 = True
    if None not in [flux0, freq0, alpha0]:
        check_keys(catalogue, flux0)
        logging.info("evaluating models using a generic power law from a "
                    "reference frequency and assumed spectral index.")
        model4 = True
    if not (model0 or model1 or model2 or model3 or model4):
        raise RuntimeError("not enough information to evaluate models!")


    catalogue = catalogue[np.isfinite(catalogue[ref_key])]
    logging.info("Sources after nan removal: {}".format(len(catalogue)))

    catalogue = catalogue[catalogue[ref_key] > ref_threshold]
    logging.info("Source after reference flux threshold: {}".format(len(catalogue)))

    catalogue = catalogue[np.where((d_limit[0] <= catalogue[dec_key]) &
                                   (catalogue[dec_key] <= d_limit[1]))[0]]
    logging.info("Source after declination cut: {}".format(len(catalogue)))

    coords = SkyCoord(ra=catalogue[ra_key], dec=catalogue[dec_key], 
                      unit=(u.deg, u.deg))
    beam_lobes = []
    for i in range(len(centres)):
        seps = centres[i].separation(coords).value
        lobe_catalogue = catalogue[seps < radius]
        beam_lobes.append(lobe_catalogue)
        logging.info("Sources in lobe {}: {}".format(i, len(lobe_catalogue)))
    
    catalogue = vstack(beam_lobes)
    logging.info("Sources after radial cut(s): {}".format(len(catalogue)))

    coords = SkyCoord(ra=catalogue[ra_key], dec=catalogue[dec_key], 
                      unit=(u.deg, u.deg))

    _, seps2, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)

    coords = coords[seps2.value*60. > exclusion_zone]
    catalogue = catalogue[seps2.value*60. > exclusion_zone]

    logging.info("Sources after creating exclusion zones: {}".format(len(catalogue)))


    at_flux = np.full_like(catalogue[ref_key], np.nan)
    at_flux_atten = np.full_like(catalogue[ref_key], np.nan)
    models = []

    stokesI = beam_value(catalogue[ra_key], catalogue[dec_key], t, delays, 
                         freq*1.e6, return_I=True)

    # if alpha is None:
    #     alpha = np.nanmean(catalogue["beta_p"])

    for i in range(len(catalogue)):

        row = catalogue[i]

        if model0:
            # Curved power law from full model components:
            if not np.asarray([np.isnan(catalogue[key][i]) for key in 
                [powerlaw_index, powerlaw_amplitude, powerlaw_curvature]]).all():

                at_flux[i] = cpowerlaw(freq, 
                                       a=catalogue[powerlaw_amplitude][i], 
                                       b=catalogue[powerlaw_index][i], 
                                       c=catalogue[powerlaw_curvature][i])

        if model1 and np.isnan(at_flux[i]):
            # Curved power law from reference measurement:
            if not np.asarray([np.isnan(catalogue[key][i]) for key in
                [powerlaw_index, powerlaw_curvature, flux0]]).all():

                a = cpowerlaw_amplitude(x0=freq0,
                                        y0=catalogue[flux0][i],
                                        b=catalogue[powerlaw_index][i],
                                        c=catalogue[powerlaw_curvature][i])
                at_flux[i] = cpowerlaw(freq, 
                                       a=a, 
                                       b=catalogue[powerlaw_index][i], 
                                       c=catalogue[powerlaw_curvature][i])

        if model2 and np.isnan(at_flux[i]):
            # Normal power law from full model components:
            if not np.asarray([np.isnan(catalogue[key][i]) for key in
                [powerlaw_amplitude, powerlaw_index]]).all():

                at_flux[i] = cpowerlaw(freq,
                                       a=catalogue[powerlaw_amplitude][i],
                                       b=catalogue[powerlaw_index][i],
                                       c=0.)

        if model3 and np.isnan(at_flux[i]):
            # Normal power law from reference measurement:
            if not np.asarray([np.isnan(catalogue[key][i]) for key in
                [powerlaw_index, flux0]]).all():

                a = cpowerlaw_amplitude(x0=freq0,
                                        y0=catalogue[flux0][i],
                                        b=catalogue[powerlaw_index][i],
                                        c=0)
                at_flux[i] = cpowerlaw(freq,
                                       a=a,
                                       b=catalogue[powerlaw_index][i],
                                       c=0.)

        if model4 and np.isnan(at_flux[i]):
            if not np.isnan(catalogue[flux0][i]):

                at_flux[i] = from_index(freq, 
                                        x1=freq0,
                                        y1=catalogue[flux0][i],
                                        index=alpha0)



    catalogue = catalogue[np.isfinite(at_flux)]
    stokesI = stokesI[np.isfinite(at_flux)]
    at_flux = at_flux[np.isfinite(at_flux)]

    logging.info("Sources after estimating flux density: {}".format(len(at_flux)))

    atten_flux = at_flux*stokesI

    catalogue = catalogue[np.isfinite(atten_flux)]
    at_flux = at_flux[np.isfinite(atten_flux)]
    atten_flux = atten_flux[np.isfinite(atten_flux)]
    
    catalogue = catalogue[atten_flux > threshold]
    at_flux = at_flux[atten_flux > threshold]
    atten_flux = atten_flux[atten_flux > threshold]
    
    logging.info("Sources after attenuating by the primary beam: {}".format(len(at_flux)))

    if len(catalogue) > nmax:
        logging.info("More than nmax={0} sources in catalogue - trimming.".format(nmax))
        idx = np.argsort(atten_flux)[::-1]
        catalogue = catalogue[idx[:nmax]]
        at_flux = at_flux[idx[:nmax]]
        atten_flux = atten_flux[idx[:nmax]]

        # a_cut = round(np.sort(atten_flux)[::-1][nmax-1], 1)
        
        # catalogue = catalogue[atten_flux > a_cut]
        # at_flux = at_flux[atten_flux > a_cut]
        # atten_flux = atten_flux[atten_flux > a_cut]
        a_cut = np.nanmin(atten_flux)

        logging.info("Sources after new flux treshold of {0} Jy: {1}".format(
                     a_cut, len(catalogue)))

    coords = SkyCoord(ra=catalogue[ra_key], dec=catalogue[dec_key], 
                      unit=(u.deg, u.deg))

    for i in range(len(catalogue)):

        if exclude_coords is not None:

            excl_seps = coords[i].separation(exclude_coords)
            if (excl_seps.value < bright_exclusion_zone/60.).any():
                logging.debug("{} within exclusion zone.".format(i))
                atten_flux[i] = np.nan
                continue

            else:
                logging.debug("{} not within exclude coords".format(i))

        if ignore_magellanic:
# 
            in_magellanic = False
            
            for magellanic in SOURCES:
                sep_magellanic = magellanic["coords"].separation(coords[i])
                if sep_magellanic.value <= magellanic["radius"]:
                    logging.debug("Skipping {} as it is within the {}".format(i, magellanic["name"]))
                    in_magellanic = True

            if in_magellanic:
                atten_flux[i] = np.nan
                continue

    catalogue = catalogue[np.isfinite(atten_flux)]
    at_flux = at_flux[np.isfinite(atten_flux)]
    atten_flux = atten_flux[np.isfinite(atten_flux)]
    
    
    logging.info("Sources after excluding sources: {}".format(len(catalogue)))
    

    coords = SkyCoord(ra=catalogue[ra_key], dec=catalogue[dec_key], 
                      unit=(u.deg, u.deg))


    freqs_to_predict = [round(v, 3) for v in np.linspace(freq-0.5*30.72, freq+0.5*30.72, nfreqs_to_predict)]

    with open(outname, "w+") as o:
        o.write("skymodel fileformat 1.1\n")

        for i in range(len(catalogue)):

            r, d = coords[i].to_string("hmsdms").split()

            flux = []
            for f in freqs_to_predict:

                if model0:
                    # Curved power law from full model components:
                    if not np.asarray([np.isnan(catalogue[key][i]) for key in 
                        [powerlaw_index, powerlaw_amplitude, powerlaw_curvature]]).all():

                        flux.append(cpowerlaw(f, 
                                              a=catalogue[powerlaw_amplitude][i], 
                                              b=catalogue[powerlaw_index][i], 
                                              c=catalogue[powerlaw_curvature][i]))

                elif model1:
                    # Curved power law from reference measurement:
                    if not np.asarray([np.isnan(catalogue[key][i]) for key in
                        [powerlaw_index, powerlaw_curvature, flux0]]).all():

                        a = cpowerlaw_amplitude(x0=freq0,
                                                y0=catalogue[flux0][i],
                                                b=catalogue[powerlaw_index][i],
                                                c=catalogue[powerlaw_curvature][i])
                        flux.append(cpowerlaw(f, 
                                              a=a, 
                                              b=catalogue[powerlaw_index][i], 
                                              c=catalogue[powerlaw_curvature][i]))

                elif model2:
                    # Normal power law from full model components:
                    if not np.asarray([np.isnan(catalogue[key][i]) for key in
                        [powerlaw_amplitude, powerlaw_index]]).all():

                        flux.append(cpowerlaw(f,
                                              a=catalogue[powerlaw_amplitude][i],
                                              b=catalogue[powerlaw_index][i],
                                              c=0.))

                elif model3:
                    # Normal power law from reference measurement:
                    if not np.asarray([np.isnan(catalogue[key][i]) for key in
                        [powerlaw_index, flux0]]).all():

                        a = cpowerlaw_amplitude(x0=freq0,
                                                y0=catalogue[flux0][i],
                                                b=catalogue[powerlaw_index][i],
                                                c=0)
                        flux.append(cpowerlaw(f,
                                              a=a,
                                              b=catalogue[powerlaw_index][i],
                                              c=0.))

                elif model4:
                    if not np.isnan(catalogue[flux0][i]):

                        flux.append(from_index(f, 
                                               x1=freq0,
                                               y1=catalogue[flux0][i],
                                               index=alpha0))


            name = "J{}{}{}{}".format(r[:2], r[3:5], d[:3], d[4:6])

            entry_format = point_formatter(name=name,
                                           ra=r,
                                           dec=d,
                                           freq=freqs_to_predict,
                                           flux=flux)

            o.write(entry_format)

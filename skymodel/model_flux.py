#!/usr/bin/env python

from __future__ import print_function, division

import os
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u

from skymodel.parsers import parse_ao, parse_metafits
from skymodel.get_beam import atten_source, beam_value, make_beam_image, find_lobes
from skymodel.peel_suggester import slice_ao

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def total_flux(aocal, freq=None, alpha=-0.7, metafits=None, attenuate=False,
               curved=True, radius=360., coords=None, nlobes=1, beam_image=None):
    """Get total flux from aocal file. 

    Assume single source, and calculate the flux at a given frequency for each
    component before summing component fluxes.

    If a metafits file is supplied the frequency information there is used
    in place of the `freq` parameter.

    Parameters
    ----------
    aocal : str
        Skymodel format 1.0/1.1 file with source parameters.
    freq : float
        Frequency in MHz at which to estimate flux density.
    alpha : float, optional
        Spectral index used if only one flux density measurement, assuming
        a simple power law model.
    metafits : str, optional
        A metafits file for an observation to attenuate the brightness 
        by the primary beam response of the given observation. The beam value
        is taken at the position of the zeroth component.

    Returns
    -------
    float
        Total flux density of all components/sources in `aocal` at `freq`.

    """

    sources = parse_ao(aocal)

    if metafits is not None:
        t, delays, at_freq, pnt = parse_metafits(metafits)
        if coords is None:
            coords = pnt
    elif freq is not None:
        at_freq = freq
    else:
        raise ValueError("Frequency information not found!")

    tflux = 0
    fluxes, ras, decs = [], [], []

    if nlobes > 1 and beam_image is not None:
        lobe_table = find_lobes(beam_image, perc=0.1, return_table=True)
        centres = SkyCoord(
            ra=np.asarray(lobe_table[:nlobes]["ra"]),
            dec=np.asarray(lobe_table[:nlobes]["dec"]),
            unit=(u.deg, u.deg)
        )
    else:
        centres = [coords]

    for source in sources:
        # First calculate the flux density at the given frequency:

        # If coords are provided, check that the source is inside the given 
        # radius - if not, set its flux to zero so it isn't included in the 
        # model. 
        isin = False
        for coords in centres:
            if coords is not None:
                seps = np.asarray([source.components[i].radec.separation(coords).value 
                                for i in range(source.ncomponents)])
                if (seps < radius).any():
                    isin = True
                    continue
            else:
                isin = True
                continue
        
        if not isin:
            logger.debug("skipping as outside of radius")
            continue

        source.at_freq(freq=at_freq,
                       components=range(source.ncomponents),
                       alpha=alpha,
                       curved=curved)

        sflux = np.nansum(np.array([source.components[i].at_flux 
                                    for i in range(source.ncomponents)]))


        fluxes.append(sflux)
        ras.append(np.mean([source.components[i].radec.ra.value \
                for i in range(source.ncomponents)]))
        decs.append(np.mean([source.components[i].radec.dec.value \
                for i in range(source.ncomponents)]))
        

    if (metafits is not None) and attenuate:
        pseudoI = beam_value(ra=np.asarray(ras),
                             dec=np.asarray(decs),
                             t=t,
                             delays=delays,
                             freq=at_freq,
                             return_I=True)
        sflux_atten = pseudoI * np.asarray(fluxes)
        tflux = np.nansum(sflux_atten)

    else:
        tflux = np.sum(sflux)

    return tflux


def prep_model(inp, metafits, threshold, outname="./all_models.txt",
               prefix="model", exclude=None, curved=True, radius=360.,
               nlobes=1, export_prefix=None, pnt=None):
    """Prepare a combined AO-style model, using models in a directory.

    Parameters
    ----------
    input : str
        Directory containing AO-style model files. Must begin with 'model' and 
        end with '.txt'. 
        Alternatively, a list of filenames.
    metafits : str
        Filepath to a metafits file for a particular observation. This is used
        to attenuate the brightness by the primary beam response.
    threshold : float,
        The threshold above which to add a model to the combined model file.

    """

        

    all_models = open(outname, "w+")
    all_models.write("skymodel fileformat 1.1\n")

    if isinstance(inp, (list, np.ndarray)):
        files = inp
    elif os.path.isdir(inp):
        files = [inp+"/"+f for f in os.listdir(inp) 
                 if (f.endswith(".txt") and f.startswith(prefix))]
    elif not isinstance(inp, (list, np.ndarray)):
        files = [inp]
    else:
        raise ValueError("Unable to parse `inp`: {}".format(inp))

    # files_to_use = ""
    total_fluxes = ""

    if nlobes > 1:
        t, delays, freq, pnt_ = parse_metafits(metafits)
        if pnt is None:
            pnt = pnt_
        else:
            pnt = SkyCoord(ra=pnt[0]*u.deg, dec=pnt[1]*u.deg)
        beam_image = make_beam_image(t, delays, freq,
            ra=pnt.ra.value,
            return_hdu=True)
    else:
        beam_image = None
        if pnt is not None:
            pnt = SkyCoord(ra=pnt[0]*u.deg, dec=pnt[1]*u.deg)

    for spec in files:

        if exclude is not None:
            # if np.asarray([x in spec for x in exclude]).any():
                # continue
            if spec in exclude:
                continue
            if np.array([os.path.basename(spec).replace(prefix, "") in os.path.basename(ex) for ex in exclude]).any():
                continue

        tflux = total_flux(spec, 
                           attenuate=True, 
                           metafits=metafits,
                           curved=curved, 
                           radius=radius,
                           nlobes=nlobes,
                           beam_image=beam_image,
                           coords=pnt)
        
        if tflux > threshold:

            if export_prefix is not None:
                single_model_file = open(os.path.basename(spec).replace(prefix, export_prefix), "w+")
        
            with open(spec, "r+") as f:
                lines = f.readlines()
                for line in lines:
                    if "skymodel" in line:
                        continue
                    elif "source" in line:
                        # all_models.write("\n"+line.lstrip())
                        line_to_write = "\n"+line.lstrip()

                    elif "fluxdensity" in line:
                        # all_models.write("fluxdensity Jy {} 0 0 0\n".format(
                            # line.split()[2]))
                        line_to_write = "fluxdensity Jy {} 0 0 0\n".format(
                            line.split()[2]
                        )
                    else:
                        # all_models.write(line.lstrip())
                        line_to_write = line.lstrip()
                    
                    all_models.write(line_to_write)
                    if export_prefix is not None:
                        single_model_file.write(line_to_write)


            # files_to_use += "{}\n".format(spec)

        total_fluxes += "{}: {}\n".format(spec, tflux)

        # print("{}: {}".format(spec, tflux))

        

    print(total_fluxes)
    all_models.flush()
    all_models.close()



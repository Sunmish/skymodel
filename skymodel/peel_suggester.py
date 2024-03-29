#! /usr/bin/env python

from __future__ import print_function

import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)

from skymodel.get_beam import atten_source
from skymodel.parsers import parse_ao, parse_metafits


def slice_ao(source, aofile, method="peel"):
    """Slice source out of aofile.

    Parameters
    ----------
    source : peel_suggester.Source object
        An initialised peel_suggester.Source object.
    aofile : str
        A skymodel format 1.0/1.1 file.

    Returns
    -------
    str
        Name of new source model file.

    """

    outname = method+"-"+source.name.split()[0]+".txt"  # consistent with prep_model

    f = open(aofile, "r")
    with open(outname, "w+") as g:
        g.write("skymodel fileformat 1.1\n")
        lines = f.readlines()

        found_source = False
        found_other_source = False
        for i, line in enumerate(lines):

            if i+1 < len(lines):
                if len(line.split()) == 0:  # Tidy up white space lines.
                    continue
                elif "source" in line and source.name in lines[i+1]:
                    if found_other_source:
                        h.flush()
                        h.close()
                        found_other_source = False
                    found_source = True
                elif "source" in line:
                    found_source = False
                    
                    found_other_source = True

                    other_outname = "model-"+lines[i+1].split()[1].replace("\"", "")+".txt"
                    h = open(other_outname, "w+")
                    h.write("skymodel fileformat 1.1\n")

                if found_source:
                    g.write(line)
                elif found_other_source:
                    h.write(line)

    f.close()

    return outname


def autoprocess(aofile, metafits, peel_threshold=25., peel_radius=0., 
                subtract_threshold=10., subtract_radius=0.,alpha=-0.7, 
                verbose=False, duplicates=True, pnt=None):
    """Attenuate models in an `aofile`.

    Additionally, write out individual models, if attenuated brightness is
    above `threshold` for use in peeling later.

    Parameters
    ----------
    aofile : str
        A skymodel format 1.0/1.1 file.
    metafits : str
        MWA metafits file.
    threshold : float, optional
        Threshold in Jy for suggesting peeling and slicing out model. [Default 25]
     alpha : float, optional
        Assumed spectral index for when there is only one flux density 
        measurement. [Default -0.7]

    Returns
    -------
    np.ndarray
        Array of the form: Source.name, model name, apparent brightness,
        if and only if any sources are above `threshold`. Note that the array
        has a string dtype.
    None
        If no sources with apparent brightness above `threshold` are found.

    """

    t, delays, freq, pnt_ = parse_metafits(metafits)
    if pnt is None:
        pnt = pnt_
    else:
        pnt = SkyCoord(ra=pnt[0]*u.deg, dec=pnt[1]*u.deg)

    writeout = ""

    i = 0
    names, models, abrights, ra, dec, method = [], [], [], [], [], []

    for ao in aofile:
        sources = parse_ao(ao)
        for source in sources:
            if not source.name in names or duplicates:
                logging.debug("Working on source {}".format(source.name))
                apparent_brightness = atten_source(source=source,
                                                   t=t,
                                                   delays=delays,
                                                   freq=freq,
                                                   alpha=-0.7)

                sep = pnt.separation(source.components[0].radec).value

                writeout += "{:<22}: {:.2f} Jy\n".format(source.name, 
                                                         apparent_brightness)
                
                if (apparent_brightness > peel_threshold) and (sep > peel_radius):
                    # Slice out model to use in peeling later:
                    model_name = slice_ao(source, ao, method="peel")
                    names.append(source.name)
                    models.append(model_name)
                    abrights.append(apparent_brightness)
                    ra.append(source.components[0].radec.ra.value)
                    dec.append(source.components[0].radec.dec.value)
                    method.append("peel")
                elif (apparent_brightness > subtract_threshold) and \
                    (sep > subtract_radius):
                    # Slice out model to use in subtracting later:
                    model_name = slice_ao(source, ao, method="subtract")
                    names.append(source.name)
                    models.append(model_name)
                    abrights.append(apparent_brightness)
                    ra.append(source.components[0].radec.ra.value)
                    dec.append(source.components[0].radec.dec.value)
                    method.append("subtract")

            else:
                logging.warn("{} ingnored as it has already been added".format(source.name))


    logging.info("Sources and their apparent brightnesses:")

    if verbose:
        print(writeout)

    try:
        peel = np.array([np.asarray(abrights),
                         np.asarray(names),
                         np.asarray(models),
                         np.asarray(ra),
                         np.asarray(dec),
                         np.asarray(method)]).T
        peel = peel[peel[:, 0].astype("f").argsort()[::-1]]  # brightest first
    except Exception:
        # raise
        return None
    else:
        return peel


    

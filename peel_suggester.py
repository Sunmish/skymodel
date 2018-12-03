#! /usr/bin/env python

from __future__ import print_function

import numpy as np

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)

from .get_beam import atten_source
from .parsers import parse_ao, parse_metafits


def slice_ao(source, aofile):
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

    outname = "peel-"+source.name.split()[0]+".txt"  # consistent with prep_model

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


def autoprocess(aofile, metafits, threshold=25., radius=0., alpha=-0.7, verbose=False,
                duplicates=True):
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

    t, delays, freq, pnt = parse_metafits(metafits)

    writeout = ""

    i = 0
    names, models, abrights, ra, dec = [], [], [], [], []

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
                print(sep)

                writeout += "{:<22}: {:.2f} Jy\n".format(source.name, 
                                                         apparent_brightness)
                
                if (apparent_brightness > threshold) and (sep > radius):
                    # Slice out model to use in peeling later:
                    model_name = slice_ao(source, ao)
                    names.append(source.name)
                    models.append(model_name)
                    abrights.append(apparent_brightness)
                    ra.append(source.components[0].radec.ra.value)
                    dec.append(source.components[0].radec.dec.value)
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
                         np.asarray(dec)]).T
        peel = peel[peel[:, 0].astype("f").argsort()[::-1]]  # brightest first
    except Exception:
        return None
    else:
        return peel


    

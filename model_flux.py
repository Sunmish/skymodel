#!/usr/bin/env python

import os
import numpy as np

from .parsers import parse_ao, parse_metafits
from .get_beam import atten_source


def total_flux(aocal, freq=None, alpha=-0.7, metafits=None, attenuate=False):
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
    elif freq is not None:
        at_freq = freq
    else:
        raise ValueError("Frequency information not found!")

    tflux = 0
    for source in sources:
        # First calculate the flux density at the given frequency:
        source.at_freq(freq=at_freq,
                       components=range(source.ncomponents),
                       alpha=alpha)

        
        if (metafits is not None) and attenuate:
            sflux = atten_source(source=source,
                                 t=t,
                                 delays=delays,
                                 freq=at_freq)
        else:
            sflux = np.nansum(np.array([source.components[i].at_flux*pseudoI[i] 
                                        for i in range(source.ncomponents)]))

        tflux += sflux

    return tflux


def prep_model(indir, metafits, threshold, outname="./all_models.txt"):
    """Prepare a combined AO-style model, using models in a directory.

    Parameters
    ----------
    indir : str
        Directory containing AO-style model files. Must begin with 'model' and 
        end with '.txt'. 
    metafits : str
        Filepath to a metafits file for a particular observation. This is used
        to attenuate the brightness by the primary beam response.
    threshold : float,
        The threshold above which to add a model to the combined model file.

    """

    all_models = open(outname, "w+")
    all_models.write("skymodel fileformat 1.1\n")

    files = os.listdir(indir)

    files_to_use = ""
    total_fluxes = ""

    for spec in files:
        if spec.endswith(".txt") and spec.startswith("model-"):

            tflux = total_flux(indir+"/"+spec, 
                               attenuate=True, 
                               metafits=metafits)
            
            if tflux > threshold:

                with open(indir+"/"+spec, "r+") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "skymodel" in line:
                            pass
                        elif "source" in line:
                            all_models.write("\n"+line.lstrip())
                        elif "fluxdensity" in line:
                            all_models.write("fluxdensity Jy {} 0 0 0\n".format(
                                line.split()[2]))
                        else:
                            all_models.write(line.lstrip())

                files_to_use += "{}\n".format(spec)

            total_fluxes += "{}\n".format(tflux)

            print("{}: {}".format(spec, tflux))

    print(files_to_use)
    print(total_fluxes)

    all_models.flush()
    all_models.close()



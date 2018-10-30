#! /usr/bin/env python

from __future__ import print_function

import numpy as np
import os
import argparse
from subprocess32 import Popen

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)

from scipy.optimize import curve_fit

# Astopy imports:
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u

from . import beam_value




class Component(object):
    """Component object."""


    def __init__(self, ra, dec, flux, freq):

        if isinstance(ra, str):
            self.radec = SkyCoord(ra, dec)
        elif isinstance(ra, float):
            self.radec = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    
        if not hasattr(flux, "__getitem__"):
            self.flux = np.array([flux])
            self.freq = np.array([freq])
        else:
            self.flux = np.asarray(flux)
            self.freq = np.asarray(freq)

        if len(self.flux) != len(self.freq): 
            raise ValueError("len(flux) = {} != len(freq) = {}!".format(len(self.flux),
                                                                        len(self.freq)))


    def add_freq(self, flux, freq):
        """Add additional frequency data.

        Only one of these are stored at a time - mostly for temporary uses.
        """

        self.at_freq = freq
        self.at_flux = flux



class Source(object):
    """Source object."""


    def __init__(self, name):
        """Initialise with one component."""

        self.name = name
        self.components = np.array([])
        self.ncomponents = 0


    def add_component(self, ra, dec, flux, freq):
        """Add additional components if required."""

        self.components = np.append(self.components,
                                    Component(ra, dec, flux, freq))
        self.ncomponents = len(self.components)


    def at_freq(self, freq, components=0, curved=True, alpha=-0.7):
        """Calculate the flux density of the source at a given frequency.

        This is done via various methods depending on number of flux density
        measurements for a given component:

        > 2 measurements: either powerlaw or curved powerlaw model is fit, 
            and a flux extrapolated/interpolated from that fit model.
        2 measurements: a two-point power law index is calculated and 
            a flux density is extrapolated/interpolated from the measured data.
        1 measurements: a spectral index is assumed, and a flux density is
            extrapolated/interpolated from the measured flux density.

        Parameters
        ----------
        freq : float
            Frequency in Hz at which to extrapolated/interpolate a flux density.
        components : int or list, optional
            Component indices to calculate. [Default 0]
        curved : bool, optional
            Select True if wanted curved power law instead of regular power law.
            [Default True] 
        alpha : float, optional
            Assumed spectral index for when there is only one flux density 
            measurement. [Default -0.7]


        This adds an extra attribute to the Component object, and overrides
        any existing Component.at_freq and Component.at_flux attributes.

        """

        if not hasattr(components, "__getitem__"):
            components = [components]

        for c in components:
            comp = self.components[c]

            if curved and len(comp.freq) > 2:
                model = cpowerlaw
                params = [-0.8, 0, 1]
            elif len(comp.freq) > 2:
                model = powerlaw
                params = [-1, 1]
            elif len(comp.freq) == 2:
                index = two_point_index(x1=comp.freq[0], 
                                        x2=comp.freq[1],
                                        y1=comp.flux[0],
                                        y2=comp.flux[1])
                flux_at_freq = from_index(x=freq,
                                          x1=comp.freq[0], 
                                          y1=comp.flux[0],
                                          index=alpha)
                self.components[c].add_freq(flux=flux_at_freq, freq=freq)
                continue
            else:
                flux_at_freq = from_index(x=freq, 
                                          x1=comp.freq[0], 
                                          y1=comp.flux[0],
                                          index=alpha)
                self.components[c].add_freq(flux=flux_at_freq, freq=freq)
                continue


            popt, pcov = curve_fit(model, comp.freq, comp.flux, params, method="lm")
            perr = np.sqrt(np.diag(pcov))
            flux_at_freq = model(freq, *popt)

            self.components[c].add_freq(flux=flux_at_freq, freq=freq)




# Models for fitting --------------------------------------------------------- #

def two_point_index(x1, x2, y1, y2):
    """Calculate spectral index from two measurements."""
    return np.log10(y1/y2)/np.log10(x1/x2)

def from_index(x, x1, y1, index):
    """Calculate flux from measured value and measured/assumed index."""
    return y1*(x/x1)**index

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

# ---------------------------------------------------------------------------- #


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



def atten_source(source, t, delays, freq, alpha=-0.7):
    """Attenuate a source by the primary beam response.

    Attenuate each component at the given frequency, then sum the components
    to determine total apparent brightness.
    """


    ra = np.array([source.components[i].radec.ra.value for i in range(source.ncomponents)])
    dec = np.array([source.components[i].radec.dec.value for i in range(source.ncomponents)])

    XX, YY = beam_value(ra=ra,
                        dec=dec,
                        t=t,
                        delays=delays,
                        freq=freq)

    source.at_freq(freq=freq,
                   components=range(source.ncomponents),
                   alpha=alpha)

    pseudoI = 0.5*(XX + YY)
    atten_flux = np.array([source.components[i].at_flux*pseudoI[i] for i in
                           range(source.ncomponents)])

    total_atten_flux = np.nansum(atten_flux)

    return total_atten_flux



def parse_metafits(metafits):
    """Read in metafits file and return relevant information."""

    m = fits.getheader(metafits)

    delays = [int(d) for d in m["DELAYS"].split(",")]
    t = Time(m["DATE-OBS"], format="isot", scale="utc")
    freq = m["FREQCENT"] * 1.e6  # in Hz

    return t, delays, freq



def parse_ao(aofile):
    """Parse Andre Offringa's skymodel format 1.0/1.1 file."""

    # print(aofile)

    sources = []

    with open(aofile, "r") as f:
        lines = f.readlines()

        found_source = False
        found_component = False
        for i, line in enumerate(lines):
    
            if found_component and found_source:
                if "frequency" in line:
                    freq.append(float(line.split()[1])*1.e6)
                    flux.append(float(lines[i+1].split()[2]))
                    if "}" in lines[i+2] and "}" in lines[i+3]:
                        source.add_component(ra=ra,
                                             dec=dec,
                                             flux=flux,
                                             freq=freq)
                        logging.debug("Found flux, freq: {}, {}".format(flux,
                                                                        freq))
                        found_component = False

            elif "name" in line:
                if found_source:
                    sources.append(source)
                name = " ".join(line.split()[1:]).strip("\n").strip("\"")
                logging.debug("Found {}".format(name))
                source = Source(name)
                found_source = True

            elif "component" in line and found_source:
                if "point" in lines[i+1]:  # Only deal with point sources.
                    found_component = True
                    ra = lines[i+2].split()[1]
                    dec = lines[i+2].split()[2].strip("\n")
                    logging.debug("Found component at {}, {}".format(ra, dec))
                    flux, freq = [], []
                else:
                    logging.warning("Only point sources are currently supported. ({})".format(name))
                    break

        
        if found_source:
            sources.append(source)
                

    return np.asarray(sources)



def slice_ao(source, aofile):
    """Slice source out of aofile.

    Produces two output files: a skymodel format 1.1 file of the source model,
    and a version of `aofile` without the source in it.

    Parameters
    ----------
    source : peel_suggester.Source object
        An initialised peel_suggester.Source object.
    aofile : str
        A skymodel format 1.0/1.1 file.

    Returns
    -------
    str
        Name of source model file.

    """

    # print(source.name)
    outname = source.name.split()[0]+".model"
    # sliced = aofile.split(".")[0]+"_no_"+outname.replace(".model", ".txt")

    f = open(aofile, "r")
    # h = open(sliced, "w+")
    with open(outname, "w+") as g:
        g.write("skymodel fileformat 1.1\n")
        lines = f.readlines()

        found_source = False
        for i, line in enumerate(lines):
            
            if i+1 < len(lines):
                if len(line.split()) == 0:  # Tidy up white space lines.
                    continue
                elif "source" in line and source.name in lines[i+1]:
                    found_source = True
                elif "source" in line:
                    found_source = False

                if found_source:
                    g.write(line)
                # else:
                #     h.write(line)

            # elif i < len(lines):
            #     h.write(line)
    f.close()
    # h.close()

    return outname



def autoprocess(aofile, metafits, threshold=25., alpha=-0.7, verbose=False,
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

    t, delays, freq = parse_metafits(metafits)

    writeout = ""

    i = 0
    names, models, abrights = [], [], []

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

                writeout += "{:<22}: {:.2f} Jy\n".format(source.name, apparent_brightness)
                
                if apparent_brightness > threshold:
                    # Slice out model to use in peeling later:
                    model_name = slice_ao(source, ao)
                    names.append(source.name)
                    models.append(model_name)
                    abrights.append(apparent_brightness)
            else:
                logging.warn("{} ingnored as it has already been added".format(source.name))

    logging.info("Sources and their apparent brightnesses:")
    if verbose:
        print(writeout)

    try:
        peel = np.array([np.asarray(abrights),
                         np.asarray(names),
                         np.asarray(models)]).T
        peel = peel[peel[:, 0].astype("f").argsort()[::-1]]  # brightest first
    except Exception:
        return None
    else:
        return peel



def main():

    ps = argparse.ArgumentParser(description="Find sources to peel using a "
                                             "file with models.")

    help_ = {"f": "A skymodel format 1.0/1.1 file with a list of models.",
             "m": "MWA metafits file.",
             "t": "Threshold in Jy above which to suggest peeling. [Default 25]",
             "alpha": "Spectral index to assume if components/sources do not "
                      "have more than 1 flux density measurement. [Default -0.7]",
            "p": "Switch to enable use of Andre Offringa's 'peel' tool. All "
                 "sources above the set threshold will then be peeled in order "
                 "of apparent brightness. [Default False]",
            "i": "Interval for peeling. Passed to 'peel'. [Default 120]",
            "u": "Minimum uv (in m) for peeling. Passed to 'peel'. "
                 "[Default 60]",
            "ms": "MeasurementSet for peeling. Passed to 'peel'. Required."
            }

    ps.add_argument("-f", "--aofile", "--models", "--skymodel", type=str,
                    help=help_["f"], dest="aofile", default=None, nargs="*")
    ps.add_argument("-m", "--metafits", type=str, help=help_["m"], default=None)
    ps.add_argument("-t", "--threshold", type=float, help=help_["t"], 
                    default=25.)
    ps.add_argument("-a", "--alpha", "--spectral_index", dest="alpha",
                    help=help_["alpha"], default=-0.7, type=float)
    ps.add_argument("-v", "--verbose", action="store_true")
    ps.add_argument("-D", "--duplicates", action="store_true")


    args = ps.parse_args()
    if args.aofile is None:
        raise ValueError("An aofile/skymodel must be supplied.")
    elif args.metafits is None:
        raise ValueError("An MWA metafits file must be supplied.")

    peel_sources = autoprocess(aofile=args.aofile, 
                               metafits=args.metafits, 
                               threshold=args.threshold, 
                               alpha=args.alpha,
                               verbose=args.verbose,
                               duplicates=args.duplicates)

    if peel_sources is not None:
        for source in peel_sources:
            if not args.verbose:
                print("{} {} {}".format(source[1], source[2], source[0]))


if __name__ == "__main__":
    main()
    

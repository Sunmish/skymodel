#! /usr/bin/env python

import numpy as np

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)

from . import fitting
from .get_beam import beam_value


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
                model = fitting.cpowerlaw
                params = [-0.8, 0, 1]
            elif len(comp.freq) > 2:
                model = fitting.powerlaw
                params = [-1, 1]
            elif len(comp.freq) == 2:
                index = fitting.two_point_index(x1=comp.freq[0], 
                                                x2=comp.freq[1],
                                                y1=comp.flux[0],
                                                y2=comp.flux[1])
                flux_at_freq = fitting.from_index(x=freq,
                                                  x1=comp.freq[0], 
                                                  y1=comp.flux[0],
                                                  index=index)
                self.components[c].add_freq(flux=flux_at_freq, freq=freq)
                continue
            else:
                flux_at_freq = fitting.from_index(x=freq, 
                                                  x1=comp.freq[0], 
                                                  y1=comp.flux[0],
                                                  index=alpha)
                self.components[c].add_freq(flux=flux_at_freq, freq=freq)
                continue

            popt, perr = fitting.fit(f=model,
                                     x=comp.freq,
                                     y=comp.flux,
                                     params=params)
            flux_at_freq = model(freq, *popt)

            self.components[c].add_freq(flux=flux_at_freq, freq=freq)


def parse_ao(aofile):
    """Parse Andre Offringa's skymodel format 1.0/1.1 file."""

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
                # if "point" in lines[i+1]:  # Only deal with point sources.
                found_component = True
                ra = lines[i+2].split()[1]
                dec = lines[i+2].split()[2].strip("\n")
                logging.debug("Found component at {}, {}".format(ra, dec))
                flux, freq = [], []
                # else:
                #     logging.warning("Only point sources are currently supported. ({})".format(name))
                #     break

        
        if found_source:
            sources.append(source)
                

    return np.asarray(sources)



def parse_metafits(metafits):
    """Read in metafits file and return relevant information."""

    m = fits.getheader(metafits)

    delays = [int(d) for d in m["DELAYS"].split(",")]
    t = Time(m["DATE-OBS"], format="isot", scale="utc")
    freq = m["FREQCENT"] * 1.e6  # in Hz
    pnt = SkyCoord(ra=m["RA"], dec=m["DEC"], unit=(u.deg, u.deg))


    return t, delays, freq, pnt

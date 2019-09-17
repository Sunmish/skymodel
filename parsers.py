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


    def __init__(self, ra, dec, flux, freq, a=None, b=None, pa=None,
                 alpha=-0.77):

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

        if a is not None and b is not None and pa is not None:
            self.a = float(a)
            self.b = float(b)
            self.pa = float(pa)
        else:
            self.a = None
            self.b = None
            self.pa = None

        self.alpha = alpha


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


    def add_component(self, ra, dec, flux, freq, a=None, b=None, pa=None,
                      alpha=None):
        """Add additional components if required."""

        self.components = np.append(self.components,
                                    Component(ra, dec, flux, freq, a, b, pa, alpha))
        self.ncomponents = len(self.components)


    def at_freq(self, freq, components=0, curved=True, alpha=-0.7, 
                nearest_freq=2.):
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
        nearest_freq : float, optional
            If a flux density measurement exists within `nearest_freq` then 
            it is used. 


        This adds an extra attribute to the Component object, and overrides
        any existing Component.at_freq and Component.at_flux attributes.

        """

        if not hasattr(components, "__getitem__"):
            components = [components]

        for c in components:
            comp = self.components[c]

            
            
            idx = (np.abs(comp.freq - freq)).argmin()
            if abs(comp.freq[idx] - freq) < nearest_freq*1.e6:
                self.components[c].add_freq(flux=comp.flux[idx],
                                            freq=freq)
                logging.debug("using nearest freq to {} ({})".format(freq, comp.freq[idx]))

                continue


            if curved and len(comp.freq) > 3:
                logging.debug("cpowerlaw with {} parameters".format(len(comp.freq)))
                model = fitting.cpowerlaw
                amp0 = fitting.powerlaw_amplitude(comp.freq[0], comp.flux[0], -1.)
                params = [amp0, -1., 0.]
            elif len(comp.freq) > 2:
                logging.debug("powerlaw with {} parameters".format(len(comp.freq)))
                model = fitting.powerlaw
                amp0 = fitting.powerlaw_amplitude(comp.freq[0], comp.flux[0], -1.)
                params = [amp0, -1.]
            elif len(comp.freq) == 2:
                logging.debug("two-point with {} parameters".format(len(comp.freq)))
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
                logging.debug("from-index ({}) with {} parameters".format(comp.alpha, len(comp.freq)))
                flux_at_freq = fitting.from_index(x=freq, 
                                                  x1=comp.freq[0], 
                                                  y1=comp.flux[0],
                                                  index=comp.alpha)
                self.components[c].add_freq(flux=flux_at_freq, freq=freq)
                continue

            popt, perr = fitting.fit(f=model,
                                     x=comp.freq,
                                     y=comp.flux,
                                     params=params)
            logging.debug("popt: {}".format(popt))
            flux_at_freq = model(float(freq), *popt)

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

                if "frequency" in line.lower():
                
                    freq.append(float(line.split()[1])*1.e6)
                    flux.append(float(lines[i+1].split()[2]))

                    if "spectral-index" in lines[i+2].lower():
                        alpha = float(lines[i+2].split()[2])
                
                    if "}" in lines[i+2] and "}" in lines[i+3]:

                        source.add_component(ra, dec, flux, freq, a, b, pa, alpha)
                        found_component = False

            elif "name" in line.lower():
                
                if found_source:
                    sources.append(source)
                
                name = " ".join(line.split()[1:]).strip("\n").strip("\"")
                logging.debug("Found {}".format(name))
                source = Source(name)
                found_source = True

            elif "component" in line.lower() and found_source:

                found_component = True
                for j in [2, 3]:
                    if "position" in lines[i+j].lower():  
                        ra = lines[i+j].split()[1]
                        dec = lines[i+j].split()[2].strip("\n")
                logging.debug("Found component at {}, {}".format(ra, dec))
                flux, freq = [], []
                alpha = -0.77  # average over N+S sources

                a = b = pa = None

                if "gaussian" in lines[i+1].lower():
                    # Record the major and minor axes as well as the pa:
                    for j in [2, 3]:
                        if "shape" in lines[i+j]:
                            _, a, b, pa = lines[i+j].split()
                            
                    


        
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


def parse_obslist(obslist, selection="all"):
    """Read in OBS ID list.

    This may be simply a list of OBS IDs, and optionally a list of 
    tiles to flag per snapshot.
    """

    obs_ids, obs_flags = [], []

    with open(obslist, "r") as obs:

        lines = obs.readlines()
        for l in range(len(lines)):
            if (selection == "all" or l+1 in selection) and "#" not in lines[l]:

                line = lines[l]
                bits = line.replace("\n", "").split(" ")
                if len(bits) == 1:  # No flag tiles on this line.
                    flags = ""
                elif len(bits) == 2:
                    flags = bits[1]
                else:
                    logging.warning("OBS ID file line has too many items? \n" 
                                    "{0}".format(line))
                    flags = bits[1]

                obs_ids.append(bits[0])
                obs_flags.append(flags)

    return obs_ids, obs_flags

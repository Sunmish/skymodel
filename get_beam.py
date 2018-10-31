#!/usr/bin/env python

from __future__ import print_function

import numpy as np

# Astopy imports:
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u

import logging

# MWA beam-related imports:
from mwapy import ephem_utils
from mwapy.pb.primary_beam import MWA_Tile_full_EE

# # Too much unnecessary output...?
# logging.getLogger("beam_full_EE").setLevel(logging.WARNING)

MWA = EarthLocation.from_geodetic(lat=-26.703319*u.deg, 
                                  lon=116.67081*u.deg, 
                                  height=377*u.m)


def beam_value(ra, dec, t, delays, freq, interp=True, return_I=False):
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
                              pixels_per_deg=10)

    if return_I:
        return 0.5*(rX[0] + rY[0])
    else:
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
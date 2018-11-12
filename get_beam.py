#!/usr/bin/env python

from __future__ import print_function

import numpy as np

# Astopy imports:
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
from astropy.wcs import WCS

from scipy import ndimage  # For lobe finding in the primary beam images
from scipy.spatial import distance

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



def make_beam_image(t, delays, freq, outname=None, cmap="cubehelix", stretch="sqrt", 
                    npix=1500, ra=75., dec=-26., plot=False, return_hdu=False):
    """Make a FITS image of the psuedo-I beam response.

    Parameters
    ----------
    t : int or float
        GPS time in iso format. From `parse_metafits` in `skymodel.parsers`.
    delays : list
        List of delays. From `parse_metafits` in `skymodel.parsers`.
    freq : float
        Frequency in Hz. From `parse_metafits` in `skymodel.parsers`.
    outname : str
        Output file name for the beam FITS image.
    cmap : str, optional
        Colormap to use if plotting. [Default cubehelix]
    npix : int, optional
        Size of the beam image. This changes the pixel dimensions. [Default 1500]
    ra : float, optional
        RA for center of the image. [Default 75 (05:00:00)]
    dec : float, optional
        Dec. for center of the image. [Default -26]
    plot : bool, optional
        Switch true if wanting to make a simple plot. [Default False]


    """

    # Initialise a FITS image:
    hdu = fits.PrimaryHDU()
    arr = np.full((npix, npix), 0.)
    hdu.data = arr

    hdu.header["CTYPE1"] = "RA---SIN"
    hdu.header["CTYPE2"] = "DEC--SIN"
    hdu.header["CRVAL1"] = ra
    hdu.header["CRVAL2"] = dec
    hdu.header["CDELT1"] = -npix*(0.08/1500.)
    hdu.header["CDELT2"] = npix*(0.08/1500.)
    hdu.header["CRPIX1"] = npix//2 - 1
    hdu.header["CRPIX2"] = npix//2 - 1

    w = WCS(hdu.header)

    # Now get beam values for each pixel:
    indices = np.indices((arr.shape))
    x = indices[0].flatten()
    y = indices[1].flatten()

    r, d = w.all_pix2world(x, y, 0)

    hdu.data[y, x] = beam_value(r, d, t, delays, freq, return_I=True)

    if plot:
        print("Plotting not yet implemented.")  

    if outname is not None:
        hdu.writeto(outname, clobber=True)
    if return_hdu:
        return hdu



class Lobe(object):
    """Class to hold information about a lobe of the primary beam."""


    def __init__(self, x, y, arr):

        self.x = x
        self.y = y

        # TODO control +/-1 values to ensure there are actually pixels there
        self.arr = arr[min(x)-1:max(x)+1, min(y)-1:max(y)+1]

        # These are manually added to:
        self.central_coords = None
        self.maximum_size = None


    def max_size(self):
        """Calculate max separation between pixels in the array."""

        coords = [(x, y) for x in self.x for y in self.y]

        boundary = self.arr



    @staticmethod
    def pix_to_world(x, y, wcs):
        """Convert pixel to world coordinates."""

        return  wcs.all_pix2world(x, y, 0)


    @staticmethod
    def centroid(arr):
        """Find centroid of arr.

        Taken from https://stackoverflow.com/a/19125498
        """
        
        h, w = arr.shape
        x = np.arange(h)
        y = np.arange(w)
        x1 = np.ones((1, h))
        y1 = np.ones((w, 1))

        cenx = (np.dot(np.dot(x1, arr), y)) / (np.dot(np.dot(x1, arr), y1)) 
        ceny = (np.dot(np.dot(x, arr), y1)) / (np.dot(np.dot(x1, arr), y1))

        return cenx, ceny


    @staticmethod
    def cartesian_distance(x1, y1, x2, y2):
        """Cartesian distance between pair of points."""
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)



def find_lobes(hdu, perc=0.1):
    """Find main lobe and any sidelobes in a beam image."""

    w = WCS(hdu.header)  # For converting to world coordinates.

    # First set everything < 0.1 to zero and set nans to zero:
    hdu.data[np.where(np.isnan(hdu.data))] = 0.
    hdu.data[hdu.data < perc] = 0.

    # Convert to a data format usable by scipy:
    arr = hdu.data.copy().byteswap().newbyteorder().astype("float64")
    
    lobe_image, nlabels = ndimage.label(hdu.data)

    lobes = {}
    lobe_numbers = set(lobe_image.flatten()).remove(0)

    for lobe in lobe_numbers:

        x, y = np.where(lobe_image == lobe)
        l = Lobe(x, y, arr)
        cenx, ceny = Lobe.centroid(l.arr)

        max_dist = 0





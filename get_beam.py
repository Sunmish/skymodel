#!/usr/bin/env python

from __future__ import print_function

import numpy as np

# Astopy imports:
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, FK5
from astropy import units as u
from astropy.wcs import WCS

from scipy import ndimage  # For lobe finding in the primary beam images
from scipy.spatial import distance

import logging

from mwa_pb.primary_beam import MWA_Tile_full_EE

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
        # TODO: check that this shouldn't be halved!
        return (rX[0] + rY[0])
    else:
        return rX[0], rY[0]


def atten_source(source, t, delays, freq, alpha=-0.7):
    """Attenuate a source by the primary beam response.

    Attenuate each component at the given frequency, then sum the components
    to determine total apparent brightness.
    """


    ra = np.array([source.components[i].radec.ra.value for i in range(source.ncomponents)])
    dec = np.array([source.components[i].radec.dec.value for i in range(source.ncomponents)])

    pseudoI = beam_value(ra=ra,
                         dec=dec,
                         t=t,
                         delays=delays,
                         freq=freq,
                         return_I=True)

    source.at_freq(freq=freq,
                   components=range(source.ncomponents),
                   alpha=alpha)

    atten_flux = np.array([source.components[i].at_flux*pseudoI[i] for i in
                           range(source.ncomponents)])

    total_atten_flux = np.nansum(atten_flux)

    return total_atten_flux



def make_beam_image(t, delays, freq, ra=None, outname=None, cmap="cubehelix", stretch="sqrt", 
                    npix=1500, dec=None, plot=False, return_hdu=False,
                    reference_image=None):
    """Make a FITS image of the psuedo-I beam response.

    Parameters
    ----------
    t : int or float
        GPS time in iso format. From `parse_metafits` in `skymodel.parsers`.
    delays : list
        List of delays. From `parse_metafits` in `skymodel.parsers`.
    freq : float
        Frequency in Hz. From `parse_metafits` in `skymodel.parsers`.
    ra : float
        RA for center of the image. No default as this is obs-dependent. 
    outname : str
        Output file name for the beam FITS image.
    cmap : str, optional
        Colormap to use if plotting. [Default cubehelix]
    npix : int, optional
        Size of the beam image. This changes the pixel dimensions. [Default 1500]
    dec : float, optional
        Dec. for center of the image. [Default -26]
    plot : bool, optional
        Switch true if wanting to make a simple plot. [Default False]


    """

    if reference_image is None:


        # Set RA/DEC as ALT/AZ = 90.0/180.0:
        if ra is None or dec is None:
            altaz = AltAz(alt=90.*u.deg, 
                          az=180.*u.deg,
                          obstime=t,
                          location=MWA)
            radec_at_zenith = altaz.transform_to(FK5)
            ra = radec_at_zenith.ra.value
            dec = radec_at_zenith.dec.value

        # Initialise a FITS image:
        hdu = fits.PrimaryHDU()
        arr = np.full((npix, npix), 0.)
        hdu.data = arr

        hdu.header["CTYPE1"] = "RA---SIN"
        hdu.header["CTYPE2"] = "DEC--SIN"
        hdu.header["CRVAL1"] = ra
        hdu.header["CRVAL2"] = dec
        hdu.header["CDELT1"] = -(180./npix)
        hdu.header["CDELT2"] = 180./npix
        hdu.header["CRPIX1"] = npix//2 - 1
        hdu.header["CRPIX2"] = npix//2 - 1

        hdr = hdu.header

        indices = np.indices(arr.shape)
        x = indices[0].flatten()
        y = indices[1].flatten()

    else:
        ref = fits.open(reference_image)
        ref_arr = np.squeeze(ref[0].data)
        hdr = ref[0].header
        arr = np.full_like(ref_arr, 0.)

        indices = np.where(~np.isnan(ref_arr))
        y = indices[0].flatten()
        x = indices[1].flatten()

    w = WCS(hdr).celestial
    
    # Now get beam values for each pixel:    
    stride = 2250000  # 1500*1500
    for i in range(0, len(x), stride):
        r, d = w.all_pix2world(x[i:i+stride], y[i:i+stride], 0)   
        arr[y[i:i+stride], x[i:i+stride]] = beam_value(r, d, t, delays, freq, return_I=True)

    if plot:
        print("Plotting not yet implemented.")  

    if outname is not None:
        fits.writeto(outname, arr, hdr, overwrite=True)
    if return_hdu:
        return hdu



class Lobe(object):
    """Class to hold information about a lobe of the primary beam."""


    def __init__(self, x, y, arr):

        self.original_x = x
        self.original_y = y
        # TODO control +/-1 values to ensure there are actually pixels there
        self.arr = arr[min(x)-1:max(x)+2, min(y)-1:max(y)+2]

        self.x, self.y = np.indices((self.arr.shape[0]-1, self.arr.shape[1]-1))
        self.x = self.x.flatten()
        self.y = self.y.flatten()

        # These are manually added to:
        self.ra = None
        self.dec = None
        self.sky = None
        self.maximum_size = None


    def max_size(self):
        """Calculate max separation between pixels in the array."""

        # coords = [(x, y) for x in self.x for y in self.y]s

        boundary = [Lobe.is_boundary(self.x[i], self.y[i], self.arr) for i in range(len(self.x))]
        self.boundariesx = self.x[np.where(boundary)]
        self.boundariesy = self.y[np.where(boundary)]

        boundaries = [(self.boundariesx[i], self.boundariesy[i]) for i in range(len(self.boundariesx))]
        self.size = distance.cdist(boundaries, boundaries, "euclidean").max()


    @staticmethod
    def is_boundary(x, y, arr):
        """Determine if arr[x, y] is a boundary pixel."""

        if arr[x, y] == 0.:
            return False

        if x == 0 or x == arr.shape[0]-1 or y == 0 or y == arr.shape[1]-1:
            return False

        indicesx = np.array([x+1, x+1, x+1, x-1, x-1, x-1, x, x])
        indicesy = np.array([y+1, y-1, y, y+1, y-1, y, y-1, y+1])

        if 0 in arr[indicesx, indicesy]:
            return True
        else:
            return False


    @staticmethod
    def pix_to_world(x, y, wcs):
        """Convert pixel to world coordinates."""

        return  wcs.all_pix2world(x, y, 0)


    @staticmethod
    def centroid(x, y, arr):
        """Find weighted centroid of a set of points."""

        x = x.flatten()
        y = y.flatten()
        a = arr[x, y].flatten()
        asum = np.nansum(a)

        cenx = np.nansum(x*a)/asum
        ceny = np.nansum(y*a)/asum

        return int(cenx), int(ceny)



def find_lobes(hdu, perc=0.1):
    """Find main lobe and any sidelobes in a beam image."""

    w = WCS(hdu.header)  # For converting to world coordinates.

    # First set everything < 0.1 to zero and set nans to zero:
    hdu.data[np.where(np.isnan(hdu.data))] = 0.
    hdu.data[hdu.data < perc] = 0.

    # Convert to a data format usable by scipy:
    arr = hdu.data.copy().byteswap().newbyteorder().astype("float64")
    
    lobe_image, nlabels = ndimage.label(arr)

    lobes = {}

    lobe_numbers = set(lobe_image.flatten())
    lobe_numbers.remove(0)

    for lobe in lobe_numbers:

        x, y = np.where(lobe_image == lobe)
        l = Lobe(x, y, arr)
        l.max_size()
        l.maximum_size = l.size * hdu.header["CDELT2"]
        l.cenx, l.ceny = Lobe.centroid(x, y, arr)
        ra, dec = Lobe.pix_to_world(l.ceny, l.cenx, w)

        # Avoid those pesky zero-sized arrays:
        l.ra = float(ra)
        l.dec = float(dec) 
        l.sky = SkyCoord(ra=l.ra, dec=l.dec, unit=(u.deg, u.deg))

        lobes[lobe] = l

    return lobes




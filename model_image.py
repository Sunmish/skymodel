#! /usr/bin/env python

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.DEBUG)


def fwhm_to_sigma(fwhm):
    """Convert FWHM to sigma."""
    return fwhm/np.sqrt(8.*np.log(2.))


def gauss2d(x, y, A, beta, sigma_x, sigma_y, x0, y0):
    """2D Gaussian."""

    p0 = ((x-x0)**2) / (2*sigma_x**2)
    p1 = ((y-y0)**2) / (2*sigma_y**2)
    p2 = (beta*(x-x0)*(y-y0)) / (sigma_x*sigma_y)

    return A*np.exp(-p0-p1-p2)


def convol(arr, sigma_x, sigma_y=None, pa=0.):
    """Convolve an array with a 2D Gaussian."""

    # TODO add sigma_y support

    theta = np.radians(pa)
    kern = Gaussian2DKernel(sigma_x)
    conv = convolve_fft(arr, kern)

    return conv


def bpp(major, minor, cd1, cd2):
    """
    """
    return np.pi*(major*minor) / (abs(cd1*cd2) * 4.*np.log(2.))


def create_model(ra, dec, imsize, pixsize, outname, gaussians=None, points=None):
    """Create a model image with Gaussian and/or point source models."""

    hdu = fits.PrimaryHDU()
    arr = np.full((imsize, imsize), 0.)
    hdu.data = arr

    hdu.header["CTYPE1"] = "RA---SIN"
    hdu.header["CTYPE2"] = "DEC--SIN"
    hdu.header["CRVAL1"] = ra
    hdu.header["CRVAL2"] = dec
    hdu.header["CDELT1"] = -pixsize
    hdu.header["CDELT2"] = pixsize
    hdu.header["CRPIX1"] = imsize//2 - 1
    hdu.header["CRPIX2"] = imsize//2 - 1

    w = WCS(hdu.header)

    if points is not None:
    
     for point in points:
            # (ra, dec, A)
            r, d, A = point
            logging.debug("point at {} {}, {} Jy".format(r, d, A))
            x0, y0 = w.all_world2pix(r, d, 0)
            hdu.data[int(y0), int(x0)] = A

    if gaussians is not None:

        x, y = np.indices(arr.shape)
        x, y = x.flatten(), y.flatten()

        for gaussian in gaussians:
            # (ra, dec, major, minor, pa, I)
            r, d, major, minor, pa, I = gaussian
            logging.debug("gauss at {} {}, major={}; minor={}; I={}".format(r, d, major, minor, I))
            x0, y0 = w.all_world2pix(r, d, 0)

            sigma_x = fwhm_to_sigma(major/pixsize)
            sigma_y = fwhm_to_sigma(minor/pixsize)

            A = I / (2.*np.pi*sigma_x*sigma_y)

            logging.debug("A={}".format(A))

            beta = pa*0.  # TODO

            hdu.data[y, x] += gauss2d(x, y, A, beta, sigma_x, sigma_y, x0, y0)

    import matplotlib.pyplot as plt
    plt.imshow(hdu.data)
    plt.savefig("test.png")

    fits.writeto(outname, hdu.data, hdu.header, clobber=True)



def convolve_model(model_image, major, minor=None, pa=0., rms=None, 
                   outname=None):
    """Convolve a model image with a beam.

    Input is Jy/pixel and output is Jy/beam.

    """


    if outname is None:
        outname = model_image.replace(".fits", "_image.fits")

    model = fits.open(model_image)

    cd = abs(model[0].header["CDELT1"])
    minor = major  # TODO
    sigma_x = fwhm_to_sigma(major/3600./cd)

    # bpp allows conversion between Jy/beam and Jy/pixel
    b = bpp(major/3600., minor/3600., cd, cd)

    if rms is not None:
        rms_array = np.random.normal(loc=0., 
                                     scale=rms, 
                                     size=model[0].data.shape)
        model[0].data += rms_array        

    conv = convol(np.squeeze(model[0].data), sigma_x)

    b = bpp(major/3600., minor/3600., cd, cd)

    conv *= b

    model[0].header["BUNIT"] = "JY/BEAM"
    model[0].header["BMAJ"] = major/3600.
    model[0].header["BMIN"] = minor/3600.
    model[0].header["BPA"] = 0.

    fits.writeto(outname, conv, model[0].header, overwrite=True)








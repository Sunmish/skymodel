#! /usr/bin/env python

from __future__ import print_function, division

import sys
import numpy as np
import warnings

from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft, Kernel2D
from astropy.modeling.models import Gaussian2D, Ellipse2D
from astropy.convolution.kernels import _round_up_to_odd_integer

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.DEBUG)

SIGMA_TO_FWHM = np.sqrt(8.*np.log(2.))
MAXSIZE=512
BORDER=100

class EllipticalGaussian2DKernel(Kernel2D):
    """2D Elliptical Gaussian filter kernel.
    
    The Gaussian filter is a filter with great smoothing properties. It is
    isotropic and does not produce artifacts.

    Parameters
    ----------
    stddev_maj : float
        Standard deviation of the Gaussian kernel in direction 1
    stddev_min : float
        Standard deviation of the Gaussian kernel in direction 1
    position_angle : float
        Position angle of the elliptical gaussian
    x_size : odd int, optional
        Size in x direction of the kernel array. Default = support_scaling *
        stddev.
    y_size : odd int, optional
        Size in y direction of the kernel array. Default = support_scaling *
        stddev.
    support_scaling : int
        The amount to scale the stddev to determine the size of the kernel
    mode : str, optional
        One of the following discretization modes:
            * 'center' (default)
                Discretize model by taking the value
                at the center of the bin.
            * 'linear_interp'
                Discretize model by performing a bilinear interpolation
                between the values at the corners of the bin.
            * 'oversample'
                Discretize model by taking the average
                on an oversampled grid.
            * 'integrate'
                Discretize model by integrating the
                model over the bin.
    factor : number, optional
        Factor of oversampling. Default factor = 10.

    """

    _separable = True
    _is_bool = False


    def __init__(self, stddev_maj, stddev_min, position_angle,
                 support_scaling=8, **kwargs):

        self._model = Gaussian2D(1. / (2 * np.pi * stddev_maj * stddev_min), 0,
                                 0, x_stddev=stddev_maj, y_stddev=stddev_min,
                                 theta=position_angle)

        try:
            from astropy.modeling.utils import ellipse_extent
        except ImportError:
            raise NotImplementedError("EllipticalGaussian2DKernel requires"
                                      " astropy 1.1b1 or greater.")

        max_extent = \
            np.max(ellipse_extent(stddev_maj, stddev_min, position_angle))
        self._default_size = \
            _round_up_to_odd_integer(support_scaling * 2 * max_extent)
        super(EllipticalGaussian2DKernel, self).__init__(**kwargs)
        self._truncation = np.abs(1. - 1 / self._array.sum())



def fwhm_to_sigma(fwhm):
    """Convert FWHM to sigma."""
    return fwhm/np.sqrt(8.*np.log(2.))


def gauss2d(x, y, A, theta, sigma_x, sigma_y, x0, y0):
    """General elliptical 2D Gaussian function."""


    alpha = ((np.cos(theta)**2 / (2.*sigma_x**2)) + 
             (np.sin(theta)**2 / (2.*sigma_y**2)))
    gamma = ((np.sin(theta)**2 / (2.*sigma_x**2)) +
             (np.cos(theta)**2 / (2.*sigma_y**2)))
    beta = (np.sin(2.*theta) / (4.*sigma_x**2) - 
            np.sin(2.*theta) / (4.*sigma_y**2))

    return A*np.exp(-(alpha*(x-x0)**2 + 
                      2.*beta*(x-x0)*(y-y0) + 
                      gamma*(y-y0)**2))


# def gauss2d(x, y, A, beta, sigma_x, sigma_y, x0, y0):
#     """2D Gaussian."""

#     p0 = ((x-x0)**2) / (2*sigma_x**2)
#     p1 = ((y-y0)**2) / (2*sigma_y**2)
#     p2 = (beta*(x-x0)*(y-y0)) / (sigma_x*sigma_y)

#     return A*np.exp(-p0-p1-p2)


def pa_to_beta(pa, sigma_x, sigma_y):
    theta = np.radians(pa+90.)
    beta = (np.sin(2.*theta) / (4.*sigma_x**2)) + \
        (np.cos(theta)**2 / (2.*sigma_y**2))

    return beta

# def convol(arr, sigma_x, sigma_y=None, pa=0.):
#     """Convolve an array with a 2D Gaussian."""

#     theta = np.radians(pa)
#     kern = EllipticalGaussian2DKernel(sigma_x, sigma_y, theta)
#     # kern = Gaussian2DKernel(sigma_x, y_stddev=sigma_y, theta=theta)
#     conv = convolve_fft(arr, kern, allow_huge=True)

#     return conv


def convol(arr, sigma_x, sigma_y=None, pa=0.):
    """Conlve and array with a 2D Gaussian kernel."""

    if sigma_y is None:
        sigma_y = sigma_x

    theta = np.radians(pa + 90.)

    # kern = Gaussian2DKernel(sigma_x)

    kern = EllipticalGaussian2DKernel(sigma_x, sigma_y, pa)

    if arr.shape[-2] > MAXSIZE or arr.shape[-1] > MAXSIZE:
        # Convolve array in chunks:
        
        # Determine number of subarrays:
        nx = arr.shape[-2] // MAXSIZE
        ny = arr.shape[-1] // MAXSIZE
        rx = arr.shape[-2] % MAXSIZE
        ry = arr.shape[-1] % MAXSIZE
        if rx > 0:
            nx += 1
        if ry > 0:
            ny += 1

        garr = np.squeeze(arr)
        conv_arr = np.full_like(garr, np.nan)
        garr_indices = np.indices(garr.shape)

        n = 0
        x_range = range(BORDER, garr.shape[-2], MAXSIZE)
        y_range = range(BORDER, garr.shape[-1], MAXSIZE)
        sys.stdout.write(u"\u001b[1000D" + "{:.>6.1f}%".format(100.*n/(len(x_range)*len(y_range))))
        sys.stdout.flush()

        for i in x_range:
            for j in y_range:

                sub_arr = garr[i:i+MAXSIZE, j:j+MAXSIZE]
                subx = garr_indices[0][i:i+MAXSIZE, j:j+MAXSIZE]
                suby = garr_indices[1][i:i+MAXSIZE, j:j+MAXSIZE]

                sub_garr =   garr[i-BORDER:i+MAXSIZE+BORDER, j-BORDER:j+MAXSIZE+BORDER]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    conv = convolve_fft(sub_garr, kern, allow_huge=True, boundary="fill", fill_value=0., normalize_kernel=False)
                conv = conv[BORDER:sub_arr.shape[-2]+BORDER, BORDER:sub_arr.shape[-1]+BORDER]
                conv_arr[subx, suby]  = conv

                n += 1

                sys.stdout.write(u"\u001b[1000D" + "{:.>6.1f}%".format(100.*n/(len(x_range)*len(y_range))))
                sys.stdout.flush()

        print("")

                


    else:
        conv_arr = convolve_fft(arr, kern, allow_huge=True)

    return conv_arr


def bpp(major, minor, cd1, cd2):
    """
    """
    return np.pi*(major*minor) / (abs(cd1*cd2) * 4.*np.log(2.))


def create_model_on_template(template, gaussians=None, points=None, outname=None):
    """Create model image on a template image.
    """

    hdu = fits.open(template)[0]
    hdu.data = np.full_like(hdu.data, 0.)
    pixsize = hdu.header["CDELT2"]

    w = WCS(hdu.header).celestial

    if points is not None and points:
    
        for point in points:
            # (ra, dec, A)
            r, d, A = point
            logging.debug("point at {} {}, {} Jy".format(r, d, A))
            x0, y0 = w.all_world2pix(r, d, 0)
            hdu.data[..., int(y0), int(x0)] += A

    if gaussians is not None and gaussians:

        x, y = np.indices((hdu.data.shape[-2], hdu.data.shape[-1]))
        x, y = x.flatten(), y.flatten()

        for gaussian in gaussians:
            # (ra, dec, major, minor, pa, I)
            r, d, major, minor, pa, I = gaussian
            print("gauss at {} {}, major={}; minor={}; I={}; pa={}".format(r, d, major, minor, I, pa))
            x0, y0 = w.all_world2pix(r, d, 0)

            sigma_x = fwhm_to_sigma(major/pixsize)
            sigma_y = fwhm_to_sigma(minor/pixsize)

            A = I / (2.*np.pi*sigma_x*sigma_y)

            logging.debug("A={}".format(A))
            
            # beta = pa_to_beta(pa, sigma_x, sigma_y)
            theta = np.radians(pa + 90.)
            hdu.data[..., y, x] += gauss2d(x, y, A, theta, sigma_x, sigma_y, x0, y0)

    if outname is None:
        outname = template
    fits.writeto(outname, hdu.data, hdu.header, overwrite=True)




def create_model(ra, dec, imsize, pixsize, outname, crpix, gaussians=None, points=None,
    ):
    """Create a model image with Gaussian and/or point source models."""

    hdu = fits.PrimaryHDU()
    arr = np.full((imsize, imsize), 0.)
    hdu.data = arr

    logging.debug("RA {}".format(ra))
    logging.debug("DEC {}".format(dec))

    hdu.header["CTYPE1"] = "RA---SIN"
    hdu.header["CTYPE2"] = "DEC--SIN"
    hdu.header["CRVAL1"] = ra
    hdu.header["CRVAL2"] = dec
    hdu.header["CDELT1"] = -pixsize
    hdu.header["CDELT2"] = pixsize
    hdu.header["CRPIX1"] = crpix[0]
    hdu.header["CRPIX2"] = crpix[1]

    w = WCS(hdu.header)

    if points is not None:
    
        for point in points:
            # (ra, dec, A)
            r, d, A = point
            logging.debug("point at {} {}, {} Jy".format(r, d, A))
            x0, y0 = w.all_world2pix(r, d, 0)
            hdu.data[int(y0), int(x0)] += A

    if gaussians is not None:

        x, y = np.indices(arr.shape)
        x, y = x.flatten(), y.flatten()

        for gaussian in gaussians:
            # (ra, dec, major, minor, pa, I)
            r, d, major, minor, pa, I = gaussian
            print("gauss at {} {}, major={}; minor={}; I={}; pa={}".format(r, d, major, minor, I, pa))
            x0, y0 = w.all_world2pix(r, d, 0)

            sigma_x = fwhm_to_sigma(major/pixsize)
            sigma_y = fwhm_to_sigma(minor/pixsize)

            A = I / (2.*np.pi*sigma_x*sigma_y)

            logging.debug("A={}".format(A))
            
            # beta = pa_to_beta(pa, sigma_x, sigma_y)
            theta = np.radians(pa + 90.)
            hdu.data[y, x] += gauss2d(x, y, A, theta, sigma_x, sigma_y, x0, y0)

    fits.writeto(outname, hdu.data, hdu.header, clobber=True)



def convolve_model(model_image, major, minor=None, pa=0., rms=None, 
                   outname=None, no_bpp=False):
    """Convolve a model image with a beam.

    Input is Jy/pixel and output is Jy/beam.

    """


    if outname is None:
        outname = model_image.replace(".fits", "_image.fits")

    model = fits.open(model_image)

    cd = abs(model[0].header["CDELT1"])
    # minor = major  # TODO
    sigma_x = fwhm_to_sigma(major/3600./cd)
    sigma_y = fwhm_to_sigma(minor/3600./cd)

    # bpp allows conversion between Jy/beam and Jy/pixel
    b = bpp(major/3600., minor/3600., cd, cd)

    if rms is not None:
        rms /= b
        rms_array = np.random.normal(loc=0., 
                                     scale=rms, 
                                     size=model[0].data.shape)
        model[0].data += rms_array        

    conv = convol(np.squeeze(model[0].data), sigma_x, sigma_y, pa)

    b = bpp(major/3600., minor/3600., cd, cd)

    if not no_bpp:
        conv *= b

    model[0].header["BUNIT"] = "JY/BEAM"
    model[0].header["BMAJ"] = major/3600.
    model[0].header["BMIN"] = minor/3600.
    model[0].header["BPA"] = 0.

    fits.writeto(outname, conv, model[0].header, overwrite=True)








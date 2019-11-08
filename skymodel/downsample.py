#! /usr/bin/env python

from __future__ import print_function, division

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def downsample(image, outsize, outname):
    """
    """

    with fits.open(image) as f:

        w = WCS(f[0].header).celestial
        fdata = np.squeeze(f[0].data).copy()
        x, y = np.indices(np.squeeze(f[0].data).shape)

        # r, d = w.all_pix2world(x, y, 0)

        stridex = f[0].header["NAXIS1"] // outsize
        # stridey = f[0].header["NAXIS2"] // outsize
        # logger.debug("stridex = {}".format(stridex))

        mid_stridex = stridex // 2
        mid_r = f[0].header["NAXIS1"] % outsize
        logger.debug("stridex: {}, mid_r: {}".format(stridex, mid_r))
        # mid_r = 0
        # mid_stridey = stridey // 2

        if "CD1_1" in f[0].header.keys():
            cdx = f[0].header["CD1_1"]
            cdy = f[0].header["CD2_2"]
        elif "CDELT1" in f[0].header.keys():
            cdx = f[0].header["CDELT1"]
            cdy = f[0].header["CDELT2"]
        else:
            raise ValueError("No pixel scale information found!")

        cellx = stridex * cdx  # Negative!
        celly = stridex * cdy

        logger.debug("cell: {}".format(celly))

        out_arr = np.full((outsize, outsize), np.nan)

        X, Y = [], []
        I, J = [], []
        ra, dec = [], []

        nx = 0
        for i in range(mid_stridex, f[0].header["NAXIS1"]-stridex-mid_r, stridex):
            ny = 0
            for j in range(mid_stridex, f[0].header["NAXIS1"]-stridex-mid_r, stridex):

                # logger.debug("{}  , {}".format(nx, ny))

                X.append(nx)
                Y.append(ny)
                I.append(i)
                J.append(j)

                # r, d = w.all_pix2world(i, j, 0)

                # ra.append(r)
                # dec.append(d)

                # Get average z:
                z_avg = np.nanmean(fdata[i-mid_stridex:i+mid_stridex, j-mid_stridex:j+mid_stridex])
# 
                out_arr[nx, ny] = z_avg

                ny += 1
            nx += 1


    ra, dec = w.all_pix2world(np.asarray(I), np.asarray(J), 0)

    # Initialise output FITS image:
    hdu = fits.PrimaryHDU()
    hdu.data = out_arr

    hdu.header["CTYPE1"] = "RA---SIN"
    hdu.header["CTYPE2"] = "DEC--SIN"
    hdu.header["CRVAL1"] = ra[len(X)//2]
    hdu.header["CRVAL2"] = dec[len(X)//2]
    hdu.header["CRPIX1"] = X[len(X)//2]
    hdu.header["CRPIX2"] = Y[len(X)//2]
    hdu.header["CDELT1"] = cellx
    hdu.header["CDELT2"] = celly 

    hdu.writeto(outname, overwrite=True)

    # hdu.close()     
    return outname


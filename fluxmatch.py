import numpy as np

import sparse_grid_interpolator
reload(sparse_grid_interpolator)
rbf = sparse_grid_interpolator.rbf

from astropy.io import fits
from astropy.wcs import WCS

from skymodel.fitting import cpowerlaw, powerlaw

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def sigma_clip(ratios, indices, sigma=2.):
    """
    """

    avg = np.nanmean(ratios)
    std = np.nanstd(ratios)

    new_indices = np.where(abs(ratios-avg) < sigma*std)[0]

    return ratios[new_indices], indices[new_indices]


def nsrc_cut(table, flux_key, nsrc_max):
    """
    """

    if len(table) > nsrc_max:
        threshold = round(np.sort(table[flux_key])[::-1][nsrc_max-1], 1)
        table = table[table[flux_key] > threshold]

    return table



def fluxscale(table, freq, threshold=1., ref_freq=154., spectral_index=-0.77,
              flux_key="flux", nsrc_max=100):
    """
    """

    ref_freq_key = "S{}".format(int(ref_freq))

    predicted_flux, indices = [], []
    ratios = []

    # table = nsrc_cut(table, flux_key, nsrc_max)


    for i in range(len(table)):

        if not np.isnan(table["alpha_c"][i]):

            # Use the curved power law fit:
            f = cpowerlaw(freq, *[table[p+"_c"][i] for p in ["alpha", "beta", "gamma"]])

        elif not np.isnan(table["alpha_p"][i]):

            # Use standard power law fit:
            f = powerlaw(freq, *[table[p+"_c"][i] for p in ["alpha", "beta"]])

        elif not np.isnan(table[ref_freq_key][i]):

            f = table[ref_freq_key][i]*(freq/ref_freq)**spectral_index

        else:
            continue

        if f > threshold:

            indices.append(i)
            predicted_flux.append(f)

            ratios.append(f/table[flux_key][i])


    logging.info("Number of calibrators: {}".format(len(predicted_flux)))

    # predicted_ratios, predicted_indices = sigma_clip(ratios, indices)
    predicted_ratios, predicted_indices = ratios, indices

    return np.asarray(predicted_ratios), np.asarray(predicted_indices)



def correction_factor_map(image, pra, pdec, ratios, interpolation="linear",
                          memfrac=0.5, absmem="all", outname=None): 
    """
    """

    if outname is None:
        outname = image.replace(".fits", "_scale_factors.fits")

    rbf(image=image,
        x=pra,
        y=pdec,
        z=ratios,
        interpolation=interpolation,
        smooth=0,
        world_coords=True,
        memfrac=memfrac,
        absmem=absmem,
        outname=outname,
        const=None,
        constrain=True
        )

    return outname


def apply_corrections(image, correction_image, outname):
    """
    """

    im = fits.open(image)
    ref = fits.getdata(correction_image)

    cdata = im[0].data.copy()
    cdata *= ref

    fits.writeto(outname, cdata, im[0].header, overwrite=True)



def plot(correction_image, pra, pdec, ratios, cmap="cubehelix", 
         outname=None):
    """
    """

    size = (10, 8)
    axes = [0.15, 0.1, 0.83, 0.78]
    cbax = [0.15, 0.96, 0.83, 0.03]

    wcs = WCS(fits.getheader(correction_image)).celestial

    fig = plt.figure(figsize=size)
    ax1 = plt.subplot(111, projection=wcs)
    ax1.set_position(axes)

    ax1.scatter(pra, pdec, s=150, marker="*", c="green")

    norm = mpl.colors.Normalize(vmin=min(ratios), vmax=max(ratios))

    ax1.imshow(np.squeeze(fits.getdata(correction_image)), cmap=cmap, norm=norm,
               origin="lower")


    colorbar_axis = fig.add_axes(cbax)
    colorbar = mpl.colorbar.ColorbarBase(colorbar_axis, cmap=cmap, norm=norm,
                                         orientation="horizontal")
    colorbar.set_label("Correction factor", 
                       fontsize=24.,
                       labelpad=0.,
                       verticalalignment="top",
                       horizontalalignment="center")
    colorbar.ax.tick_params(which="major",
                            labelsize=20.,
                            length=5.,
                            color="black",
                            labelcolor="black",
                            width=1.)

    if outname is None:
        outname = correction_image.replace(".fits", ".eps")

    fig.savefig(outname, dpi=72)









import numpy as np

import sparse_grid_interpolator
reload(sparse_grid_interpolator)
rbf = sparse_grid_interpolator.rbf

from skymodel.fitting import cpowerlaw, powerlaw

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)


def clip(ratios, indices, sigma=2.):
    """
    """

    avg = np.nanmean(ratios)
    std = np.nanstd(ratios)

    new_indices = np.where(abs(ratios-avg) < sigma*std)[0]

    return ratios[new_indices], indices[new_indices]


def fluxscale(table, freq, threshold=1., ref_freq=154., spectral_index=-0.77,
              flux_key="flux"):
    """
    """

    ref_freq_key = "S{}".format(int(ref_freq))

    predicted_flux, indices = [], []
    ratios = []

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

    predicted_ratios, predicted_indices = clip(ratios, indices)

    return predicted_ratios, predicted_indices



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












import numpy as np

import sparse_grid_interpolator
reload(sparse_grid_interpolator)
rbf = sparse_grid_interpolator.rbf

from astropy.io import fits
from astropy.wcs import WCS

from skymodel.fitting import cpowerlaw, powerlaw

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.DEBUG)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import rc
# Without this the default font will have issues with MNRAS (being type 3)
rc('font', **{'family':'serif', 'serif':['Times'], 'weight':'medium'})
rc('text', usetex=True)
params = {"text.latex.preamble": [r"\usepackage{siunitx}", \
          r"\sisetup{detect-family = true}"]}
plt.rcParams.update(params)

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


def hist_plot(ratios, outname, color="black"):
    """
    """

    font_labels = 24.
    font_ticks = 20.
    size = (10, 8)
    axes = [0.15, 0.1, 0.81, 0.86]

    plt.close("all")

    fig = plt.figure(figsize=size)
    ax1 = plt.axes(axes)

    ax1.hist(ratios, 20, color=color, histtype="step")
    ax1.set_xlabel("Correction factor", fontsize=font_labels)
    ax1.set_ylabel("Number", fontsize=font_labels)
    ax1.tick_params(axis="both", which="both", labelsize=font_ticks)

    plt.savefig(outname)

    plt.close("all")


def sigma_clip(ratios, indices, sigma=2.):
    """
    """

    avg = np.nanmean(ratios)
    std = np.nanstd(ratios)

    new_indices = np.where(abs(ratios-avg) < sigma*std)[0]


    return ratios[new_indices], indices[new_indices]


def nsrc_cut(table, flux_key, indices, nsrc_max, ratios):
    """
    """


    cflux = table[indices][flux_key]*ratios


    if len(cflux) > nsrc_max:
        threshold = round(np.sort(cflux)[::-1][nsrc_max-1], 1)
        indices = indices[cflux > threshold]
        ratios = ratios[cflux > threshold]
        logging.info("New threshold set to {:.1f} Jy".format(threshold))

    return indices, ratios



def fluxscale(table, freq, threshold=1., ref_freq=154., spectral_index=-0.77,
              flux_key="flux", nsrc_max=100, histnamebase="table"):
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




    # logging.info("Number of calibrators prior to clipping: {}".format(len(predicted_flux)))


    valid = np.isfinite(ratios)
    ratios = np.asarray(ratios)[valid]
    indices = np.asarray(indices)[valid]


    logging.info("Number of calibrators prior to clipping: {}".format(len(indices)))

    hist_plot(ratios, histnamebase+"_hist1.eps", color="dodgerblue")
    
    predicted_ratios, predicted_indices = sigma_clip(np.asarray(ratios), np.asarray(indices))
    
    logging.info("Number of calibrators after clipping: {}".format(len(predicted_indices)))

    predicted_indices, predicted_ratios = nsrc_cut(table, flux_key, predicted_indices, 
                                 nsrc_max, predicted_ratios)

    hist_plot(predicted_ratios, histnamebase+"_hist2.eps", color="red")

    logging.info("Number of calibrators: {}".format(len(predicted_indices)))
    return np.asarray(predicted_ratios), np.asarray(predicted_indices)



def correction_factor_map(image, pra, pdec, ratios, interpolation="linear",
                          memfrac=0.5, absmem="all", outname=None,
                          smooth=0): 
    """
    """



    if outname is None:
        outname = image.replace(".fits", "_{}_factors.fits".format(interpolation))

    if interpolation.lower()[:3] == "con":

        factor = np.nanmean(ratios)

        with fits.open(image) as f:

            factors = np.full_like(f[0].data, factor)

            fits.writeto(outname, factors, f[0].header)

    else:

        rbf(image=image,
            x=pra,
            y=pdec,
            z=ratios,
            interpolation=interpolation,
            smooth=smooth,
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

    
    


    size = (10, 10)
    axes = [0.15, 0.1, 0.83, 0.8]
    cbax = [0.15, 0.975, 0.83, 0.02]
    fig = plt.figure(figsize=size)

    norm = mpl.colors.Normalize(vmin=min(ratios), vmax=max(ratios))


    try:
        import aplpy
        from matplotlib import rc


        rc('font', **{'family':'serif', 'serif':['Times'], 'weight':'medium'})
        rc('text', usetex=True)
        params = {"text.latex.preamble": [r"\usepackage{siunitx}",
                  r"\sisetup{detect-family = true}"]}
        plt.rcParams.update(params)

        # By default ticks are all out - they shouldn't be!
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'

        apl = aplpy.FITSFigure(correction_image, fig, axes)
        apl.show_colorscale(vmin=norm.vmin, vmax=norm.vmax, cmap=cmap)

        apl.ticks.set_length(8)
        apl.ticks.set_linewidth(1)
        apl.ticks.set_minor_frequency(10)
        apl.tick_labels.set_font(size=22.)

        apl.axis_labels.set_font(size=24.)
        apl.axis_labels.set_xpad(10)
        apl.axis_labels.set_ypad(5)
    
        apl.axis_labels.set_xtext(r"R.~A. (J2000)")
        apl.axis_labels.set_ytext(r"Decl. (J2000)")

        apl.show_markers(pra, pdec, c=ratios, edgecolors="magenta", s=100,
                         marker="o")

    except Exception:

        wcs = WCS(fits.getheader(correction_image)).celestial

        apl = plt.axes(axes)
        apl.imshow(np.squeeze(fits.getdata(correction_image)), cmap=cmap, norm=norm,
                   origin="lower")
    
        x, y = wcs.all_world2pix(pra, pdec, 0)

        apl.scatter(x, y, c=ratios, edgecolors="magenta", s=100, marker="o", 
                    norm=norm, cmap=cmap)

        apl.axes.get_xaxis().set_ticks([])
        apl.axes.get_yaxis().set_ticks([])


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

    fig.savefig(outname, dpi=72, bbox_inches="tight")









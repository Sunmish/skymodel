import numpy as np

import sparse_grid_interpolator
reload(sparse_grid_interpolator)
rbf = sparse_grid_interpolator.rbf

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

from skymodel.fitting import cpowerlaw, powerlaw

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rc
# Without this the default font will have issues with MNRAS (being type 3)
rc('font', **{'family':'serif', 'serif':['Times'], 'weight':'medium'})
rc('text', usetex=True)

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


# Coordinates from Hurley-Walker et al. 2017 and radius +0.5 degrees.
LMC = {"coords": SkyCoord(ra="05h23m35s", dec="-69d45m22s", unit=(u.hourangle, u.deg)),
       "radius": 6.0,
       "name": "LMC"}

SMC = {"coords": SkyCoord(ra="00h52m38s", dec="-72d48m01s", unit=(u.hourangle, u.deg)),
       "radius": 3.0,
       "name": "SMC"}

SOURCES = [LMC, SMC]


def writeout_region(ra, dec, ratio, outname, color="green"):
    """
    """

    with open(outname, "w+") as f:

        for i in range(len(ra)):

            size = 0.1

            f.write("fk5;circle {}d {}d {}d # color={}\n".format(ra[i], dec[i], size, color))


def flux_from_index(flux1, freq1, freq2, alpha):
    """Calculate flux density at a given frequency."""

    return flux1*(freq2 / freq1)**alpha 


def sigma_clip(ratios, indices, sigma=2., table=None,
               outname=None):
    """
    """

    avg = np.nanmean(ratios)
    std = np.nanstd(ratios)

    new_indices = np.where(abs(ratios-avg) < sigma*std)[0]

    if table is not None and outname is not None:
       
        clipped_indices = np.where(abs(ratios-avg) >= sigma*std)[0]
        clipped_ratios = ratios[clipped_indices]
        clipped_ra = table["ra"][clipped_indices]
        clipped_dec = table["dec"][clipped_indices]

        writeout_region(ra=clipped_ra,
                        dec=clipped_dec,
                        ratio=clipped_ratios,
                        outname=outname,
                        color="yellow")

    return ratios[new_indices], indices[new_indices]


def nsrc_cut(table, flux_key, indices, nsrc_max, ratios):
    """
    """


    cflux = table[indices][flux_key]/ratios


    if len(cflux) > nsrc_max:
        threshold = round(np.sort(cflux)[::-1][nsrc_max-1], 1)
        indices = indices[cflux > threshold]
        ratios = ratios[cflux > threshold]
        logger.info("New threshold set to {:.1f} Jy".format(threshold))

    return ratios, indices



def fluxscale(table, freq, threshold=1., ref_freq=154., spectral_index=-0.77,
              flux_key="flux", nsrc_max=100, region_file_name="table",
              ignore_magellanic=True, extrapolate=False, curved=True):
    """
    """


    ref_freq_key = "S{:0>3}".format(int(ref_freq))

    predicted_flux, indices = [], []
    ratios = []

    for i in range(len(table)):

        if ignore_magellanic:

            in_magellanic = False

            source_coords = SkyCoord(ra=table["ra"][i], 
                                     dec=table["dec"][i],
                                     unit=(u.deg, u.deg))

            for magellanic in SOURCES:
                sep_magellanic = magellanic["coords"].separation(source_coords)
                if sep_magellanic.value <= magellanic["radius"]:
                    logger.debug("Skipping {} as it is within the {}".format(i, magellanic["name"]))
                    in_magellanic = True

            if in_magellanic:
                continue


        if (not np.isnan(table["alpha_c"][i])) and (not extrapolate) and curved:

            # Use the curved power law fit:
            f = cpowerlaw(freq, *[table[p+"_c"][i] for p in ["alpha", "beta", "gamma"]])


        elif not np.isnan(table["alpha_p"][i]):

            if extrapolate:
                if not np.isnan(table[ref_freq_key][i]):
                    f = flux_from_index(flux1=table[ref_freq_key][i], 
                                        freq1=ref_freq,
                                        freq2=freq, 
                                        alpha=table["beta_p"][i])
                else:
                    continue

            else:

                # Use standard power law fit:
                f = powerlaw(freq, *[table[p+"_p"][i] for p in ["alpha", "beta"]])


        elif not np.isnan(table[ref_freq_key][i]):

            f = flux_from_index(flux1=table[ref_freq_key][i],
                                freq1=ref_freq,
                                freq2=freq,
                                alpha=spectral_index)

        else:
            continue

        if f > threshold:

            indices.append(i)
            predicted_flux.append(f)

            ratios.append(table[flux_key][i]/f)



    valid = np.isfinite(ratios)
    ratios = np.asarray(ratios)[valid]
    indices = np.asarray(indices)[valid]


    logger.info("Number of calibrators prior to clipping: {}".format(len(indices)))

    all_ratios = [ratios]

    p_ratios, p_indices = sigma_clip(ratios=np.asarray(ratios), 
                                     indices=np.asarray(indices),
                                     table=table,
                                     outname=region_file_name.replace(".reg", "_sigma3.reg"),
                                     sigma=3)
    p_ratios, p_indices = sigma_clip(ratios=p_ratios,
                                     indices=p_indices,
                                     table=table,
                                     outname=region_file_name.replace(".reg", "_sigma2.reg"),
                                     sigma=2)

    
    logger.info("Number of calibrators after clipping: {}".format(len(p_indices)))

    all_ratios.append(p_ratios)

    p_ratios, p_indices = nsrc_cut(table, flux_key, p_indices, 
                                 nsrc_max, p_ratios)

    all_ratios.append(p_ratios)

    logger.info("Number of calibrators: {}".format(len(p_indices)))

    return p_ratios, p_indices, all_ratios



def correction_factor_map(image, pra, pdec, ratios, interpolation="linear",
                          memfrac=0.5, absmem="all", outname=None,
                          smooth=0, writeout=True): 
    """
    """


    if writeout:
        writeout_region(pra, pdec, ratios, image.replace(".fits", "_calibrators.reg"),
            color="magenta")


    if outname is None:
        outname = image.replace(".fits", "_{}_factors.fits".format(interpolation))

    if interpolation.lower()[:3] == "con":

        factor = np.nanmedian(ratios)

        with fits.open(image) as f:

            factors = np.full_like(f[0].data, factor)

            fits.writeto(outname, factors, f[0].header, overwrite=True)

    else:

        logger.info("Passing {} off to sparse_grid_interpolator...".format(image))
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

    logger.info("Mean factor S_predic/S_meas: {}".format(np.nanmean(ref)))

    cdata = im[0].data.copy()
    cdata /= ref


    fits.writeto(outname, cdata, im[0].header, overwrite=True)




def plot(correction_image, pra, pdec, ratios, cmap="cubehelix", 
         outname=None):
    """
    """

    cmap = plt.get_cmap(cmap, 11)
    colors = [cmap(2), cmap(7)]


    plt.close("all")
    
    font_labels = 20.
    font_ticks = 16.
    R = 1.


    figsize = (12, 8)
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(2, 3, wspace=0.08*R, hspace=0.12, left=0.02*R, 
                  right=1-0.1*R, top=0.95, bottom=0.15)


    apl = plt.subplot(gs[:, :-1])
    ax2 = plt.subplot(gs[0, -1])
    ax3 = plt.subplot(gs[1, -1])

    ax2.boxplot(ratios, medianprops={"color": colors[0]}, vert=False)
    ax2.yaxis.tick_right()
    ax2.tick_params(axis="both", which="both", labelsize=font_ticks)
    ax2.tick_params(axis="y", which="major", pad=7)

    labels = [item.get_text() for item in ax2.get_yticklabels()]
    labels[0] = "Original"
    labels[1] = r"$\sigma$-clip"
    labels[2] = r"$N_{\mathrm{src}}$ cut"
    ax2.set_yticklabels(labels, rotation=90, va="center")


    map_ratios = fits.getdata(correction_image).flatten()


    ax3.hist([map_ratios, ratios[2]], bins=40, histtype="step",
             color=colors, fill=False, 
             label=["Interpolation", "Calibrators"], density=True)
    legend = ax3.legend(loc="upper right", shadow=False, fancybox=False, frameon=True,
                        fontsize=16, numpoints=1)
    legend.get_frame().set_edgecolor("dimgrey")

    ax3.set_xlabel(r"$S_\mathrm{measured}/S_\mathrm{predicted}$", fontsize=font_labels)
    ax3.yaxis.tick_right()
    ax3.tick_params(axis="both", which="both", labelsize=16.)

    norm = mpl.colors.Normalize(vmin=min(ratios[2]), vmax=max(ratios[2]))

    sb1 = SubplotSpec(gs, 0)
    sp1 = sb1.get_position(figure=fig).get_points().flatten()
    x = sp1[0]

    gp1 = apl.get_position().get_points().flatten() 
    dx = gp1[2] - gp1[0]
    sb2 = SubplotSpec(gs, 4)
    sp2 = sb2.get_position(figure=fig).get_points().flatten()
    y = sp2[1]
    cbax = [x, y-0.025, dx, 0.02]

    wcs = WCS(fits.getheader(correction_image)).celestial


    
    apl.imshow(np.squeeze(fits.getdata(correction_image)), cmap=cmap, norm=norm,
               origin="lower", aspect="auto")

    x, y = wcs.all_world2pix(pra, pdec, 0)

    apl.scatter(x, y, c=ratios[2], edgecolors="magenta", s=100, marker="o", 
                norm=norm, cmap=cmap)

    apl.axes.get_xaxis().set_ticks([])
    apl.axes.get_yaxis().set_ticks([])


    colorbar_axis = fig.add_axes(cbax)
    colorbar = mpl.colorbar.ColorbarBase(colorbar_axis, cmap=cmap, norm=norm,
                                         orientation="horizontal")
    colorbar.set_label(r"$S_\mathrm{measured}/S_\mathrm{predicted}$", 
                       fontsize=20.,
                       labelpad=0.,
                       verticalalignment="top",
                       horizontalalignment="center")
    colorbar.ax.tick_params(which="major",
                            labelsize=16.,
                            length=5.,
                            color="black",
                            labelcolor="black",
                            width=1.)

    if outname is None:
        outname = correction_image.replace(".fits", ".png")

    fig.savefig(outname, dpi=300, bbox_inches="tight")









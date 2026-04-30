#! /usr/bin/env python

from argparse import ArgumentParser

import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

# from skymodel.create_skymodel import point_formatter
from skymodel.get_beam import beam_value
from skymodel.parsers import parse_metafits

from flux_warp.models import cpowerlaw, cpowerlaw_amplitude, powerlaw, from_index


from tqdm import trange

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)



def get_args():
    ps = ArgumentParser()
    ps.add_argument("metafits")
    ps.add_argument("catalogue")
    ps.add_argument("catalogue_freq", type=float)
    ps.add_argument("-o", "--outname", default=None)
    ps.add_argument("-i", "--image_template", default=None)
    ps.add_argument("--flux_key", default="int_flux")
    ps.add_argument("--ra_key", default="ra")
    ps.add_argument("--dec_key", default="dec")
    ps.add_argument("--alpha_key", default=None)
    ps.add_argument("--assumed_alpha", default=-0.8, type=float)
    ps.add_argument("--apply_beam", action="store_true")
    ps.add_argument("--fits_mask", action="store_true")
    ps.add_argument("--box_size", default=1, type=int)
    ps.add_argument("--flux_threshold", default=0. ,type=float)
    return ps.parse_args()


def point_formatter(name, ra, dec, freq, flux, precision=3):
    """Format point source for ao-cal skymodel."""

    measurements = ""
    for i in range(len(freq)):
        measurements += "measurement {{\n" \
                       "frequency {freq} MHz\n" \
                       "fluxdensity Jy {flux} 0.0 0.0 0.0 \n" \
                       "}}\n".format(freq=freq[i], flux=flux[i])

    point = "\nsource {{\n" \
            "name \"{name}\"\n" \
            "component {{\n" \
            "type point\n" \
            "position {ra} {dec}\n" \
            "{measurement}" \
            "}}\n" \
            "}}\n".format(name=name, ra=ra, dec=dec, measurement=measurements)

    return point


def create_single_attenuated_model(table, metafits, outname,
    catalogue_freq,
    template_image=None,
    flux_key="int_flux",
    ra_key="ra",
    dec_key="dec",
    alpha_key=None,
    alpha_assumed=-0.8,
    threshold=0.,
    apply_beam=False,
    box_size=1
):
    """Create a single model based on a input catalogue.
    Returns Stokes I values attenuated by the beam
    """

    t, delays, freq, pnt = parse_metafits(metafits)
    freq /= 1.e6

    fluxes = np.full((len(table),), np.nan)

    if alpha_key is None:
        alpha = np.full((len(table),), alpha_assumed)
    else:
        alpha = table[alpha_key]

    # freqs = np.arange(285, 316, 5)
    freqs = np.asarray([288.66, 295.98, 300,303.3, 310.62])
    # freqs = np.asarray([145., 147., 150., 152., 154., 158., 160., 165.])
    fluxes = np.full((len(table),len(freqs)), np.nan)

    # print(fluxes.shape)
    for i in range(len(freqs)):
        fluxes[:, i] = from_index(
            x=freqs[i],
            x1=catalogue_freq,
            y1=table[flux_key],
            index=alpha
        )

    if apply_beam:
        for i in range(len(freqs)):
            beam_vals = np.asarray(beam_value(
                ra=table[ra_key].value,
                dec=table[dec_key].value,
                t=t,
                delays=delays,
                freq=freqs[i]*1e6,
                return_I=True
            ))
            fluxes[:, i] *= beam_vals

        # atten_fluxes = fluxes*beam_vals


    # idx = np.where(np.isfinite(fluxes))[0]
    coords = SkyCoord(
        ra=table[ra_key],
        dec=table[dec_key],
        unit=(u.deg, u.deg)
    )

    if template_image is not None:

        fluxes = fluxes[:, 2]
        print(fluxes)
        idx = np.where(fluxes > threshold)[0]
        print("Keeping {}/{} sources above apparent brightness of {}".format(
            len(idx), len(fluxes), threshold
        ))
        
        with fits.open(template_image) as f:
            f[0].data[..., :, :] = 0
            wcs = WCS(f[0].header).celestial
            x, y = wcs.all_world2pix(coords.ra.value, coords.dec.value, 0)
            y = y.astype(int)
            x = x.astype(int)
            for i in range(len(coords)):
                if i in idx:
                    try:
                        f[0].data[..., 
                            y[i]-box_size:y[i]+box_size+1, 
                            x[i]-box_size:x[i]+box_size+1
                        ] = 1
                    except IndexError:
                        pass

            f.writeto(outname, overwrite=True)

        
                    

            
    
    else:

        with open(outname, "w+") as f:
            f.write("skymodel fileformat 1.1\n")

            for i in range(len(fluxes[:, 0])):

                if np.isfinite(fluxes[i, :]).all() and np.nanmin(fluxes[i, :]) > 0.:

                    r, d = coords[i].to_string("hmsdms").split()
                    name = "J{}{}{}{}".format(r[:2], r[3:5], d[:3], d[4:6])
                    
                    entry_format = point_formatter(
                        name=name,
                        ra=r,
                        dec=d,
                        freq=freqs,
                        flux=fluxes[i, :]
                    )
                
                    f.write(entry_format)



def clip_catalogue_with_template(catalogue, image_template,
    ra_key="ra_1",
    dec_key="dec_1"):
    """
    """

    table = Table.read(catalogue)

    coords = SkyCoord(
        ra=table[ra_key],
        dec=table[dec_key],
        unit=(u.deg, u.deg)
    )

    with fits.open(image_template) as f:
        wcs = WCS(f[0].header).celestial
        y, x = wcs.all_world2pix(coords.ra.value, coords.dec.value, 0)
        cond1 = np.isfinite(x)
        cond2 = np.isfinite(y)
        idx = np.where(cond1 & cond2)[0]
        y = y[idx]
        x = x[idx]
        table = table[idx]
        y = y.astype(int)
        x = x.astype(int)
        x_indices = np.indices(f[0].data.shape)[-1]
        y_indices = np.indices(f[0].data.shape)[-2]
        idx = []

        
        idx = np.where(np.isin(x, x_indices) & np.isin(y, y_indices))[0]
        # for i in pbar:
            # if np.isfinite(x[i]) and np.isfinite(y[i]):
                # if x[i] in x_indices and y[i] in y_indices:
                    # idx.append(i)

    idx = np.asarray(idx)
    print("{} / {} sources within snapshot boundary".format(
        len(idx), len(table)
    ))

    return table[idx]




def cli(args):
    if args.outname is None:
        args.outname = args.metafits.split(".")[0]+"_skymodel.txt"
    
    print(args.image_template)
    if args.image_template is not None:
        print("Clipping catalogue to template image bounds")
        model_table = clip_catalogue_with_template(
            catalogue=args.catalogue,
            image_template=args.image_template,
            ra_key=args.ra_key,
            dec_key=args.dec_key
        )
    else:
        model_table = Table.read(args.catalogue)

    if args.fits_mask and args.image_template is not None:
        template = args.image_template
    else:
        template = None
    create_single_attenuated_model(
        table=model_table,
        metafits=args.metafits,
        outname=args.outname,
        template_image=template,
        catalogue_freq=args.catalogue_freq,
        flux_key=args.flux_key,
        ra_key=args.ra_key,
        dec_key=args.dec_key,
        threshold=args.flux_threshold,
        alpha_key=args.alpha_key,
        alpha_assumed=args.assumed_alpha,
        apply_beam=args.apply_beam
    )


if __name__ == "__main__":
    cli(get_args())










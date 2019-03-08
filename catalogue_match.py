import numpy as np
import os

from astropy.io import fits, votable
from astropy.coordinates import SkyCoord
from astropy import units as u

import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Catalogue(object):
    """Easy handling of a catalogue."""


    def __init__(self, catalogue, ra_key, dec_key,
                 flux=None, eflux=None,
                 local_rms=None):
        """

        Parameters
        ----------
        catalogue : str
            Filepath to a catalogue in a .fits or .vot format.
        ra_key : str
            Key for RA column in `catalogue`.
        dec_key : str
            Key for DEC column in `catalogue`.
        """

        self.name = os.path.basename(catalogue)

        self.table = Catalogue.open_catalogue(catalogue)
        self.ra_key = ra_key
        self.dec_key = dec_key

        self.coords = SkyCoord(ra=self.table[self.ra_key],
                               dec=self.table[self.dec_key],
                               unit=(u.deg, u.deg)) 

        self.flux = flux
        self.eflux = eflux
        self.local_rms = local_rms


    def exclude(self, exclude_coords, exclusion_zone):
        """Exclude sources within the specified exclusion zone/s."""

        indices = []

        for i in range(len(self.coords)):

            excl_seps = self.coords[i].separation(exclude_coords)
            if (excl_seps.value < exclusion_zone).any():
                continue
            else:
                indices.append(i)

        self.table = self.table[indices]
        self.coords = self.coords[indices]


    def only_within(self, coords, radius):
        """Remove sources outside of the radius specified."""

        sep = coords.separation(self.coords)

        self.table = self.table[np.where(sep.value < radius)]
        self.coords = self.coords[np.where(sep.value < radius)]


    def clip(self, threshold):

        if self.flux is not None:
            self.coords = self.coords[self.table[self.flux] > threshold]
            self.table = self.table[self.table[self.flux] > threshold]




    @staticmethod
    def open_catalogue(catalogue):
        """Open a catalogue file. 

        Must be either .vot or .fits."""

        if catalogue.endswith(".vot"):
            table = votable.parse_single_table(catalogue).array
        elif catalogue.endswith(".fits"):
            table = fits.open(catalogue)[1].data
        else:
            raise IOError("Catalogue file ({}) must be either a VOTable (.vot) "
                          "or a FITS table (.fits).".format(catalogue))

        return table



def match(cat1, cat2, separation, exclusion=0.):
    """Match two catalogues.

    Incorporates a minimum separation and an exclusion zone.
    """

    idx1, sep1, _ = cat1.coords.match_to_catalog_sky(cat2.coords)
    idx2, sep2, _ = cat1.coords.match_to_catalog_sky(cat2.coords,
                                                     nthneighbor=2)

    if isinstance(separation, np.ndarray):
        separation = separation[idx1]


    cond = np.where((sep1.value < separation) & (sep2.value > exclusion))[0]

    indices = np.array([cond, idx1[cond]]).T

    if len(idx1[cond]) == 0.:
        raise RuntimeError("No matches between {} and {}".format(cat1.name,
                                                                 cat2.name))
    
    uniq = np.unique(indices[:, 1], return_counts=True)[1]
    indices = indices[np.where(uniq == 1)]

    duplicates = [uniq > 1]
    logger.debug("Removed {} duplicate matches between {} and {}".format(
                  len(duplicates), cat1.name, cat2.name))

    # indices[:, 0] --> cat1 indices
    # indices[:, 1] --> cat2 indices
    return indices 
    

def write_out(cat1, cat2, indices, outname, nmax=100):
    """Write out a catalogue of `cat2` coordinates with `cat1` appended.

    """

    if cat1.flux is not None and len(indices[:, 0]) > nmax:
        threshold = round(np.sort(cat1.table[cat1.flux][indices[:, 0]])[::-1][nmax-1], 1)
        cut = cat1.table[cat1.flux][indices[:, 0]] > threshold
    else:
        cut = slice(0, len(indices[:, 0]))

            
    columns_to_add = [fits.Column(name=column, format=cat2.table.columns[column].format,
                                  array=cat2.table[column][indices[:, 1]][cut])
                      for column in cat2.table.columns.names
                      ]


    columns_to_add += [fits.Column(name="old_ra", format="E",
                                   array=cat1.table[cat1.ra_key][indices[:, 0]][cut]),
                       fits.Column(name="old_dec", format="E",
                                   array=cat1.table[cat1.dec_key][indices[:, 0]][cut])
                       ]

    if cat1.flux is not None:
        columns_to_add += [fits.Column(name="flux", format="E",
                                       array=cat1.table[cat1.flux][indices[:, 0]][cut])]
    if cat1.eflux is not None:
        columns_to_add += [fits.Column(name="eflux", format="E",
                                       array=cat1.table[cat1.eflux][indices[:, 0]][cut])]

    if hasattr(cat1, "local_rms"):
        columns_to_add += [fits.Column(name="local_rms", format="E",
                                       array=cat1.table[cat1.local_rms][indices[:, 0]][cut])]


    if not outname.endswith(".fits"):
        outname += ".fits"

    hdu = fits.BinTableHDU.from_columns(columns_to_add)
    hdu.writeto(outname, overwrite=True)


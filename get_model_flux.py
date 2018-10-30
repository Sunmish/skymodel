#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from argparse import ArgumentParser

# Use aocal parser from:
from peel_suggester import parse_ao, parse_metafits


def total_flux(aocal, freq, alpha=-0.7):
	"""Get total flux from aocal file. 

	Assume single source, and calculate the flux at a given frequency for each
	 component before summing component fluxes.

	Parameters
	----------
	aocal : str
		Skymodel format 1.0/1.1 file with source parameters.
	freq : float
		Frequency in MHz at which to estimate flux density.
	alpha : float, optional
		Spectral index used if only one flux density measurement, assuming
		a simple power law model.

	Returns
	float
		Total flux density of all components/sources in `aocal at `freq`.

	"""

	sources = parse_ao(aocal)

	tflux = 0
	for source in sources:
		# First calculate the flux density at the given frequency:
		source.at_freq(freq=freq*1.e6,
                   	   components=range(source.ncomponents),
                       alpha=alpha)

		sflux = 0
		for c in source.components:
			sflux += c.at_flux

		tflux += sflux

	return tflux



if __name__ == "__main__":

	parser = ArgumentParser(description="Get total flux density from a skymodel.")
	parser.add_argument("-a", "--aocal", "--aofile", dest="aocal", default=None, 
						help="Skymodel format 1.0/1.1 file.")
	parser.add_argument("-f", "--freq", dest="freq", default=None, 
						help="Frequency in MHz at which to estimate flux density")
	parser.add_argument("-m", "--metafits", dest="metafits", default=None, 
						help="MWA metafits file to read frequency from.")
	parser.add_argument("-s", "--alpha", "--si", dest="alpha", default=-0.7,
					    help="Spectral index used to estimate flux if only "
					    "one flux density measurement available for a component.",
					    type=float)

	options = parser.parse_args()

	if options.aocal is None:
		raise ValueError("Skymodel format 1.0/1.1 file required.")
	elif (options.freq is None) and (options.metafits is None):
		raise ValueError("Frequency information needed.")

	if options.metafits is not None:
		print("Getting frequency from metafits file: {}".format(options.metafits))
		_, _, freq = parse_metafits(options.metafits)
		freq /= 1.e6
	else:
		freq = options.freq

	print("Estimating flux density at {} MHz".format(freq))

	tflux = total_flux(aocal=options.aocal,
					   freq=freq,
					   alpha=options.alpha)

	print("Total flux = {} Jy".format(tflux))




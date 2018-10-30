#! /usr/bin/env python

from argparse import ArgumentParser
import logging
logging.basicConfig(format="%(levelname)s (%(module)s): %(message)s",
                    level=logging.INFO)


def main():

	parser = ArgumentParser(description="Get total flux density from a skymodel, "
										"optionally attenuating by the primary "
										"beam response for a given observation.")
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
	parser.add_argument("-A", "--attenuate", dest="atten", action="store_true", 
						 help="Switch True if wanting to attenuate by the primary "
						 	  "beam response. To do this, a metafits file must "
						 	  "be specified.")

	args = parser.parse_args()

	if args.aocal is None:
		raise ValueError("Skymodel format 1.0/1.1 file required.")
	if (args.metafits is None) and (args.freq is None):
		raise ValueError("Frequency information needed.")

	print("Estimating flux density at {} MHz".format(freq))

	tflux = total_flux(aocal=args.aocal,
					   freq=args.freq,
					   alpha=args.alpha,
					   metafits=args.metafits,
					   attenuate=args.atten)

	print("Total flux = {} Jy".format(tflux))



if __name__ == "__main__":
	main()

	




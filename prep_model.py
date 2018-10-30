#! /usr/bin/env python

from __future__ import print_function

import numpy as np
import sys
import os
import argparse

import get_model_flux



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", dest="dir", default="./")
    parser.add_argument("-m", "--metafits", dest="metafits", default=None, 
                        help="MWA metafits file to read frequency from.")
    parser.add_argument("-t", "--threshold", dest="threshold", default=2.)

    options = parser.parse_args()

    if options.metafits is None:
        raise ValueError("A metafits file must be supplied.")


    all_models = open("all_models.txt", "w+")
    all_models.write("skymodel fileformat 1.1\n")


    files = os.listdir(options.dir)

    files_to_use = ""
    total_fluxes = ""

    for spec in files:
        if spec.endswith(".txt") and spec.startswith("model-"):

            tflux = get_model_flux.total_flux(options.dir+"/"+spec, 
                                              attenuate=True, 
                                              metafits=options.metafits)
            
            if tflux > options.threshold:

                with open(options.dir+"/"+spec, "r+") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "skymodel" in line:
                            pass
                        elif "source" in line:
                            all_models.write("\n"+line.lstrip())
                        elif "fluxdensity" in line:
                            all_models.write("fluxdensity Jy {} 0 0 0\n".format(
                                line.split()[2]))
                        else:
                            all_models.write(line.lstrip())

                files_to_use += "{}\n".format(spec)

            total_fluxes += "{}\n".format(tflux)

            print("{}: {}".format(spec, tflux))

    print(files_to_use)
    print(total_fluxes)


if __name__ == "__main__":
    main()
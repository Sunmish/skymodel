# skymodel
A collection of scripts to read and deal with models for AO calibration tools. These are mostly used for MWA data-processing.

## Requirements
Besides the standard `python` setup for astronomy, along with `astropy`, `numpy`, etc., this also requires the old `mwapy` package, however the new version `mwa_py` ([hosted here](https://github.com/MWATelescope/mwa_pb)) will work fine once the `mwapy.pb` references have been changed to `mwa_pb`.  
## References
1. A piece of code written by Natasha Hurley-Walker to access the beam value at a given RA and Dec is adapted for use here.

## Sumarised by
1. `create_skymodel` Create a skymodel using the GLEAM catalogue.
2. `get_model_flux` Get total flux at a given frequency within a model file.
3. `prep_model` Prepare a combined model file.
4. `peel_suggester` Given a set of models and an observation, suggest to peel or not. NO CONSIDERATION FOR BRIGHT, RUNNER-UP SOURCES.


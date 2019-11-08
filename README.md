# skymodel
A collection of scripts to read and deal with sky models for AO calibration tools. These are mostly used for MWA data-processing. Additionally, tools to play with the MWA primary beam are packaged here as they are required for creating good sky models. This code is used extensively within the MWA Phase II Pipeline ([`piip`](https://gitlab.com/Sunmish/piip/)).

## Requirements
Besides the standard `python` setup for astronomy, along with `astropy`, `numpy`, etc., this also requires `mwa_py` ([hosted here](https://github.com/MWATelescope/mwa_pb)) for handling the MWA primary beam, and `flux_warp` ([here](https://gitlab.com/Sunmish/flux_warp)) for source models and for a nice and shiny buildt-in all-sky model based on the GLEAM EGC (Hurley-Walker et al. 2017).  

## Sumarised by
1. `create_model_image` Create an image from an AO model convolved with a user-defined beam.
2. `create_skymodel` Create a observation-specific sky model using the GLEAM (or other) all-sky catalogue.
3. `get_beam_image` Create an image of the Stokes I primary beam for the MWA for a given observation.
4. `get_beam_lobes` Get the main and sidelobes of the MWA Stokes I primary beam.
5. `get_model_flux` Get total flux at a given frequency within an AO-model file.
6. `model2reg` Convert an AO-model to a DS9 region file.
7. `obs2reg` Convert a set of MWA observations to a DS9 region file showing the FoV.
8. `peel_suggester` Given a set of AO-models and an observation, suggest to peel or not. (No consideration for bright, runner-up sources.)
9. `prep_model` Prepare an AO-model file by combining a collection of individual AO-model files.

## Installing
After cloning the repository try the following (to install into a standard location):
```
git clone https://github.com/Sunmish/skymodel.git
cd skymodel
python setup.py install
```

## References
1. Hurley-Walker N., et al., 2017, MNRAS, 464, 1146 ([10.1093/mnras/stw2337](https://doi.org/10.1093/mnras/stw2337))
2. Sokolowski M., et al., 2017, PASA, 34, e062 ([10.1017/pasa.2017.54](https://doi.org/10.1017/pasa.2017.54))

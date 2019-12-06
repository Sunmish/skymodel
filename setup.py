#! /usr/bin/env python

import setuptools
import sys

with open("README.md", "r") as f:
    long_description = f.read()

if sys.version_info[0] == 3:
    _version = ">=3.0.0"
elif sys.version_info[0] == 2:
    _version = "<3.0.0"
else:
    _version = ""

reqs = [
    "astropy"+_version,
    "numpy",
    "scipy",
    "psutil"
]

scripts = [
    "scripts/create_model_image",
    "scripts/create_skymodel",
    "scripts/get_beam_image",
    "scripts/get_beam_lobes",
    "scripts/get_model_flux",
    "scripts/model2reg",
    "scripts/obs2reg",
    "scripts/peel_suggester",
    "scripts/prep_model",
]


setuptools.setup(
    name="skymodel",
    version="0.0.0",
    author="Stefan W Duchesne",
    author_email="stefanduchesne@gmail.com",
    description="A small package to create sky models for MWA calibration.",
    long_description=long_description,
    url="https://github.com/Sunmish/skymodel",
    install_requires=reqs,
    packages=["skymodel"],
    scripts=scripts,
)


A collection of scripts to read and deal with models for AO calibration tools.

create_skymodel.py:
```
usage: create_skymodel.py [-h] [-g CATALOGUE] [-m METAFITS] [-o OUTNAME]
                          [-t THRESHOLD] [-r RADIUS] [-R RATIO] [-n NMAX]
                          [--plot] [-x [EXCLUDE [EXCLUDE ...]]]

Create sky model from GLEAM.

optional arguments:
  -h, --help            show this help message and exit
  -g CATALOGUE, --catalogue CATALOGUE, --gleam CATALOGUE
                        Input GLEAM catalogue location.
  -m METAFITS, --metafits METAFITS
                        Name/location of metafits file for observation.
  -o OUTNAME, --outname OUTNAME
                        Output skymodel name.
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold below which to cut sources [1 Jy].
  -r RADIUS, --radius RADIUS
                        Radius within which to select sources [120 deg].
  -R RATIO, --ratio RATIO
                        Ratio of source size to beam shape to determine if
                        point source [1.1].
  -n NMAX, --nmax NMAX  Max number of sources to return. The threshold is
                        recalculated if more sources than nmax are found above
                        it.
  --plot
  -x [EXCLUDE [EXCLUDE ...]], --exclude_model [EXCLUDE [EXCLUDE ...]]
                        Skymodel v1.1 format file with existing models. These
                        will be create an exclusion zones of 10 arcmin around
                        these sources.
```

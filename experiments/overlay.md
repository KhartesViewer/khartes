# Viewing color overlays in khartes

This document describes a method of overlaying colored
information (indicators or detected ink) on top of the
scroll data viewed in khartes.

This method, which relies on special scripts and
a hacky data format, is intended to be temporary, and should eventually
be replaced by an overlay method that is entirely
inside of khartes.

## Overview

At the bottom of this file, I describe the data format,
in case you want to write a script to convert your own
data plus overlay into a form that can be viewed in khartes.

However, if you already have scroll data and indicator data
in a particular format (3D TIFF files), you can use the scripts
that I describe in the next section instead of writing your own.

## Creating an overlay file using existing scripts

The process of creating a file with overlays, using
the existing scripts, has several steps:

0) It is assumed that you alread have an indicator 
file (uint8 or uint16,
single channel) in 3D TIFF format.

1) Use the `ind_to_tif.py` script to create a uint8 RGBA file
(in 3D TIFF format)
from the indicator file.  This script gives each indicator
a color, according to a specified color map.

2) Use the `3dtif_to_nrrd.py` program to overlay the data
in the RGBA file from step 1 onto the scroll data, which
 assumed to be provided in a 3D TIFF file.
It is assumed
that the grids of the two files are aligned.  As its
name suggests, this script creates an NRRD file with
the headings that khartes expects.  Note that this NRRD
file contains uint16 data, but is encoded in a format that
can only be made sense of by the latest version of khartes.

3) Load the NRRD file into khartes, and view it.

## `ind_to_tif.py`

The `ind_to_tif.py` script reads an indicator file
(a file with an indicator number at each pixel),
and creates an RGBA (color) file, with the
indicator-to-color conversion based on a color map.

The script
takes two mandatory arguments:

1) The name of the indicator file.
It is assumed that the file contains indicators
in single-channel uint8 or uint16, in 3D TIFF format, 
and that a pixel value of `0` means that no indicator is present in
that pixel.

2) The name of the output file.  This will be a 
a 3D TIFF file, with each pixel represented by
4 channels (RGBA), each channel being a uint8.

There are also two optional arguments:

1) `--colormap`, which specifies the name of the colormap
used to convert indicator numbers to colors.  The
available colormaps are shown (and illustrated)
at https://cmap-docs.readthedocs.io/en/stable/catalog/ .
The default colormap is `bmr_3c`.

2) `--alpha`, which specifies the alpha (transparency)
to be stored in the A channel of the RGBA data.  The
allowed ranges are 0.0 (fully transparent) to
1.0 (fully opaque).  The default alpha value is 1.0
(opaque).

## `3d_tif_to_nrrd.py`

The `3d_tif_to_nrrd.py` script reads a 3D TIFF file and
creates a NRRD file containing the headers that
khartes expects.

Optionally, it also can read the 3D TIFF RGBA file
created by `ind_to_tif.py`, and overlay the colors
in that file on top of the gray-scale scroll data.
The RGB channels in this file specify the overly color, 
and the A channel specifies the alpha (opacity) of the overlay color.

In order to store colors in the output gray-scale
file (which is a single-channel
uint16 khartes-compatible NRRD file), color information
is encoded in a special format that the latest version
of khartes knows how to interpret.

`3d_tif_to_nrrd.py` takes two mandatory arguments:

1) The name of the 3D TIFF file that contains the
scroll data, in uint8 or uint16 format.

2) The name of the NRRD file that will be created,
which will contain the scroll data and, optionally,
a color overlay of the indicator data.

There are also two optional arguments:

1) `--overlay`, which specifies the name of the overlay
file (previously created by `ind_to_tif.py`), which contains
uint8 RGBA data in 3D TIFF form.  It is assumed that the
3D grid of the overlay file has exactly the same
size, and is in exactly the same xyz position, as the 
3D grid of the scroll data file.

2) `--alpha`, which specifies an alpha value
that will be applied to the overlay data, overriding the
value in the A channel of the RGBA file specified
by the `--overlay` argument.  Alpha can range from
0.0 (transparent) to 1.0 (opaque).  The default
value is `None`, which means that the alpha value
will be taken from the A channel of the overlay
file.

## `khartes.py`

Once you have created the NRRD file, you can load it
into the latest version of khartes, and (if all goes well)
see the indicator colors overlaid on the scroll data.

## Creating a data-plus-overlay file using your own script

In this section, I describe the format of the data, readable by khartes,
that describes an RGB image.


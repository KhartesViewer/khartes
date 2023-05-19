# khartes

Khartes (from χάρτης, an ancient Greek word for scroll) is a program
that allows users to interactively explore, and then segment, 
the data volumes created by high-resolution X-ray tomography of the Herculaneum scrolls.

Khartes is written in Python; it uses PyQt5 for the user interface, numpy and scikit for efficient computations,
pynrrd to read and write NRRD files (a data format for volume files), and OpenCV for graphics operations.

The main emphasis of khartes is on interactivity and a user-friendly GUI; no computer-vision or machine-learning
algorithms are currently in use.

The current version is really an alpha-test version; it is being provided to get some early user feedback.
One of the main things lacking is a function to export segments in a format readable by the 
volume-cartographer series of programs.

The only documentation at this point is the video below.  Note that it begins with an "artistic" 60-second
intro sequence which contains no narration, but which quickly highlights some of khartes' features.
After the intro, the video follows a more traditional format, with a voiceover and a demo.
The entire video
is about 30 minutes long.

(If you click on the image below, you will be taken to vimeo.com to watch the video)

[![Watch the video](https://i.vimeocdn.com/video/1670955201-81a75343b71db9c84b6b4275e3447c943d2128ab8b921a822051046e83db0c96-d_640)](https://vimeo.com/827515595)

## Installation

In theory, you should be able to run simply by
cloning the repository, making sure you have the proper dependencies 
(see "anaconda_installation.txt" for a list), and then typing "python khartes.py".  

When khartes starts, you will see some explanatory text on the right-hand side of the interface 
to help you get started.  This text is fairly limited; you might want to watch the video above to get a better
idea how to proceed.

A couple of notes based on early user testing (you might
want to review these again after using khartes for the first
time):

The "File / Import TIFF Files..." creates a khartes data volume
by reading TIFF files that you already have somewhere on disk.
You simply need to point the import-TIFF dialog to the folder
that contains these files.

The import-TIFF function uses more memory than it should 
(it unnecessarily duplicates the data volume in memory during
the import process).  This means that at the current time you
should be sparing of memory, creating data volumes that are no
larger than half the size of your physical memory,
if you want to avoid "memory swapping".

When you create fragments, pay attention to the triangulation
that is shown in the fragment window on the right.  Khartes'
interpolation algorithm can become erratic in areas of long,
skinny triangles, so it is a good idea to distribute enough
fragment nodes throughout the fragment, to keep the triangles
more regular.  

So when segmenting, start in the center of a fragment
and work your way out, keeping a fairly regular mesh, instead
of trying to create a huge surface first thing.  This practice
will also make it less likely that you stray onto the wrong
sheet of the scroll in difficult areas.

## Major limitation

At the moment, the fragments created in khartes are not
exportable to volume-cartographer.  The intention is to make
them exportable, but the details need to be worked out.

In the meantime, the fragments you create now in khartes can be
saved now, and viewed in khartes, and the plan is that you will
be able to export them once the export functionality is
added.

## Other things to fix

At the moment, khartes does not warn the user if there is 
unsaved data, when the user exits khartes
or reads another project.

There is no way for the user to delete nodes (my usual practice
at the moment is to move them out of the way to somewhere harmless).

There is no undo function.

Memory usage during import-TIFFs (and perhaps other operations)
needs to be optimized, to allow bigger data volumes.

Allow the user to change fragment and volume names.

Allow the user to change display settings such as node size and
crosshair thickness.

The scale bar is based on a voxel spacing of 7.9 um; allow the user to 
change this.

(Many others too uninteresting to list here)

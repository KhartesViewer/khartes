# khartes

Khartes (from χάρτης, an ancient Greek word for scroll) is a program
that allows users to interactively explore, and then segment, 
the data volumes created by high-resolution X-ray tomography of the Herculaneum scrolls.

Khartes is written in Python; it uses PyQt5 for the user interface, numpy and scikit for efficient computations,
pynrrd to read and write NRRD files (a data format for volume files), and OpenCV for graphics operations.

The current version is really an alpha-test version; it is being provided to get some early user feedback.
One of the main things lacking is a function to export segments in a format readable by the 
volume-cartographer series of programs.

The only documentation at this point is the short video below.  Note that it begins with an "artistic" 45-second
intro sequence that contains no narration, but that quickly demos some of khartes' features.
After the intro, the video follows a more traditional format.
The entire video
is about 30 minutes long.

## Installation

In theory, you should be able to run simply by
cloning the repository, making sure you have the proper dependencies 
(see "anaconda installation.txt" for a list), and then typing "python khartes.py".  

When khartes starts, you will see some explanatory text on the right-hand side of the interface 
to help you get started.  It is fairly limited; you might want to watch the video above to get a better
idea how to proceed.

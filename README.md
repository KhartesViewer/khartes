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

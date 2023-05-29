# khartes

Khartes (from χάρτης, an ancient Greek word for scroll) is a program
that allows users to interactively explore, and then segment, 
the data volumes created by high-resolution X-ray tomography of the Herculaneum scrolls.

Khartes is written in Python; it uses PyQt5 for the user interface, numpy and scikit for efficient computations,
pynrrd to read and write NRRD files (a data format for volume files), and OpenCV for graphics operations.

The main emphasis of khartes is on interactivity and a user-friendly GUI; no computer-vision or machine-learning
algorithms are currently used.

The current version is really an alpha-test version; it is being provided to get some early user feedback.

The only documentation at this point is the video below.  Note that it begins with an "artistic" 60-second
intro sequence which contains no narration, but which quickly highlights some of khartes' features.
After the intro, the video follows a more traditional format, with a voiceover and a demo.
The entire video
is about 30 minutes long.  There is no closed captioning, but the script for the video can
be found in the file demo1_script.txt.

(If you click on the image below, you will be taken to vimeo.com to watch the video)

[![Watch the video](https://i.vimeocdn.com/video/1670955201-81a75343b71db9c84b6b4275e3447c943d2128ab8b921a822051046e83db0c96-d_640)](https://vimeo.com/827515595)

## Vacation announcement

I will be unavailable to work on khartes from the end of May until the
end of June.  I will try to monitor the Vesuvius Scrolls Discord server,
but I cannot guarantee that I will be able to fix bugs or answer questions
during that time.  But my availability in July looks good!

## Installation

In theory, you should be able to run simply by
cloning the repository, making sure you have the proper dependencies 
(see "anaconda_installation.txt" for a list), and then typing `python khartes.py`.  

When khartes starts, you will see some explanatory text on the right-hand side of the interface 
to help you get started.  This text is fairly limited; you might want to watch the video above to get a better
idea how to proceed, or read the "General Workflow" section below.

A couple of notes based on early user testing (**you might
want to review these again** after using khartes for the first
time):

The `File / Import TIFF Files...` menu option
creates a khartes data volume
by reading TIFF files that you already have somewhere on disk.
You simply need to point the import-TIFF dialog to the folder
that contains these files.

The import-TIFF function uses more memory than it should 
(it unnecessarily duplicates the data volume in memory during
the import process).  This means that at the current time you
should be sparing of memory, creating data volumes that are no
larger than half the size of your physical memory,
if you want to avoid memory swapping.

## General workflow

As the programmer, I know how khartes works internally.  However,
I only have a few hours experience as a user of the software.
The advice that follows is based on this experience, but 
these suggestions
should not be treated as something engraved in stone, more
like something written on papyrus in water-based ink.

**Step 0**: The most important step, which you
must take even before you start segmenting,
is to decide in which area to work.  For your first
attempt, you should start with a sheet that is clearly separated
from its neighbors; no need to 
make your learning experience difficult.

<img src="https://github.com/KhartesViewer/khartes/assets/133787404/536c9c8f-b0f7-4079-92d1-b14711a0b7bb" width="360"/>

*This is a an example* of a fairly easy sheet.

For your next attempt, you might want to start with a sheet
that is separated on one side from its neighbors.

Keep in mind that after you
have created a fragment for one sheet, you can view that fragment
even while working on the next sheet, 
using it as a kind of guide.
So one strategy is to work on a series of sheets that are 
parallel to
each other, starting with the easiest.

There are some areas in the scroll data volume that I found to be
too difficult.  In these areas, fragments appear, disappear, and
merge into each other in a way that seems impossible to track, no
matter what software is used.  If you try working in these areas,
prepare to be frustrated. 

![squirmy](https://github.com/KhartesViewer/khartes/assets/133787404/ffa05425-d218-410e-94be-351c4367cfbe)

*This area* is very difficult or impossible to segment; the sheets
are too fragmented.

Bear in mind that khartes works only with single-valued surfaces;
it cannot handle sheets that turn over onto themselves.
If you want to work on such a sheet, turn your volume
on its side (in the lower right corner, find the Volume panel, and
change the direction of your volume from "Y" to "X").

**Step 1:** Start in an easy area of your sheet, picking some points
on the inline (top window) and crossline (middle window) slices.
This will create a diamond-shaped area in the fragment viewer
(right-hand window).  Make sure you are happy with what you see
before expanding.

**Step 2:** Expand by alternating directions.  Use the fragment viewer
to move to a new area on the fragment and create nodes on the inline
slice.  Then create nodes on the crossline slice.  You can also add
nodes to the bottom slice; these act like contour lines on a map.

**Hint for step 2:** Before you start adding new nodes onto the line,
look in the fragment viewer to see if there are any existing nodes near
that line.  IF there are, and it is feasible, move these existing nodes
onto the line.  This is to avoid the situation where a node on the line
and a node just off of the line end up close to each other, which can
cause undesirable waviness in the fragment.

<img src="https://github.com/KhartesViewer/khartes/assets/133787404/0a355d1b-25cd-4bf7-87ec-144492900d06" width="800" />

*Example of a good start.*  A couple of inline slices (the horizontal lines
in the fragment view) and a crossline slice (vertical line) have been 
interpreted.  Nodes near the lines have been moved onto the lines, to
maintain good node spacing.  
Some "contour" points have been added to the bottom slice as well.
The horizontal fibers are continuous, 
which is important (see Step 3).  The dark spot in the upper right quadrant is 
due to a lack of data to constrain the interpolation; as more nodes are
added, the spot will be replaced by the image of the sheet.

**Step 3**: Pause, verify, repair.  The most important criterion for
a good fragment is that the horizontal fibers (as seen in the fragment view)
are continuous, since the horizontal fibers (also called the circumferential fibers)
are the ones that are most likely to contain the ink.  
Where horizontal and vertical fibers cross, try
to make sure that the horizontal fibers are the ones that are the most visible

![sheet_skip](https://github.com/KhartesViewer/khartes/assets/133787404/62d4b800-9731-4310-8ecf-01ddca1e6aa5)

***This is bad!***  The horizontal fibers are not continuous.  This needs to be repaired by
moving some of the nodes so that they all lie on the same sheet.

**Step 3 continued**  The main problem to watch out for, as illustrated above,
is what I call "sheet skipping": because two adjacent sheets are close together, or
even merge in some areas, the user has unintentionally started adding nodes onto
the wrong sheet.  As a result, the fibers on the left side of this picture are from
a different sheet than the fibers on the right.  This creates a
visual discontinuity, which is a signal that the user needs
to go back, analyze the existing nodes, and move as many as necessary 
until all are on the correct sheet.
So again: pause, verify, repair.  The longer you wait to do this basic check, the
more repair work you will have to do later.

<img src="https://github.com/KhartesViewer/khartes/assets/133787404/2f685bc9-bf55-4d1a-9c54-f472e3c0dc4b" width="800" />

*The surface has been repaired;* horizontal fibers are now continuous.  The inline and crossline slices show
the location of the original (magenta) and repaired (cyan) surfaces.  Note that these overlap on the right,
but diverge on the left.

## Workflow notes

**Save often.**  You can simply type Ctrl-S to save your work; try to remember to do this whenever
a dozen or so nodes have been added or changed.  The "save" operation is very quick, since only 
fragments are saved; the volume data does not change and thus is not part of the operation.

When you create fragments, pay attention to the triangulation
that is shown in the fragment window on the right.  Khartes'
interpolation algorithm can become erratic in areas of long,
skinny triangles, so it is a good idea to distribute enough
fragment nodes throughout the fragment, to keep the triangles
more regular.  

Another reason for monitoring the shapes of your triangles is to
improve speed of interaction.  Every time a fragment node is moved
or added, khartes updates the fragment window to reflect these changes.
This means that triangles near the modified node, and the pixels
that these triangles encompass, need to be recomputed
and redrawn.  The bigger the triangles, the longer the recomputations
take.  If a node is surrounded by large triangles that cover most
of the data volume, each change may require several seconds to recompute,
meaning that khartes no longer feels interactive.  You can prevent this problem
by keep your triangles regular and local.

So when segmenting, start in the center of a fragment
and work your way out, keeping a fairly regular mesh, instead
of trying to create a huge surface first thing.  This practice
will also make it less likely that you stray onto the wrong
sheet of the scroll in difficult areas.

Remember that khartes does not have auto-save; use Ctrl-S on
a regular basis to save your latest work.

## Exporting fragments

Khartes allows you to export your fragments to `vc_render` and `vc_layers_from_ppm`.

To export your fragment:

1. Make sure your fragment is active, that is, that it is visible
in the right-hand window.
2. In the File menu, select `Export file as mesh...`.

This will create a .obj file, which contains a mesh representing your
fragment.

You can import this mesh directly into `vc_render`.  Here is how.

First, you need to make sure you know where the following files and
directories are located:

- Your .volpkg folder, the one that contains the TIFF files that you
imported into khartes
- If your .volpkg directory contains more than one volume, you need
to know the number of the volume that contains the TIFF files
that you used.
- The .obj mesh file that you just created
- The directory where you want to create a .ppm file, and the name
that you want to give the .ppm file.  The .ppm file is needed by
`vc_layers_from_ppm`.

So the command you want to type will look something like:
```
vc_render -v [your volpkg directory] --input-mesh [your .obj file] --output-ppm [the name of the ppm file you want to create]
```
You might need to use --volume to specify your volume as well, if your volpkg has more than one.

As already mentioned, the .ppm file that `vc_render` creates can be used in `vc_layers_from_ppm` to create a 
flattened surface volume.


## Things to fix

When the user exits khartes
or reads another project, khartes does not warn the
user if there is unsaved data.

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

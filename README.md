# khartes

Khartes (from χάρτης, an ancient Greek word for scroll) is a program
that allows users to interactively explore, and then segment, 
the data volumes created by high-resolution X-ray tomography of the Herculaneum scrolls.

Khartes is written in Python; it uses PyQt5 for the user interface, numpy and scipy for efficient computations,
pynrrd to read and write NRRD files (a data format for volume files), and OpenCV for graphics operations.

The main emphasis of khartes is on interactivity and a user-friendly GUI; no computer-vision or machine-learning
algorithms are currently used.

## 2-minute video

The video below give a quick, 2-minute overview of khartes.

(If you click on the image below, you will be taken to vimeo.com to watch the video)

[![Watch the video](https://i.vimeocdn.com/video/1703962064-7e3db7142e72fe9b85887c02bd8902a65a1a6ef9c644e3e12bb5485b3519ece3-d_640)](https://vimeo.com/849799186)

## 30-minute video

This next video provides a more extensive introduction to khartes.
Note that it begins with an "artistic" 60-second
intro sequence which contains no narration, but which quickly highlights some of khartes' features.
After the intro, the video follows a more traditional format, with a voiceover and a demo.
The entire video
is about 30 minutes long.  There is no closed captioning, but the script for the video can
be found in the file demo1_script.txt.

(If you click on the image below, you will be taken to vimeo.com to watch the video)

[![Watch the video](https://i.vimeocdn.com/video/1670955201-81a75343b71db9c84b6b4275e3447c943d2128ab8b921a822051046e83db0c96-d_640)](https://vimeo.com/827515595)

# Installation

You should be able to run simply by
cloning the repository, making sure you have the proper dependencies 
(see "anaconda_installation.txt" for a list), and then typing `python khartes.py`.  

When khartes starts, you will see some explanatory text on the right-hand side of the interface 
to help you get started.  This text is fairly limited; you might want to watch the video above to get a better
idea how to proceed, and read the Manual/Tutorial below.

# Manual/Tutorial

Ideally, khartes should come with both a user manual and a tutorial (and perhaps even a "cookbook"),
but at this point there exists only a single document, a tutorial that also tries
to act as a user manual of sorts.

This tutorial covers the following steps:
* Creating a project
* Creating a data volume from TIFF files
* Exploring the user interface and the data volume
* Creating a fragment ("segmentation")
* General workflow for creating coherent, consistent fragments
* Exporting the fragment as a mesh
* Working with multiple fragments

## Creating a project

If you are running khartes for the first time, or
if you are starting a new project, select the `File / New Project...` menu item
to create a new project.  When you create a new project, you will
immediately be asked for the name of your project, and the location
on disk where you wish it to be created.  This is so that khartes
can begin to store data there, 
which it does even before the first time you invoke "Save".

## Converting TIFF files to khartes data volumes

In order to perform at interactive speeds, khartes works with data
volumes (3D volumes of data) rather than individual TIFF files.
Khartes provides an interface to convert a set of TIFF files into
a data volume.
This conversion process
assumes that the TIFF files are already somewhere on your disk;
khartes does not download TIFF files over the internet.

The `File / Create volume from TIFF files...` menu item brings up a dialog box where
you can specify which TIFF files you want to include in your
data volume, and the x and y ranges of the pixels within each TIFF
file that you want to include.  A status bar at the bottom of
the dialog box shows how large the
resulting data volume will be, in gigabytes.  **Be aware that
every time you run khartes, the entire
data volume will be read into the physical memory (RAM) of your
computer, so be careful how large you make the volume.**

The first step, of course, is to find your TIFF files.  Use
the file selector in the TIFF loader dialog box to find
the *directory* that contains all the TIFF files.  Once
you double-click on the *directory* name, the TIFF loader
will read the TIFF files, determine how many
there are and their x and y extent (width and height).

Be aware that the TIFF loader assumes that the TIFF files are
consecutively numbered (no gaps), and that they all have the
same width and height.

The TIFF loader will display the maximum possible range of the
volume that you can create from the TIFF files, which in most
cases would vastly exceed the capacity of your computer.

By adjusting these numbers in the TIFF loader dialog box,
you specify the `Min` (minimum) 
and `Max` (maximum) x, y and z (TIFF file)
values for the data volume you want to create.

You can also specify a `Step`, which is the step
size (distance) between the pixels from the original
image that will be incorporated into the data volume.  One way to
look at `Step` is as a decimation factor.  A step
of 1 means use every pixel, 2 means use every 2nd pixel,
and so on.  Choosing a higher step size reduces the
size of the data volume over a given range,
but the loss of resolution, even going from a step size
of 1 to 2, is quite noticeable when you are trying to
segment the data.

The main use of `Step` is to allow an
overview of the data, in order to determine which areas will
merit a closer examination later, at a step size of 1.  For instance,
if you set a step size of 10 in all 3 directions (x, y, and z),
you can reduce an entire scroll, typically 1.5 Tb or so,
to a very manageable 1.5 Gb.  This resolution of this volume
is much too low for segmentation, but it is good enough to give you
an idea of where the interesting areas of the scroll are.

**As the first step of this tutorial**, create a data volume that
encompasses your entire set of TIFF files.  Use a `Step` of 10
in all 3 directions, in order to keep the data volume small.

Name this new volume `all10`, to remind yourself that the volume
encompasses all the data, but with a decimation factor of 10.
Set the color to something you like.

When you press the `Create` button, if you get a warning that your
data volume will be large, check to make sure that all 3 `Step`
values are set to 10.

This process will take some fraction of an hour, depending on
how fast your computer is, and how many TIFF files you have.
For instance, if you have a scroll with 14,000 TIFF files, and
a step size of 10, khartes will need to read 1,400 of these files.

The TIFF file loader has a status bar at the bottom that shows
the name of the file that is currently being read.

For future reference: if you already have an `all10` data volume,
typically stored in a file called `all10.nrrd`, you can use the
`File / Import NRRDs...` menu item to import this file, which
is quite a bit faster than reading all those TIFFs.

## Exploring the user interface

<img src="images/labelled_overview.jpg" width="800"/>

Now that you have loaded a data set, you can more easily explore the
user interface of khartes.

Referring to the figure above, the main areas where you will work are the 3 *Data Slices*
on the left side, and the *Fragment View* in the upper right.
The *Control Area* in the lower right has tabs that let you
make adjustments to volumes, fragments, and display parameters.
At the bottom is the *Status Bar*, which usually shows you the 3D coordinates
of the cursor.  And at the top is the *Menu*, which handles file import and export.

At the moment, the *Fragment View* is blank, because you have not created
any fragments.

Instead, focus on the 3 data slices.  These 3 windows represent 3 mutually
perpendicular slices throught the 3D data volume.  The 3 slices meet in the
middle of each window, where the crosshairs intersect.

### Staying oriented

There are a few cues to help you stay oriented.  

1) In the upper left of each
window is a label that gives the current position of the given slice.
For instance, the upper slice, which corresponds to one of the original TIFF files,
has a label in the upper left indicating which image (TIFF file) it is from.

2) Below the 3 data slices is the status bar.  This gives the current
3D location of the cursor.  For instance, if you move the cursor around inside of the
top data slice window, and watch the status bar, you will see that the IMG (image) coordinate
remains constant (and is the same as in the label of the data slice), while
the X and Y coordinates change.

3) Each of the 3 data slices has a colored border.  And inside each data slice, the
two intersecting crosshairs are different colors.  These colors act as an additional orientation
cue.  Think, for a moment, about the red data slice (the one with a red border).  Somewhere in 3D
space its plane intersects the plane of the green data slice.  These two planes are mutually
perpendicular, so the intersection is a line.  In the window of the red data slice, this
line of intersection is drawn in green; it is the green crosshair, to show that this is where
the green slice crosses the red slice.  Likewise, in the window of the green data slice, this
same intersection line is drawn in red, so that when you are looking at the green window,
you know where the red slice crosses it.

### Navigation by mouse

To navigate within the data volume, simply hold down the left mouse button while inside one
of the data slice windows, and drag the slice.  The other slices will change, to ensure that
the mutual intersection point of the three slices remains in the middle of the crosshairs
of all 3 slices.

You can use the mouse scroll wheel while in any of the data slice windows to zoom in or out.
The crosshairs are always the center of the zoom, since the mutual intersection point of the three
slice planes does not change during zooming.

### Navigation by keyboard

You can use the four arrow key to move the current slice (the one where the mouse cursor is
currently located).  As when navigating with the mouse, the other slices will change to ensure
that their mutual interesection point remains in the centers of all 3 pairs of crosshairs.

You can also use the page-up and page-down keys to increment or decrement the slice in the current
window.  To see this, watch the label in the upper left corner of the current data slice, and see
how this changes as you press page-up and page-down.

If your keyboard offers auto-repeat, you can hold down an arrow key, or a page-up/down key, and
watch as the current slice slowly moves in the indicated direction.

The `a`, `s`, `w`, and `m` keys behave the same as the arrow keys, and `e` and `c` behave the
same as the page-up/down keys.

### Getting a feel for 3D navigation

To fully explain the navigation, and the meanings of the 3 slices, would require many diagrams
that I haven't had time to draw yet.  So for now, get a feeling for how the navigation works by
dragging slices, observing the labels in the upper left of each slice, and observing the 3D
coordinates in the status bar.

Your goal is to understand the navigation well enough so that you can predict, when you drag
one of the slices around, how the other slices and their labels will behave.  Once you reach
that stage, you will be able to navigate within your 3D data volume with confidence.

## Creating a high-resolution data volume from TIFFs

<img src="images/tiff_loader.jpg" width="800"/>

If you have been following this tutorial exactly, you have created,
and have been navigating within, a data volume that
contains the entire scroll, but at very low resolution.

Now you will create a high-resolution data volume that covers a much
smaller range.

To begin, bring up the TIFF loader dialog by selecting the
`File / Create volume from TIFF files...` menu item.

The dialog box will still have the same settings as you entered
last time.  This means that you must do the following before proceeding:

**Important step**: Check the `Step` settings in the TIFF loader dialog.
These must all be set to 1, to ensure that you create a high-resolution
data volume.  If you had previously  set the `Step` values to something different, you
must set them all back to 1 now.

Move the TIFF loader box to the side, so you can see the data slices.
Now zoom out, so that you can see your entire data volume (as mentioned
before, this assumes
that you are following the tutorial exactly, so that this data volume is
a low-resolution version of the entire scroll).

There should be a box hugging the outside of the data volume, drawn in
the "Volume color" shown in the TIFF loader.

This colored box is interactive; you can drag its corners in order to specify the
volume of interest that you want to load from the TIFF files.  

Place your mouse on one of the corners of this box; the cursor should turn into a
diagonal two-way arrow.  Press and drag the mouse to adjust the position of the corner.
Notice that as you change the position of the corner, the corresponding values
change in the TIFF loader dialog.  Likewise, if you change the numbers in
the TIFF loader dialog, the box will move correspondingly.

If, as you adjust the box, it disappears from some of the data slices, you can
find it again by pressing the `Re-center view` button in the TIFF loader.
This will shift the mutual intersection point of the data slices so that they
all pass through the current center of the box.

As you adjust the box, keep an eye on the Gb size that is displayed at the bottom
of the TIFF loader; you want to make sure your data volume will fit in the RAM of
your computer.

Finally, give your volume a name and a color, and hit the `Create` button.
In general, the time the loader takes to run is proportional to the number
of TIFF files that need to be processed.

### Advanced topic: loading TIFF files from vc_layers

This sub-section is not part of the tutorial, but it belongs with the
current topic: loading TIFF files.

The program `vc_layers` outputs TIFF files that represent flattened
layers adjacent to a flattened segment.  These files have a different
orientation than the TIFF files usually read by khartes.  To alert
khartes to this difference, so that the data volume created from these
`vc_layer` TIFFs
is properly oriented, check the box labeled `TIFFs are from vc_layers`.

## Control Area: Volumes

<img src="images/volumes_tab.JPG" width="800"/>

Go to the Control Area (the area in the lower right corner of khartes) and select the `Volumes` tab.
Here you should see listed the two volumes you have created so far: all10, and your high-resolution volume.
The checkbox in the left column allows you to select which volume should be made visible.
The next two columns show the name and the color of each volume.  The name cannot be changed.
However, you can change the volume's color by clicking on the color box.
The volume's color is used when the volume's bounding box is drawn as an overlay
on the data slices.

The 'Ld' column lets you know whether each volume is currently loaded into memory.  By default,
only one volume at a time is present in memory, in order to conserve RAM.  A future version of
khartes may allow multiple volumes to be present in memory simultaneously, in order to
permit the user to switch more quickly between volumes.

The next column allows you to change the orientation of the volume.  By default, the topmost
data slice window displays each TIFF image in its original orientationm which in khartes is called the Y
orientation.  Using the selector
in the fifth column of the Volumes tab, you can change the orientation of the TIFF image, turning it on
its side, by selecting the X orientation.  Later on, you will see why this is useful (it has to do with
the fragments being limited to be single-valued surfaces).

Feel free to change the orientation from Y to X to see how the data volume changes appearance in the
data slice windows, but change it back to Y when you are done, since the rest of the tutorial assumes
that you are in the Y orientation.

The next nine columns display the range of the volume within the scroll; these numbers are not editable.

The final column displays the size of the data volume in Gbytes.

Above the list is the `Volume Boxes Visible` checkbox.  When this box is checked, the outlines
(bounding boxes) of all the volumes (except the currently-loaded volume) will be overlaid
on the data slices.

## Creating a fragment (segment)

Now you are ready to create your first fragment ("segment" and "fragment" are used
interchangeably in this tutorial, but khartes prefers "fragment").

In the Control Area, select the Fragments tab.  There you will find
the button `Start New Fragment`.  Press this button, and a new fragment will
appear in the list of fragments.

At this point you can begin adding points (called nodes) to define
the fragment.  To add a node, position the cursor in one of the
data slices, hold down the shift key (so the cursor turns into
crosshairs), and click with the left mouse button.

It would take a lot of words and pictures to describe the process of adding
nodes to create a fragment; better to watch this 2-minute video (this
is the "2-minute video" mentioned at the top of this README):

[https://vimeo.com/849799186](https://vimeo.com/849799186)

The video shows a slightly older version of khartes; for
instance, the cursor did not turn into crosshairs before a new
node was created, and re-centering the view on an existing node
was more awkward than it is now.

Also, the video does not mention the key presses and button
presses made by the user.

But here are some general user-interface aspects that you should
have gathered from the video.

Each node that the user creates is added to a surface (the fragment)
that is displayed in map view in the Fragment View.  Of course, if
the new nodes all form a line, only a line, rather than a surface,
is shown in the Fragment view.  Once the user begins creating
nodes in a different data slice, a surface can be created, and
this surface is shown in the Fragment View.

In each data slice, the fragment is drawn as a line; this represents
the intersection between the fragment and the data slice.  If any nodes
lie on the plane of the slice, they are shown as red circles.  When
the cursor passes near a node, it turns from red to cyan.

The color of the fragment line in the data slices, and the color of
the triangular mesh in the Fragment View, are the color of the
fragment, as shown in the Fragments tab in the Control Area (you
won't actually see the Fragments tab in the video, since in the video
only the Volumes tab is shown).

It is alomst time for you to begin creating your own fragment,
but first you need to learn just a little about an advanced topic.

This advanced topic would preferably be introduced later in the
tutorial, but you need to be alerted to it now, in case you
run into it by accident.

**Advanced topic that needs to be introduced now**:  Khartes has
two modes of user interaction in the data slices and the fragment
view: "normal" mode, and "add-node" mode.  In normal mode, the
cursor usually looks like a hand (unless it is near a node), and
to add a node, you need to hold down the shift key (so the cursor
turns into crosshairs) before pressing the left mouse button.

In "add-node" mode, things are reversed.  The cursor always
takes the form of crosshairs, and to add a node, you simply need to
click with the left mouse button.

There are two ways to toggle between the two modes.  One way is to
click on the button, to the right of the `Files` menu, that contains
a picture of crosshairs.  When this button is light, khartes is in
normal mode, and when it is dark, khartes is in add-node mode.

The other, more convenient, way to toggle between normal and add-node mode
is to press and then quickly release the shift button.  You can see which
mode you are in by seeing whether the cursor afterwards is a hand (normal mode)
or crosshairs (add-node mode).

This advanced topic needed to be introduced now, because you might, by accident, quickly
press and release the shift button, unintentionally toggling between modes.
Now that you understand the two modes, you won't be mystified if this happen.

## Choosing a region for segmentation

Before you start creating a fragment, you should make sure that there
is a region in your high-resolution data volume that is possible to segment.

What you are looking for is a region where the papyrus sheets, as
seen in the top data slice, are not too tilted (you want an angle
of less than 45 degrees or so).  This region doesn't have
to be very large, but if you can't
find even a small area, then you may
need to create another high-resolution data volume.

## Building and modifying the segment

At this point you are ready to build the segment.
As you saw in the video, you simply add nodes to the
data slices.  Notice that **the order you add nodes does not matter**;
if you feel your nodes are too far apart, simply add more.

But what do you do if you discover that a node is in the wrong
place?  You have a couple of options.

### Deleting a node

It is very easy to delete a node.  Simply move the cursor over
the node, so that the node turns cyan (if you are currently in
"add-node mode", the node won't change color; you need to switch to normal mode).
At this point, simply hit the "Delete" or "Backspace" key (either will work).
But bear this in mind: **There is currently no "undo" in khartes!**

### Moving a node

More often, you want to move a node instead of deleting it.  This is likewise
easy to do, assuming you are in "normal mode".  Again, move the cursor to the
node you are interested in moving; the node will turn cyan.  (Notice that
the node
will be drawn in cyan wherever it is visible;
this is often a handy way of keeping track of
which node is which in the different views).

Once the node has turned cyan, there are a couple of ways to move it.  

If you are
working in a data slice, you can now press the left mouse button,
hold it down, and drag the node to where you want it.  Be aware that
you cannot use the mouse to drag nodes in the fragment view (this is to
prevent accidents).

If you need more precision, you can use the arrow keys (or `a`, `s`, `w`, and `d`)
to move the node to where you want it.  This works in all views, including
the Fragment View.

If you are working in the Fragment View, you can also use the page-up and page-down
keys to move the node.  These keys will move the node perpendicular to the plane of
the fragment view (to see what this means, make sure your node is also visible in the
top left data slice, and see how the node moves in the data slice while you press
page-up or page-down in the Fragment View).

**The key capability of khartes is that as you move nodes, the data image in the
fragment view updates
interactively, so you can instantly judge the quality of your segmentation.**

### What the cursor is telling you

As you work, you have probably noticed that the cursor changes its shape,
depending on where it is in the slice.  As a rule, the shape
of the cursor tells you
what the left mouse button will do if you push it.

Here are the various shapes, and what they mean.  Someday I will attach
pictures of the different cursors, and not just words...

* **Hand**: This tells you that khartes is in "normal mode" (in contrast to
"add-node mode").  If you press the left mouse button and drag, the current slice
will move in synchrony with the mouse.

* **Arrow** (pointing upwards and slightly to the left): If you are in a data slice or
the fragment view, this tells you that you are in "normal mode", but more specifically,
you are near a node (you should see a cyan node nearby).  If you press the left mouse
button and drag, the node will move in synchrony with the mouse.

* **Semi-transparent hand**: You will only see this if the cursor is in the Fragment View and
near a node (the node will be cyan).  The hand turns semi-transparent so as not to obscure
the data surface.  If you press the left mouse button, you will be able to drag the data view,
but not the node.  However, as long as the node is cyan, you will be able to move it with the arrow
keys, the page-up and page-down keys, and the aswd/ec equivalents.

* **Crosshairs**: If you see this shape, it means either that you are in "add-node mode", or that you
are currently holding down the shift key.  In either case, when you press the left mouse button,
a new node will be created at the center of the crosshairs.

* **Diagonal two-way arrow**: You will only see this while using the TIFF Loader to create a new
data volume.  This cursor shape means that the cursor is over a corner of the box that shows the
outline of the proposed data volume; if you press the left mouse button, you can drag the
corner to change the shape of the proposed data volume.

### Hot keys

Khartes provides a number of hot keys (keys that you can press to take some action).  Here is the list:

* **x**: This extremely useful hot key was suggested by the Vesuvius Challenge Segmentation Team.
If you place the cursor over a node, so the node changes to cyan, and then press `x`, the view
will shift so as to place the designated node in the center of all 3 data slices.  This is very
helpful if you want to examine the data volume near the node, or add other nodes nearby.  Think
of `x` as meaning "X marks the spot", or representing crosshairs (slightly tilted).

* **Ctrl-s**: This hot key performs the same function as the `Save` menu item in the `File`
menu: it saves your current work to the project file.  You should try to remember to hit
`Ctrl-s` as often as possible; khartes does not create auto-backups.  The "save" operation is
very quick, since only fragments are written to disk each time; the data volumes never change
once they have been initially created on disk.

* **shift**: As mentioned in the "Advanced" section above, quickly pressing and releasing the
shift key will toggle between "normal mode" and "add-node mode".

* **delete** and **backspace**: If you place the cursor over a node, so that the node turns cyan,
you can delete the node by pressing either of these two keys.  As mentioned above, khartes does
not have an "undo" function, so think carefully before doing this.

* **v**: This hot key controls the visibility of the mesh that is overlaid on the
data in the Fragment View.  More precisely, it toggles the "visibility" setting of
the current fragment; the distinction between a fragment being "active" and "visible"
is an advanced topic discussed below, in the "Control Area: Fragments" section.  For now, think of
`v` as meaning "visible".

* **t**: This key turns on and off an advanced capability called "tracking cursors".
Tracking cursors are described in the "Tracking Cursors" section below.

## General segmentation workflow

This section of the tutorial provides some
suggestions on how to create large, self-consistent
fragments.  The key, as you will learn, is to
follow horizontal papyrus fibers, using them
as an indicator that you have not accidently
allowed your fragment to slip onto a neighboring
sheet.

**Step 0**: The most important step, which you
must take even before you start segmenting,
is to decide in which area to work.  For your first
attempt, you should start with a sheet that is clearly separated
from its neighbors; no need to 
make your learning experience difficult.

<img src="images/easier.JPG" width="360"/>

*This is a an example* of a fairly easy sheet.

For your next attempt, you might want to start with a sheet
that is separated on one side from its neighbors.

Keep in mind that after you
have created a fragment for one sheet, you can view that fragment
even while working on the next sheet, 
using it as a kind of guide.
So one strategy is to create fragments on a series of sheets that are 
parallel to
each other, starting with the easiest.

There are some areas in the scroll data volume that I found to be
too difficult to fragment.  In these areas, sheets appear, disappear, and
merge into each other in a way that seems impossible to track, no
matter which software is used.  If you try working in these areas,
prepare to be frustrated. 

![squirmy](images/squirmy.JPG)

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

**Hint for step 2:** Before you start adding new nodes onto the 
inline or crossline slice,
look in the fragment viewer to see if there are any existing nodes near
the line you are working on.
If there are, and it is feasible, move these existing nodes
onto the line.  This is to avoid the situation where a node on the line
and a node just off of the line end up close to each other, which can
cause undesirable waviness in the fragment.

<img src="images/good start.JPG" width="800" />

*Example of a good start.*  A couple of inline slices (the horizontal lines
in the fragment view) and a crossline slice (vertical line) have been 
interpreted.  Nodes near the lines have been moved onto the lines, to
maintain good node spacing.
Some contour points have been added to the bottom slice as well.
The horizontal fibers are continuous, 
which is a sign that the segmentation has been done correctly (see Step 3).
The dark spot in the upper right quadrant is 
due to a lack of data to constrain the interpolation; as more nodes are
added, the spot will be replaced by the image of the sheet.

**Step 3**: Pause, verify, repair.  The most important criterion for
a good fragment is that the horizontal fibers (as seen in the fragment view)
are continuous, since the horizontal fibers (also called the circumferential fibers)
are the ones that are most likely to contain text.
Where horizontal and vertical fibers cross, try
to make sure that the horizontal fibers are the ones that are the most visible

![sheet_skip](images/sheet_skip.JPG)

***This is bad!***  The horizontal fibers are not continuous.  This needs to be repaired by
moving nodes so that all the nodes lie on the same sheet.

**Step 3 continued**  The main problem to watch out for, as illustrated above,
is "sheet skipping": because two adjacent sheets are close together, or
even merge in some areas, the user has unintentionally started adding nodes onto
the wrong sheet.  As a result, the fibers on the left side of this picture are from
a different sheet than the fibers on the right.  This creates a
visual discontinuity, which is a signal that the user needs
to go back, analyze the existing nodes, and move as many as necessary 
until all are on the same sheet.
So again: pause, verify, repair.  The longer you wait to do this basic check, the
more repair work you will have to do later.

<img src="images/repaired.JPG" width="800" />

*The surface has been repaired;* horizontal fibers are now continuous.  The inline and crossline slices show
the location of the original (magenta) and repaired (cyan) surfaces.  Note that these overlap in one half
of the slice,
but diverge in the other half.

## Workflow notes

**Save often.**  You can simply type Ctrl-S to save your work; try to remember to do this whenever
you have added or moved a dozen or so nodes.  The "save" operation is very quick, since only 
fragments are saved; the volume data does not change and thus does not slow down the operation.

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

## Control Area: Fragments

<img src="images/fragments_tab.JPG" width="800"/>

Go to the Control Area and open the `Fragments` tab.
This tab allows you to control various aspects of your fragments.

You have already visited this tab, in order to create new fragments;
now you will have a chance to study it in detail.

The first thing to notice is that one of the rows is hightlighted with a
beige background.  The highlighting shows which fragment is
accepting new nodes.  That is, if you create a new node
in one of the data windows, this is where the node will be added.

The first two columns in the fragment list, `Active` and `Visible`, have
similar, but not identical, meanings.

If a fragment is `Visible`, this means that the fragment's
nodes and cross-section line will be drawn over the
data in the Data Slices.

When a fragment is `Active`, this means that the data that
lies on the fragment's surface (also called the fragment's texture)
will be displayed in the Fragment View.
(One exception: if a fragment's direction, which can be either X or Y,
as shown in the fifth column,
is different than the current volume's direction,
then that fragment's texture will not be displayed.)

If a fragment is set to be both `Active` and `Visible`, its
mesh and nodes will be overlaid on the fragment texture in the Fragment View,
as well as in the Data Slices.
This means that if you want to view a fragment's texture without
the overlying mesh, turn `Visible` off.  (Recall that you can also
toggle the visibility by using the `v` hot key).

As many fragments as you like can be made visible; their cross sections will all be
overlaid on the Data Slices.  This is useful if you want to compare different
fragments.  Note that the mesh of a given `Visible` fragment will not be
overlaid on the Fragment View unless that fragment is also `Active`.

In general use, only one fragment at a time can be actve; if you set a fragment to be active
by clicking its `Active` check box, the previously active fragment will be deactivated.
An exception will be discussed as an advanced topic below.

As for the remaining columns in the Fragments tab:

The third column displays the name of the fragment, and the fourth displays
the color.  These both can be edited.  When you create a new fragment,
khartes tries (not very cleverly) to guess what to name it, and chooses
a color at random.  

The list of fragments is always alphabetized by name,
so if you want to keep two fragments close to each other in the
list, give them similar name.  A wide range of characters are acceptable
in the name, so you can give your fragments names like "sheet 1 / fiber 3", "sheet 1 / fiber 4",
etc.

The fifth column in the fragments list indicates whether the fragment considers the X axis,
or the Y axis, to be the vertical direction.  The direction of the axis is important because
khartes only works with single-valued surfaces; the direction of "single value" is the 
fragment's vertical direction.  This direction is set at the time the fragment is created, and
is based on the active volume's vertical direction at that time.  The fragment's axis direction
cannot be changed once the fragment is created.

The sixth column simply shows how many nodes (points) the fragment contains.  The seventh column
gives the fragment's area in square centimeters.  This is a true area as measured in 3D space;
it is not based simply on the projected area of the fragment onto the Fragment View.

### Advanced topic: Multiple active fragments

There are times when you may want to 
have more than one active fragment at a time.  One scenario is
where each fragment represents a papyrus fiber rather than an
entire sheet of a scroll; in this case it is convenient to
be able to display simultaneously all the fibers in the fragment view window.
Another scenario is where a single sheet is divided into more
than one fragment in order to work around khartes' requirement
that each fragment be single-valued in either X or Y.

Normally, if you click on a checkbox in the `Active` column in
order to make a fragment active, the currently active fragment
will be made inactive.  In other words, normally only one fragment
can be active at a time.

However, if you hold down the Ctrl key when checking an `Active`
checkbox, that fragment will be made active, while any previous
fragments will remain active.  So use Ctrl-click in the `Active`
checkbox to allow mutiple active fragments.

As before, the beige row denotes the fragment that is accepting
new nodes.  When there are multiple active fragments, the
"accepting" row will be the one closest to the bottom of the list.

## Exporting fragments

Khartes allows you to export your fragments to [`MeshLab`](https://www.meshlab.net/),
created and maintained by the [Visual Computing Lab](https://vcg.isti.cnr.it/) of [CNR-ISTI](https://www.isti.cnr.it/en/) (Italy),
as well as to
`vc_render` and `vc_layers_from_ppm`, which are two components of [`volume-cartographer`](https://github.com/educelab/volume-cartographer),
created and maintained by the [Digital Restoration Initiative](https://www.cs.uky.edu/dri/), part
of [EduceLab](https://educelab.engr.uky.edu/)
at the [University of Kentucky](https://uky.edu/) (USA).

To export your fragment:

1. Make sure your fragment is active, that is, that it is visible
in the right-hand window.
2. In the File menu, select `Export file as mesh...`.

This will create a `.obj` file, which contains a mesh representing your
fragment, as well as a `.tif` file and a `.mtl` file.  
These three files can be loaded directly into the `MeshLab` 3D viewer,
to render
a 3D view of your surface, with the volume data textured onto
the surface.

In addition,
you can import the .obj file directly into `vc_render`.  Here is how.

(**Note for advanced users**: If multiple fragments are active,
all the active fragments will be saved into a single `.obj` file.
This is convenient for viewing in `MeshLab`, but beware! Multi-fragment
.obj files cannot be imported into `vc_render`.)

First, you need to make sure you know where the following files and
directories are located:

- Your `.volpkg` folder, the one that contains the TIFF files that you
imported into khartes
- If your `.volpkg` directory contains more than one volume, you need
to know the number of the volume that contains the TIFF files
that you used.
- The `.obj` mesh file that you just created
- The directory where you want to create a `.ppm` file, and the name
that you want to give the `.ppm` file.  The `.ppm` file is needed by
`vc_layers_from_ppm`.

So the command you want to type will look something like:
```
vc_render -v [your volpkg directory] --input-mesh [your .obj file] --output-ppm [the name of the ppm file you want to create]
```
You might need to use `--volume` to specify your volume as well, if your volpkg has more than one.

As already mentioned, the `.ppm` file that `vc_render` creates can be read into `vc_layers_from_ppm` to create a 
flattened surface volume.

## Advanced Topic: Control Area: Settings

<img src="images/settings_tab.JPG" width="800"/>

The Settings tab gives you control of some of the internal parameters of khartes.
When you are first starting out, you probably will not need to access this
tab, but as you continue, you may discover that you would like to customize
some aspects of khartes.

In the left column of this tab are some settings that control the appearance of
the entities (nodes, lines, axes, etc) that are overlaid on top of the data windows.
For each entity, you can control the size (width in the case of lines, radius in
the case of circles) by setting the corresponding value.  If you set a
size to zero, the corresponding entity will be hidden.

The opacity section is a bit more complicated to explain; the complexity is
due to certain limitations in the computer graphics library that khartes uses.

Opacity is a number ranging from 0.0 to 1.0.  An opacity of 0.0 corresponds to
total transparency, meaning that the overlay is invisible.
An opacity of 1.0 corresponds to no transparency at all; the overlay completely
hides anything behind it.  An opacity of 1.0 is the default.

For each entity, you have the choice of using the default opacity value (1.0),
or of using the value set in the `Opacity` box.  If the entity's "Apply opacity"
checkbox is checked, then the value shown in the `Opacity` box is applied when the
entity is drawn.  If the
checkbox is not checked, the entity is drawn fully opaque.

If you want to return to the original values used by khartes, press the `Restore Defaults`
button.



(currently working here)

## Things to fix

There is no undo function.

The "Loading..." overlay, which appears when a file is loading, sometimes fails to
disappear afterwards.  
This problem is so intermittent that I have not been able
to figure out how to prevent it.
Unfortunately, the only workaround is to save your work, and then exit and restart khartes.

## License

Khartes is open-source code, licensed under the GPL 3.0 license.  See LICENSE.txt for details.

## Contact

I prefer not to make my name and email address available on the public internet.
I can be reached on the [Vesuvius Challenge Discord server](https://discord.com/invite/6FgWYNjb4N)
under my user name, @khartes_chuck .

## Acknowledgements

I wrote khartes for my own gratification, and in the hope that it will be useful; 
I did not receive a grant or salary from any organization
to develop this software.

However, I benefitted immensely from the [Vesuvius Challenge](https://scrollprize.org/), who
provided the following:
* The X-ray tomography data set, without which I could not have developed khartes (along with
making the data available under reasonable conditions,
they provided the technical infrastructure that made it possible to
download the data over the internet);
* Permission to use small excerpts from the data set in this README file and the associated videos;
* A [lively community](https://discord.com/invite/6FgWYNjb4N) of fellow enthusiasts;
* In particular, the Segmentation Team (especially @hari_seldon and @djosey),
who provided invaluable user feedback and suggestions while I
was developing khartes;
* A [cash prize](https://scrollprize.substack.com/p/segmentation-tooling-winners-new), which I appreciated
for its monetary value, but more so for the recognition of the work done so far,
and for the encouragement to continue.

So a big "Thank You" to the sponsors and organizers of the Vesuvius Challenge!

And of course I wish to thank my spouse, whose support and encouragement made this all possible.





import tifffile
import numpy as np
import os
import zarr
import time
import pathlib
import re
import queue
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from utils import Utils

CHUNK_SIZE = 500

def load_tif(path):
    """This function will take a path to a folder that contains a stack of .tif
    files and returns a concatenated 3D zarr array that will allow access to an
    arbitrary region of the stack.

    We support two different styles of .tif stacks.  The first are simply
    numbered filenames, e.g., 00.tif, 01.tif, 02.tif, etc.  In this case, the
    numbers are taken as the index into the zstack, and we assume that the zslices
    are fully continuous.  Masked versions of these are available for download here:
    http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/volumes_masked/20230205180739/

    The second follows @spelufo's reprocessing, and are not 2D images but 3D cells
    of the data.  These should be labeled

    cell_yxz_YINDEX_XINDEX_ZINDEX

    where these provide the position in the Y, X, and Z grid of cuboids that make
    up the image data.  They are available to download here:
    http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/volume_grids/20230205180739/

    Note that if data is missing (e.g., a missing 00001.tif or missing cube) that 
    that region of the zarr is filled in as zeros.  For the cubes this is desirable
    to reduce space use to store non-scroll information.  A CSV annotating only
    cubes with valid scroll information is available here:
    https://github.com/spelufo/vesuvius-build/blob/main/masks/scroll_1_54_mask.csv
    
    These arrays are non-writable.
    """
    # Get a list of .tif files
    tiffs = [filename for filename in os.listdir(path) if filename.endswith(".tif")]
    if all([filename[:-4].isnumeric() for filename in tiffs]):
        # This looks like a set of z-level images
        tiffs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        paths = [os.path.join(path, filename) for filename in tiffs]
        store = tifffile.imread(paths, aszarr=True)
    elif all([filename.startswith("cell_yxz_") for filename in tiffs]):
        # This looks like a set of cell cuboid images
        pattern=r"cell_yxz_(\d+)_(\d+)_(\d+)"
        images = tifffile.TiffSequence(os.path.join(path, "*.tif"), pattern=pattern)
        # The indices (locations of chunks in zarr) are not
        # set correctly in TiffSequence, so reset them
        # below, based on the file names.
        # They will be used by images.aszarr()
        new_indices = []
        pattern_compiled = re.compile(pattern)
        maxx = 0
        maxy = 0
        maxz = 0
        for i in range(len(images.indices)):
            file = images.files[i]
            m = pattern_compiled.search(file)
            ystr,xstr,zstr = m.groups()
            # -1 because the file names are indexed from 1,1,1
            ix = int(xstr)-1
            iy = int(ystr)-1
            iz = int(zstr)-1
            new_index = (iy,ix,iz)
            maxx = max(maxx, ix)
            maxy = max(maxy, iy)
            maxz = max(maxz, iz)
            new_indices.append(new_index)
        images.indices = new_indices
        images.shape=(maxy+1,maxx+1,maxz+1)
        store = images.aszarr(axestiled={0: 1, 1: 2, 2: 0}, fillvalue=0)

    stack_array = zarr.open(store, mode="r")
    return stack_array

def load_zarr(dirname):
    stack_array = zarr.open(dirname, mode="r")
    return stack_array

def load_writable_volume(path):
    """This function takes a path to a zarr DirectoryStore folder that contains
    chunked volume information.  This zarr Array is writable and can be used
    for fast persistent storage.
    """
    if not os.path.exists(path):
        raise ValueError("Error: f{path} does not exist")
    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=False)
    return root.volume

def create_writeable_volume(path, volume_shape):
    """Generates a new zarr Array object serialized to disk as an empty array of
    zeros.  Requires the size of the array to be given; this may be arbitrarily
    large.  When initialized, this takes up very little space on disk since all
    chunks are empty.  As it is written, it can get much larger.  Be sure you
    have enough disk space!
    """
    if os.path.exists(path):
        raise ValueError("Error: f{path} already exists")
    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=True)
    volume = root.zeros(
        name="volume",
        shape=volume_shape,
        chunks=tuple([CHUNK_SIZE for d in volume_shape]),
        dtype=np.uint16,
        write_empty_chunks=False,
    )
    return volume

def slice_to_hashable(slice):
    return (slice.start, slice.stop)

def hashable_to_slice(item):
    return slice(item[0], item[1], None)


class TransposedDataView():
    def __init__(self, data, direction=0):
        self.data = data
        assert direction in [0, 1]
        self.direction = direction

    @property
    def shape(self):
        shape = self.data.shape
        if self.direction == 0:
            return (shape[2], shape[0], shape[1])
        elif self.direction == 1:
            return (shape[1], shape[0], shape[2])

    def __getitem__(self, key):
        if self.direction == 0:
            xslice, zslice, yslice = key
        elif self.direction == 1:
            yslice, zslice, xslice = key
        result = self.data[zslice, yslice, xslice]
        if len(result.shape) == 1:
            # Fancy-indexing collapses the shape, so we don't need to transpose
            return result
        if self.direction == 0:
            return result.transpose(2, 0, 1)
        elif self.direction == 1:
            return result.transpose(1, 0, 2)
'''
LRU (least-recently-used) cache based on the version
in https://github.com/zarr-developers/zarr-python.
It has been modified to work in threaded mode:
that is, when __getitem__ is called, if the requested
chunk is not in cache, a KeyError is immediately returned
to the caller (telling the caller to treat the chunk as all
zeros), and a request is submitted to ThreadPoolExecutor
to run a thread to retrieve the chunk.  Once the thread has retrieved the
chunk, the chunk is added to the cache, and (optionally)
a callback is called.
NOTE: In khartes, this callback is set to MainWindow.zarrFutureDoneCallback.

Sometimes the caller would rather wait for the
data, instead of having the request put on the work queue.
In this case, the caller needs to call 
setImmediateDataMode(True), before making requesting any data,
and after the data has been retrieved, call setImmediateDataMode(False)
(to restore request queueing).
'''
class KhartesThreadedLRUCache(zarr.storage.LRUStoreCache):
    def __init__(self, store, max_size):
        super().__init__(store, max_size)
        self.future_done_callback = None
        self.callback_called = False
        self.zero_vols = set()
        self.submitted = set()
        self.immediate_data_mode = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        # This is a dubious thing to do from a coding standpoint,
        # but over a slow connection, the user probably wants to
        # see the most-recently-requested data first.
        self.executor._work_queue = queue.LifoQueue()

    def __getitem__old(self, key):
        print("get item", key)
        return super().__getitem__(key)

    def setImmediateDataMode(self, flag):
        with self._mutex:
            self.immediate_data_mode = flag

    def __getitem__(self, key):
        try:
            # first try to obtain the value from the cache
            with self._mutex:
                value = self._values_cache[key]
                # cache hit if no KeyError is raised
                self.hits += 1
                # treat the end as most recently used
                self._values_cache.move_to_end(key)
                return value

        except KeyError:
            # cache miss, retrieve value from the store
            # print("cache miss", key)

            # wait_for_data = True means do the read right away.
            # Note that this blocks the calling program until
            # the data has been read.
            # wait_for_data = False means submit the request
            # to the thread pool.
            wait_for_data = False
            if self.immediate_data_mode:
                wait_for_data = True
            elif len(key) == 0: # not sure this ever happens
                wait_for_data = True
            else:
                # Check if key is the name of a metadata file
                # (for instance, '0/.zarray'), in which case
                # the value must be read immediately
                parts = key.split('/')
                if parts[-1][0] == '.':
                    wait_for_data = True
            raise_error = False
            with self._mutex:
                # check whether
                # key is known to correspond to an all-zeros volume,
                # or key has already been submitted to the thread queue:
                if key in self.zero_vols or key in self.submitted:
                    # this tells the caller to treat the current
                    # chunk as all zeros
                    # raise KeyError(key)
                    raise_error = True
                if not raise_error and not wait_for_data:
                    # the add() is done here, instead of below,
                    # where the request is submitted, because here
                    # the add() operation is protected by the _mutex
                    self.submitted.add(key)
                # print("submitted",self.submitted)
            if raise_error:
                raise KeyError(key)
            print("nsz miss", wait_for_data, key)
            if wait_for_data:  # read the value immediately
                value = self.getValue(key)
                self.cacheValue(key, value)
                return value
            else:  # submit to the thread pool a request to read the value
                future = self.executor.submit(self.getValue, key)
                future.add_done_callback(lambda x: self.processValue(key, x))
                raise KeyError(key)

    def getValue(self, key):
        # print("getValue", key)
        value = self._store[key]
        # print("  found", key)
        return value

    def cacheValue(self, key, value):
        with self._mutex:
            self.misses += 1
            # need to check if key is not in the cache, as it may have been cached
            # while we were retrieving the value from the store
            if key not in self._values_cache:
                # print("pv caching",key)
                self._cache_value(key, value)
                # print("  pv done")

    # This is called when the thread reports that it has
    # completed the getValue (disk read) operation
    def processValue(self, key, future):
        # print("pv", key)
        with self._mutex:
            self.submitted.discard(key)
            # print("pv submitted", self.submitted)
        try:
            # get the result
            value = future.result()
            # print("pv got value")
        except KeyError:
            # KeyError implies that the data store has no file
            # corresponding to this key, meaning that the data
            # for this chunk is all zeros.
            # The "return" statement below means that this 
            # KeyError will be relayed to the caller, who will
            # know what it means (chunk is all zeros).
            # print("pv key error", key)
            with self._mutex:
                self.zero_vols.add(key)
            if self.future_done_callback is not None:
                self.future_done_callback(key)
            return
        self.cacheValue(key, value)
        if self.future_done_callback is not None:
            self.future_done_callback(key)


class CachedZarrVolume():
    """An interface to cached volume data stored on disk as .tif files
    but not fully loaded into memory.

    Mimics the Khartes-based interface so we can use it instead of the
    existing Volume() class.


    There are four different styles of coordinates used:
    - Global - [z, y, x]:  equivalent to indexing directly in the full zarr
    - Data - [z, y, x]: for standard volumes, direct indexing in the data array,
        and offset (and potentially slices) relative to the global data.
        For cached zarr volumes, this is equivalent to global.
    - Transposed (direction=0) - [x, z, y]:  Equivalent to data but with the indices
        transposed.
    - Transposed (direction=1) - [y, z, x]:  Equivalent to data but with the indices
        transposed.
    """

    def __init__(self):
        self.data = None
        self.trdatas = None
        self.is_zarr = True
        self.data_header = None

        self.valid = False
        self.error = "no error message set"
        self.active_project_views = set()
        self.from_vc_render = False
        self.processing = False

    @property
    def shape(self):
        return self.data.shape

    def setImmediateDataMode(self, flag):
        self.klru.setImmediateDataMode(flag)
    
    def trshape(self, direction):
        shape = self.shape
        if direction == 0:
            return (shape[2], shape[0], shape[1])
        else:
            return (shape[1], shape[0], shape[2])

    @staticmethod
    def createErrorVolume(error):
        """Creates and returns an empty volume with an error message set
        """
        vol = CachedZarrVolume()
        vol.error = error
        return vol

    @staticmethod
    def sliceSize(start, stop, step):
        """Counts the number of items in a particular slice.
        """
        if step != 1:
            raise ValueError("Zarr volumes do not support slicing")
        width = stop - start
        quotient = width // step
        remainder = width % step
        size = quotient
        if remainder != 0:
            size += 1
        return size

    @staticmethod
    def globalIjksToTransposedGlobalIjks(gijks, direction):
        if direction == 0:
            return gijks[:,(1,2,0)]
        else:
            return gijks[:,(0,2,1)]

    @staticmethod
    def transposedGlobalIjksToGlobalIjks(tijks, direction):
        if direction == 0:
            return tijks[:, (2,0,1)]
        else:
            return tijks[:,(0,2,1)]

    @staticmethod
    def sortVolumeList(vols):
        """Does an in-place sort of a list of volumes by name"""
        vols.sort(key=lambda v: v.name)

    # TODO: createFromZarr and createFromTiffs share a lot of code!
    @staticmethod
    def createFromZarr(
            project,
            zarr_directory,
            name,
            # max_width=150,
            max_width=240,
        ):
        """
        Generates a new volume object from a zarr directory
        """
        tdir = pathlib.Path(zarr_directory)
        if not tdir.is_dir():
            err = f"{tdir} is not a directory"
            print(err)
            return CachedZarrVolume.createErrorVolume(err)

        output_filename = name
        if not output_filename.endswith(".volzarr"):
            output_filename += ".volzarr"
        filepath = os.path.join(project.volumes_path, output_filename)
        if os.path.exists(filepath):
            err = f"{filepath} already exists"
            print(err)
            return CachedZarrVolume.createErrorVolume(err)

        timestamp = Utils.timestamp()
        header = {
            "khartes_version": "1.0",
            "khartes_created": timestamp,
            "khartes_modified": timestamp,
            "zarr_dir": zarr_directory,
            "max_width": max_width,
        }
        # Write out the project file
        with open(filepath, "w") as outfile:
            for key, value in header.items():
                outfile.write(f"{key}\t{value}\n")

        volume = CachedZarrVolume.loadFile(filepath)
        # print("about to set callback")
        project.addVolume(volume)
        return volume


    @staticmethod
    def createFromTiffs(
            project, 
            tiff_directory, 
            name,
            max_width=240,
        ):
        """
        Generates a new volume object from a directory of TIF files.
        """
        tdir = pathlib.Path(tiff_directory)
        if not tdir.is_dir():
            err = f"{tdir} is not a directory"
            print(err)
            return CachedZarrVolume.createErrorVolume(err)
        
        output_filename = name
        if not output_filename.endswith(".volzarr"):
            output_filename += ".volzarr"
        filepath = os.path.join(project.volumes_path, output_filename)
        if os.path.exists(filepath):
            err = f"{filepath} already exists"
            print(err)
            return CachedZarrVolume.createErrorVolume(err)
        
        timestamp = Utils.timestamp()
        header = {
            "khartes_version": "1.0",
            "khartes_created": timestamp,
            "khartes_modified": timestamp,
            "tiff_dir": tiff_directory,
            "max_width": max_width,
        }
        # Write out the project file
        with open(filepath, "w") as outfile:
            for key, value in header.items():
                outfile.write(f"{key}\t{value}\n")
        
        volume = CachedZarrVolume.loadFile(filepath)
        project.addVolume(volume)
        return volume

    @staticmethod
    def loadFile(filename):
        try:
            header = {}
            with open(filename, "r") as infile:
                for line in infile:
                    key, value = line.split("\t")
                    header[key] = value
            tiff_directory = header.get("tiff_dir", None)
            zarr_directory = header.get("zarr_dir", None)
            max_width = int(header["max_width"])
        except Exception as e:
            err = f"Failed to read input file {filename}"
            print(err)
            return CachedZarrVolume.createErrorVolume(err)
        if tiff_directory is None and zarr_directory is None:
            err = f"Input file {filename} does not specify a data store"
            print(err)

        volume = CachedZarrVolume()
        volume.data_header = header
        volume.max_width = max_width
        volume.path = filename
        _, volume.name = os.path.split(filename)
        volume.version = float(header.get("khartes_version", 0.0))
        volume.created = header.get("khartes_created", "")
        volume.modified = header.get("khartes_modified", "")
        volume.from_vc_render = False
        volume.valid = True

        # These are set in common for all zarr arrays:  they always start
        # at the global origin with no stepping.
        volume.gijk_starts = [0, 0, 0]
        volume.gijk_steps = [1, 1, 1]
        ddir = ""
        if tiff_directory is not None:
            ddir = tiff_directory.strip()
        elif zarr_directory is not None:
            ddir = zarr_directory.strip()

        try:
            if tiff_directory is not None:
                print(f"Loading tiff directory {ddir}")
                array = load_tif(ddir)
            elif zarr_directory is not None:
                print(f"Loading zarr directory {ddir}")
                array = load_zarr(ddir)
        except Exception as e:
            err = f"Failed to read input directory {ddir}\n  specified in {filename}"
            print(err)
            return CachedZarrVolume.createErrorVolume(err)
        max_mem_gb = 8
        klru = KhartesThreadedLRUCache(array.store, max_size=max_mem_gb*2**30)
        zdata = zarr.open(klru, mode="r")

        # if zdata is a zarr group rather than a zarr array
        # (as is the case if the input is an OME directory),
        # look for '0', which by convention is the 
        # highest-resolution data array in the OME hierarchy, and 
        # set zdata to point to this array.  Note that this is NOT
        # taking advantage of the multi-resolution data
        # in the OME directory!  That is yet to be coded...
        if isinstance(zdata, zarr.hierarchy.Group):
            zdata = zdata['0']
        volume.data = zdata
        volume.klru = klru

        volume.trdatas = []
        volume.trdatas.append(TransposedDataView(volume.data, 0))
        volume.trdatas.append(TransposedDataView(volume.data, 1))
        volume.sizes = tuple(int(size) for size in volume.data.shape)
        return volume

    def setCallback(self, cb):
        print("setting callback")
        # self.klru.before_callback = cb
        self.klru.future_done_callback = cb

    def dataSize(self):
        """Size of the whole dataset in bytes
        """
        if self.data_header is None:
            return 0
        # TODO: fix assumption that data words are two bytes (uint16)
        return 2 * self.sizes[0] * self.sizes[1] * self.sizes[2]

    def averageStepSize(self):
        """Computes the geometric average of steps in all dimensions. Note
        that zarr arrays do not support steps so we can set this directly to 1.
        """
        return 1

    def setVoxelSizeUm(self, voxelSizeUm):
        self.apparentVoxelSize = self.averageStepSize() * voxelSizeUm

    def corners(self):
        """Returns a numpy array containing the Global positions of 
        the corners of the dataset.
        """
        gmin = np.array([0, 0, 0], dtype=np.int32)
        gmax = np.array(self.sizes, dtype=np.int32) - 1
        return np.array((gmin, gmax))

    def loadData(self, project_view):
        self.active_project_views.add(project_view)

    def unloadData(self, project_view):
        self.active_project_views.discard(project_view)
        self.data.store.invalidate()

    def createTransposedData(self):
        pass

    def ijkToTransposedIjk(self, ijk, direction):
        i,j,k = ijk
        if direction == 0:
            return (j,k,i)
        else:
            return (i,k,j)

    def transposedIjkToIjk(self, ijkt, direction):
        it,jt,kt = ijkt
        if direction == 0:
            return (kt,it,jt)
        else:
            return (it,kt,jt)

    def transposedIjkToGlobalPosition(self, ijkt, direction):
        return self.transposedIjkToIjk(ijkt, direction)

    def globalPositionsToTransposedIjks(self, gpoints, direction):
        if direction == 0:
            return gpoints[:,(1,2,0)]
        else:
            return gpoints[:,(0,2,1)]

    def transposedIjksToGlobalPositions(self, ijkts, direction):
        if direction == 0:
            return ijkts[:,(2,0,1)]
        else:
            return ijkts[:,(0,2,1)]

    def getGlobalRanges(self):
        arr = []
        for i in range(3):
            # Note we reverse the indices here
            arr.append([0, self.data.shape[2 - i]])
        return arr

    def globalAxisFromTransposedAxis(self, axis, direction):
        if direction == 0:
            return (axis + 1) % 3
        else:
            return (0,2,1)[axis]

    def getSlice(self, axis, ijkt, direction):
        """ijkt are the coordinates of the center point in transposed space
        """
        # it, jt, kt are the coordinates of the center point
        # in the untransposed grid
        (it, jt, kt) = self.transposedIjkToIjk(ijkt, direction)
        islice = slice(
            max(0, it - self.max_width), 
            min(self.data.shape[2], it + self.max_width + 1),
            None,
        )
        jslice = slice(
            max(0, jt - self.max_width), 
            min(self.data.shape[1], jt + self.max_width + 1),
            None,
        )
        kslice = slice(
            max(0, kt - self.max_width), 
            min(self.data.shape[0], kt + self.max_width + 1),
            None,
        )

        if direction == 0:
            if axis == 1:
                return self.data[kt, jslice, islice].T
            elif axis == 0:
                return self.data[kslice, jt, islice].T
            elif axis == 2:
                return self.data[kslice, jslice, it]
        else:
            if axis == 1:
                return self.data[kt, jslice, islice]
            elif axis == 2:
                return self.data[kslice, jt, islice]
            elif axis == 0:
                return self.data[kslice, jslice, it].T
        raise ValueError

    def ijIndexesInPlaneOfSlice(self, axis):
        return ((1,2), (0,2), (0,1))[axis]

    def ijInPlaneOfSlice(self, axis, ijkt):
        inds = self.ijIndexesInPlaneOfSlice(axis)
        x, y = ijkt[inds[0]], ijkt[inds[1]]
        return (min(x, self.max_width), min(y, self.max_width))

    def ijCornerInPlaneOfSlice(self, axis, ijkt):
        zij = self.ijInPlaneOfSlice(axis, ijkt)
        inds = self.ijIndexesInPlaneOfSlice(axis)
        rij = (ijkt[inds[0]], ijkt[inds[1]])
        return(rij[0]-zij[0], rij[1]-zij[1])

    def getSlices(self, ijkt, direction):
        depth = self.getSlice(2, ijkt, direction)
        xline = self.getSlice(1, ijkt, direction)
        inline = self.getSlice(0, ijkt, direction)

        return (depth, xline, inline)

class Loader():
    """Provides a cached read-only interface to a Zarr array.  

    When provided with z, x, and y slices, queries whether that
    data is available from any previously cached data.  If it
    is, returns the subslice of that cached data that contains
    the data we need.  

    If it isn't, then we:
    - clear out the oldest cached data if we need space
    - load the data we need, plus some padding
        - padding depends on the chunking of the zarr array and the
          size of the data loaded
    - store that data in our cache, along with the access time.

    The cache is a dictionary, indexed by a namedtuple of
    slice objects (zslice, xslice, yslice), that provides a
    dict of 
    {"accesstime": last access time, "array": numpy array with data}
    """

    def __init__(self, zarr_array, max_mem_gb=5):
        self.shape = zarr_array.shape
        print("Generating loader...")
        print("Data array shape: ", self.shape)
        self.cache = {}
        self.zarr_array = zarr_array
        self.max_mem_gb = max_mem_gb
        
        #full_size = self.estimate_slice_size(
        #    slice(None, None, None),
        #    slice(None, None, None),
        #    slice(None, None, None),
        #)
        #print(f"Estimated full array size (GB): {full_size:.2f}")

        chunk_shape = self.zarr_array.chunks
        if chunk_shape[0] == 1:
            self.chunk_type = "zstack"
        else:
            self.chunk_type = "cuboid"


    def _check_slices(self, cache_slice, new_slice, length):
        """Queries whether a new slice's data is contained
        within an older slice.

        Note we don't handle strided slices.
        """
        if isinstance(new_slice, int):
            new_start = new_slice
            new_stop = new_slice + 1
        else:
            new_start = 0 if new_slice.start is None else max(0, new_slice.start)
            new_stop = length if new_slice.stop is None else min(length, new_slice.stop)
        if isinstance(cache_slice, int):
            cache_start = cache_slice
            cache_stop = cache_slice + 1
        else:
            cache_start = 0 if cache_slice.start is None else cache_slice.start
            cache_stop = length if cache_slice.stop is None else cache_slice.stop
        if (new_start >= cache_start) and (new_stop <= cache_stop):
            # New slice should be the slice in the cached data that gives the request
            if isinstance(new_slice, int):
                return new_start - cache_start
            else:
                return slice(
                    new_start - cache_start,
                    new_stop - cache_start,
                    None
                )
        else:
            return None

    def check_cache(self, islice, jslice, kslice):
        """Looks through the cache to see if any cached data
        can provide the data we need.
        """
        for key in self.cache.keys():
            cache_islice = hashable_to_slice(key[0])
            cache_jslice = hashable_to_slice(key[1])
            cache_kslice = hashable_to_slice(key[2])
            sub_islice = self._check_slices(cache_islice, islice, self.shape[0])
            sub_jslice = self._check_slices(cache_jslice, jslice, self.shape[1])
            sub_kslice = self._check_slices(cache_kslice, kslice, self.shape[2])
            if sub_islice is None or sub_jslice is None or sub_kslice is None:
                continue
            # At this point we have a valid slice to index into this subarray.
            # Update the access time and return the array.
            self.cache[key]["accesstime"] = time.time()
            return self.cache[key]["array"][sub_islice, sub_jslice, sub_kslice]
        return None

    @property
    def cache_size(self):
        total_mem_gb = 0
        for _, data in self.cache.items():
            total_mem_gb += data["array"].nbytes / 1e9
        return total_mem_gb

    def view_cache(self):
        for key, value in self.cache.items():
            print("Access time: {}".format(value["accesstime"]))
            print("Slices: {}".format(key))
            print("Est size (GB): {}".format(value["array"].nbytes / 1e9))
    
    def empty_cache(self):
        """Removes the oldest item in the cache to free up memory.
        """
        if not self.cache:
            return
        oldest_time = None
        oldest_key = None
        for key, value in self.cache.items():
            if oldest_time is None or value["accesstime"] < oldest_time:
                oldest_time = value["accesstime"]
                oldest_key = key
        del self.cache[oldest_key]

    def clear_cache(self):
        while len(self.cache) > 0:
            self.empty_cache()

    def estimate_slice_size(self, islice, jslice, kslice):
        def slice_size(s, l):
            if isinstance(s, int):
                return 1
            elif isinstance(s, slice):
                s_start = 0 if s.start is None else s.start
                s_stop = l if s.stop is None else s.stop
                return s_stop - s_start
            raise ValueError("Invalid index")
        return (
            self.zarr_array.dtype.itemsize * 
            slice_size(islice, self.shape[0]) *
            slice_size(jslice, self.shape[1]) * 
            slice_size(kslice, self.shape[2])
        ) / 1e9

    def pad_request(self, islice, jslice, kslice):
        """Takes a requested slice that is not loaded in memory
        and pads it somewhat so that small movements around the
        requested area can be served from memory without hitting
        disk again.

        For zstack data, prefers padding in jk, while for cuboid
        data, prefers padding in i.
        """
        def pad_slice(old_slice, length, int_add=350):

            if isinstance(old_slice, int):
                if length == 1:
                    return old_slice
                return slice(
                    max(0, old_slice - int_add),
                    min(length,old_slice + int_add + 1),
                    None
                )
            start = old_slice.start if old_slice.start is not None else 0
            stop = old_slice.stop if old_slice.stop is not None else length
            adj_width = (stop - start) // 2 + 1
            return slice(
                max(0, start - adj_width),
                min(length, stop + adj_width),
                None
            )
        est_size = self.estimate_slice_size(islice, jslice, kslice)
        if (3 * est_size) >= self.max_mem_gb:
            # No padding; the array's already larger than the cache.
            return islice, jslice, kslice
        if self.chunk_type == "zstack":
            # First pad in j and k, then i (the z-index)
            jslice = pad_slice(jslice, self.shape[1])
            est_size = self.estimate_slice_size(islice, jslice, kslice)
            if (3 * est_size) >= self.max_mem_gb:
                return islice, jslice, kslice
            kslice = pad_slice(kslice, self.shape[2])
            est_size = self.estimate_slice_size(islice, jslice, kslice)
            if (3 * est_size) >= self.max_mem_gb:
                return islice, jslice, kslice
            islice = pad_slice(islice, self.shape[0], int_add=1)
        elif self.chunk_type == "cuboid":
            # First pad in Z by 5 in each direction if we have space, then in XY
            islice = pad_slice(islice, self.shape[0])
            est_size = self.estimate_slice_size(islice, jslice, kslice)
            if (3 * est_size) >= self.max_mem_gb:
                return islice, jslice, kslice
            jslice = pad_slice(jslice, self.shape[1])
            est_size = self.estimate_slice_size(islice, jslice, kslice)
            if (3 * est_size) >= self.max_mem_gb:
                return islice, jslice, kslice
            kslice = pad_slice(kslice, self.shape[2])
        return islice, jslice, kslice

    def __getitem__(self, key):
        """Overloads the slicing operator to get data with caching
        """
        islice, jslice, kslice = key
        #print(f"Querying {key}")
        for item in (islice, jslice, kslice):
            if isinstance(item, slice) and item.step is not None:
                raise ValueError("Sorry, we don't support strided slices yet")
            if not any([isinstance(item, slice), isinstance(item, int), isinstance(item, np.ndarray)]):
                print("Sorry, we don't yet support arbitrary access to the array.")
                print(type(item))
                raise ValueError()

        fancy_indexed = False
        if isinstance(islice, np.ndarray):
            minval, maxval = islice.min(), islice.max()
            adj_i_array = islice - minval
            islice = slice(minval, maxval + 1, None)
            fancy_indexed = True

        if isinstance(jslice, np.ndarray):
            minval, maxval = jslice.min(), jslice.max()
            adj_j_array = jslice - minval
            jslice = slice(minval, maxval + 1, None)
            fancy_indexed = True

        if isinstance(kslice, np.ndarray):
            minval, maxval = kslice.min(), kslice.max()
            adj_k_array = kslice - minval
            kslice = slice(minval, maxval + 1, None)
            fancy_indexed = True

        # First check if we have the requested data already in memory
        result = self.check_cache(islice, jslice, kslice)
        if result is not None:
            #print("Serving from cache")
            #print(self.cache_size)
            if fancy_indexed:
                return result[adj_i_array, adj_j_array, adj_k_array]
            return result
        # Pad out the requested slice before we pull it from disk
        # so that we cache neighboring data in memory to avoid
        # repeatedly hammering the disk
        padded_islice, padded_jslice, padded_kslice = self.pad_request(islice, jslice, kslice)
        #print(f"Padded query to {padded_islice, padded_jslice, padded_kslice}")
        est_size = self.estimate_slice_size(padded_islice, padded_jslice, padded_kslice)
        #print(f"Estimated padded size (GB): {est_size:.2f}")
        # Clear out enough space from the cache that we can fit the new
        # request within our memory limits.
        while self.cache and (self.cache_size + est_size) > self.max_mem_gb:
            self.empty_cache()
        padding = self.zarr_array[padded_islice, padded_jslice, padded_kslice]
        self.cache[(
            slice_to_hashable(padded_islice),
            slice_to_hashable(padded_jslice),
            slice_to_hashable(padded_kslice),
        )] = {
            "accesstime": time.time(),
            "array": padding,
        }

        result = self.check_cache(islice, jslice, kslice)
        if result is None:
            # We shouldn't get cache misses!
            print("Unexpected cache miss")
            print(islice, jslice, kslice)
            print(padded_islice, padded_jslice, padded_kslice)
            raise ValueError("Cache miss after cache loading")
        if fancy_indexed:
            return result[adj_i_array, adj_j_array, adj_k_array]
        return result

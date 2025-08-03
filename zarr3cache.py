import shutil
import json
from collections import OrderedDict
import threading
import asyncio
import numpy as np
import zarr
import random

# One LRU implementation:
# https://llego.dev/posts/implement-lru-cache-python/

class ArrayBackedCachingStore(zarr.storage.WrapperStore):
    def __init__(self, array, blocking=True):
        # zarr.config.set({'async.concurrency': 2})
        store = zarr.storage.MemoryStore()
        super().__init__(store)
        # print("abcs read only", self.read_only)
        self._array = array
        self.cache = OrderedDict()
        self.capacity = 1000
        self.concurrent_reads = 0
        self.max_concurrent_reads = 32
        # self.max_concurrent_reads = 100
        # self.capacity = 10
        # all_zeros = np.zeros(array.chunks, array.dtype)
        # self.all_zeros = zarr.core.buffer.cpu.Buffer.from_array_like(np.frombuffer(all_zeros.tobytes(), dtype=np.uint8))
        self.blocking = blocking
        self.empties_count = 0
        self.processing = set()
        self.future_done_callback = None
        self.stamp = 0

    def invalidate(self):
        self.processing = set()
        self.cache = OrderedDict()
        self.concurrent_reads = 0
        # self._array = None

    def exit_loop(self):
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(self.loop.stop)

    @classmethod
    def create_caching_array(cls, array, blocking=True):
        abcs = cls(array, blocking)
        return abcs.create_caching_array_internal()

    def create_caching_array_internal(self):
        oarr = self._array
        # Note that lumping chunks into shards is not necessary
        # for the purpose of caching.  The backing array might 
        # be sharded, but each chunk of the shard is loaded from disk
        # individually, as needed; the entire shard is not
        # loaded.  So caching individual chunks, instead of entire
        # shards, does not create extra disk i/o.

        narr = zarr.create_array(store=self, chunks=oarr.chunks, shape=oarr.shape, compressors=None, dtype=oarr.dtype)
        return narr

    # return value is not used
    async def deferred_get(self, key, prototype, byte_range, stamp):
        # print("dg", key)
        # await asyncio.sleep(5*random.random())
        # print("->", self.concurrent_reads)
        while self.concurrent_reads >= self.max_concurrent_reads:
            # await asyncio.sleep(.2*random.random())
            # await asyncio.sleep(min(2., (5+self.stamp-stamp)*.01))
            await asyncio.sleep(min(1., (5+self.stamp-stamp)*.001))
        if key not in self.processing:
            return
        self.concurrent_reads += 1
        try:
            f = self.get_from_array(key, prototype, byte_range)
            # print("f", f)
            ob = await f
        except Exception as e:
            print("deferred_get: read error", e)
            self.concurrent_reads -= 1
            return
        if key not in self.processing:
            return
        self.concurrent_reads -= 1
        # print("<-", self.concurrent_reads)
        self.cache[key] = ob
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            rkey, rval = self.cache.popitem(last=False)
            # print("** dg removed last item", rkey, "while adding", key)
        self.processing.remove(key)
        # print("proc", len(self.processing))
        if self.future_done_callback is not None:
            # ob is None if the chunk is all fill-value
            self.future_done_callback(key, ob is not None)
            pass
        # return None

    # This function overrides the WrapperStore get function.
    # It returns a chunk, given the chunk's key.  The chunk is
    # either read from the backing array, or loaded from the
    # cache.
    # If the "blocking" parameter was set to True
    # in ArrayBackedCachingStore(), the get() function will
    # wait until the chunk is loaded, if it is not already
    # in the cache.
    # If blocking=False, and the chunk is not already in
    # the cache, an empty chunk will be returned, and
    # deferred_get() will be called to initiate the loading
    # of the missing chunk.  The idea here is that
    # even though the current call to get() returns an empty
    # chunk, future calls will (once the chunk is loaded)
    # return the correct data.  This allows progressive
    # loading.
    # Note that returning None is equivalent to returning
    # an empty chunk (one filled with the fill value, usually 0.0)
    async def get(self, key, prototype, byte_range):
        # print("as get", key, byte_range)
        is_meta = not key.startswith('c')

        if key in self.cache:
            # print(" found in cache")
            self.cache.move_to_end(key)
            value = self.cache[key]
            return value
        # calls to metadata files must always block!
        if not self.blocking and not is_meta:
            self.empties_count += 1
            if key in self.processing:
                # return self.all_zeros
                return None
            self.processing.add(key)
            # print(" in queue")
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(loop.create_task, self.deferred_get(key, prototype, byte_range, self.stamp))
            self.stamp += 1
            return None
        f = self.get_from_array(key, prototype, byte_range)
        # print("f", f)
        ob = await f
        self.cache[key] = ob
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            rkey, rval = self.cache.popitem(last=False)
            # print("** removed last item", rkey, "while adding", key)
        # ob is None if the chunk is all fill-value
        '''
        if ob is None:
            print("get: empty fill")
            return self.all_zeros
        else:
            return ob
        '''
        return ob

    # The key applies to the array created by
    # create_caching_array.  This array is always at the
    # root (i.e. no group hierarchy), even if the backing array
    # is deep in a hierarchy of its own.  And this created array
    # is always in zarr 3 format.  Therefore, the key
    # will always follow zarr 3 conventions, for a root array.
    # NOTE that this function returns None if the chunk is
    # all fill-value.
    async def get_from_array(self, key, prototype, byte_range):
        # print("as gfa", key, byte_range)

        c0 = key.split('/')[1:]
        if len(c0) == 0:
            return None
        chunks = self._array.chunks
        x0 = [int(c0[i])*chunks[i] for i in range(len(c0))]
        x1 = [x0[i]+chunks[i] for i in range(len(c0))]
        # print("x", x0, x1)
        selection = (slice(x0[0],x1[0]),slice(x0[1],x1[1]),slice(x0[2],x1[2]))
        # print("gfa 0", key)
        # await asyncio.sleep(1.)
        # print("gfa 1", key)
        g = await self._array._async_array.getitem(selection)
        # await asyncio.sleep(1.)
        # print("gfa 2", key)
        fv = self._array.fill_value
        # print("a got g", key, g.shape, g.dtype)
        all_fill = False
        if (g==fv).all():
            # print("g is all fill")
            all_fill = True
        if all_fill:
            return None
        if g.shape != chunks:
            # print("g, chunks:", g.shape, chunks)
            newg = np.zeros(chunks, dtype=g.dtype)
            if not all_fill:
                newg[:g.shape[0], :g.shape[1], :g.shape[2]] = g
            g = newg
        '''
        print("a")
        gtb = g.tobytes()
        print("b")
        fb = np.frombuffer(gtb, dtype=np.byte)
        print("c", fb.dtype)
        ob = zarr.core.buffer.cpu.Buffer.from_array_like(fb)
        return ob
        print("d")
        '''
        out_buffer = zarr.core.buffer.cpu.Buffer.from_array_like(np.frombuffer(g.tobytes(), dtype=np.dtype("B")))
        return out_buffer


class TracerStore(zarr.storage.WrapperStore):

    def __init__(self, store):
        super().__init__(store)
        print("ts read only", self.read_only)

    async def get(self, key, prototype, byte_range=None):
        print("tr get", key, byte_range)
        # print(prototype)
        # print(byte_range)
        g = await self._store.get(key, prototype, byte_range)
        print("g", key, g)
        if g is None:
            print("tr g", g)
        else:
            gn = g.as_numpy_array()
            print("tr gn", gn.shape, gn.dtype)

        return g

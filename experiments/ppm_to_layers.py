import sys
import os
import pathlib
import numpy as np
import math
import time
import cv2
from scipy.interpolate import RegularGridInterpolator

sys.path.append(os.path.join(sys.path[0], '..'))
from ppm import Ppm

def vdirname(vdir, num):
    fstr = "%05d.tif"%num
    print("vdirname", num)
    return str(vdir / fstr)

def print_sleep(text):
    time.sleep(5)
    print(text)

# I could just use cv2.imread, but the approach here gives
# the possibility of later trying async input
def read_tif(fname):
    npbuf = None
    # t0 = time.time()
    try:
        # print("reading",fname, time.time()-t0)
        buf = open(fname, "rb").read()
        # print("frombuffer", time.time()-t0)
        npbuf = np.frombuffer(buf, dtype=np.uint8)
    except Exception as e:
        print("Read error:", str(e))
    if npbuf is None:
        return None
    # print("imdecode", time.time()-t0)
    im = cv2.imdecode(npbuf, cv2.IMREAD_UNCHANGED)
    # print("done", time.time()-t0)
    return im

def process_z_pair(ss_zi, ss_index, ss_next_tif, ss_xyzsl, ss_volume, ss_vdir):
    # print_sleep("top of loop")
    # t0 = time.time()
    ss_cur_tif = ss_next_tif
    # TODO: make sure tif is not out of range
    # if zi+1 > zend:
    #     break
    # print_sleep("reading tiff")
    # print("reading tif", time.time()-t0)
    # next_tif = cv2.imread(vdirname(vdir, i+1), cv2.IMREAD_UNCHANGED)
    ss_next_tif = read_tif(vdirname(ss_vdir, ss_zi+1))
    # for testing!
    # if ss_zi == 9367:
    #     # return ss_next_tif
    #     ss_next_tif[:,:] = 65535

    # print_sleep("creating tif_xyzs")
    # print("creating tif_xyzs", time.time()-t0)
    print("indexes",ss_index[ss_zi],ss_index[ss_zi+1])
    ss_tif_xyzs = ss_xyzsl[ss_index[ss_zi]:ss_index[ss_zi+1], :]
    # the -= operation probably affects xyzsl as well,
    # but we won't be using those rows of tif_xyzs again
    # ss_tif_xyzs[:,2] -= ss_zi
    print("tif_xyzs", ss_tif_xyzs.shape)
    # print(np.max(tif_xyzs, axis=0))

    i0 = np.floor(ss_tif_xyzs[:,0]).astype(np.int32)
    i1 = np.floor(ss_tif_xyzs[:,1]).astype(np.int32)
    i2 = np.floor(ss_tif_xyzs[:,2]).astype(np.int32)
    f0 = ss_tif_xyzs[:,0]-i0
    f1 = ss_tif_xyzs[:,1]-i1
    f2 = ss_tif_xyzs[:,2]-i2
    i2 -= ss_zi
    # iv0 = cur_tif.astype(np.float32)
    # iv1 = next_tif.astype(np.float32)
    iv0 = ss_cur_tif
    iv1 = ss_next_tif

    us = ss_tif_xyzs[:,3].astype(np.int32)
    vs = ss_tif_xyzs[:,4].astype(np.int32)
    ls = ss_tif_xyzs[:,5].astype(np.int32)
    # volume[ls,us,vs] = interp
    # interp = (
    ss_volume[ls,us,vs] = (
            (1.-f2)*((1.-f1)*((1.-f0)*iv0[i1,i0] + f0*iv0[i1,i0+1])+
            f1*((1.-f0)*iv0[i1+1,i0] + f0*iv0[i1+1,i0+1]))+
            f2*((1.-f1)*((1.-f0)*iv1[i1,i0] + f0*iv1[i1,i0+1])+
            f1*((1.-f0)*iv1[i1+1,i0] + f0*iv1[i1+1,i0+1]))
            )
    return ss_next_tif

def process_z_band(tt_z0, tt_z1, tt_vdir, tt_ijks, tt_normals, tt_zminmaxar, tt_nrange, tt_volume, tt_next_tif):
    tt_th = tt_next_tif.shape[0]
    tt_tw = tt_next_tif.shape[1]
    tt_iv = np.zeros((2,tt_th,tt_tw), dtype=np.float32)
    tt_thrange = np.arange(tt_th)
    tt_twrange = np.arange(tt_tw)
    tt_zminar = tt_zminmaxar[:,:,0]
    tt_zmaxar = tt_zminmaxar[:,:,1]
    tt_nmin = tt_nrange[0]
    tt_nmax = tt_nrange[-1]
    tt_nn = len(tt_nrange)

    # xyzsl = None
    # index = None
    print("z0", tt_z0)
    tt_inrange = ((tt_zminar>=tt_z0)&(tt_zminar<=tt_z1)|(tt_z0>=tt_zminar)&(tt_z0<=tt_zmaxar))
    print("inrange", tt_inrange.shape, tt_inrange.sum())
    # changing integer coordinates into floats to ensure
    # xyzs (created by concatenation) will be float32
    tt_uvinrange = np.argwhere(tt_inrange).astype(np.float32)
    print("uvinrange", tt_uvinrange.shape, tt_uvinrange.dtype)
    tt_xyzs = np.zeros((tt_uvinrange.shape[0], tt_nrange.shape[0], 3), dtype=np.float32)
    tt_xyzs[:,:,:] = tt_ijks[tt_inrange][:,np.newaxis,:]+tt_nrange[np.newaxis,:,np.newaxis]*tt_normals[tt_inrange][:,np.newaxis,:]
    print("xyzs",tt_xyzs.shape, tt_xyzs.dtype)
    tt_inrange = None

    # tt_narray = np.zeros((tt_xyzs.shape[0], tt_nrange.shape[0]), dtype=np.float32)
    tt_narray = np.repeat((tt_nrange[np.newaxis,:]-tt_nmin).astype(np.float32), tt_xyzs.shape[0], axis=0)[:,:,np.newaxis]
    print("narray", tt_narray.shape, tt_narray.dtype)
    # tt_uvarray = np.zeros((tt_xyzs.shape[0], tt_nrange.shape[0], 2), dtype=np.float32)
    tt_uvarray = np.repeat(tt_uvinrange[:,np.newaxis,:], tt_nn, axis=1)
    print("uvarray", tt_uvarray.shape, tt_uvarray.dtype)
    tt_uvinrange = None
    tt_xyzs = np.concatenate((tt_xyzs, tt_uvarray, tt_narray), axis=2)
    print("xyzs", tt_xyzs.shape, tt_xyzs.dtype)
    tt_xyzsl = tt_xyzs.reshape(-1,6)
    print("xyzsl", tt_xyzsl.shape, tt_xyzsl.dtype)
    ''''''
    print("sorting args")
    tt_xyzsl_sort = np.argsort(tt_xyzsl[:,2])
    print("xyzsl_sort", tt_xyzsl_sort.shape, tt_xyzsl_sort.dtype)
    print("applying sorted args")
    tt_xyzsl = tt_xyzsl[tt_xyzsl_sort]
    tt_xyzsl_sort = None
    ''''''
    '''
    # awkward way to do inplace sorting on third column.  But very slow!
    # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    # print("converting")
    # vtest = tt_xyzsl.view("f4,f4,f4,f4,f4,f4")
    # print(tt_xyzsl.dtype, tt_xyzsl.shape, vtest.dtype, vtest.shape)
    print("sorting")
    tt_xyzsl.view("f4,f4,f4,f4,f4,f4").sort(order=["f2"], axis=0)
    '''
    tt_xyzs = None
    print("xyzsl", tt_xyzsl.shape)
    # print(xyzs[100])
    tt_zrange = np.arange(tt_z1+1)
    print("indexing")
    tt_index = np.searchsorted(tt_xyzsl[:,2], tt_zrange, side='right')
    tt_zrange = None
    for tt_i in range(tt_z0,tt_z1):
        tt_next_tif = process_z_pair(tt_i, tt_index, tt_next_tif, tt_xyzsl, tt_volume, tt_vdir)
    return tt_next_tif

def compute_zmin_zmax(uu_nrange, uu_ijks, uu_normals):
    uu_nmin = uu_nrange[0]
    uu_nmax = uu_nrange[-1]
    uu_h = uu_ijks.shape[0]
    uu_w = uu_ijks.shape[1]

    print("compute normals_valid")
    uu_normals_valid = (uu_normals!=0).any(axis=2)

    print("compute zminmaxar")
    uu_zminmaxar = np.zeros((uu_h,uu_w,2), dtype=np.float32)
    uu_zminmaxar[:,:,0] = (uu_ijks[:,:,2]+uu_nmin*uu_normals[:,:,2])
    uu_zminmaxar[:,:,1] = (uu_ijks[:,:,2]+uu_nmax*uu_normals[:,:,2])
    uu_minmax_switched = uu_zminmaxar[:,:,1] < uu_zminmaxar[:,:,0]
    print(uu_minmax_switched.shape)
    # print(zminmaxar[minmax_switched,:].shape, zminmaxar[minmax_switched,:][:,[1,0]].shape)
    uu_zminmaxar[uu_minmax_switched,:] = uu_zminmaxar[uu_minmax_switched,:][:,[1,0]]
    print(uu_zminmaxar.shape, uu_zminmaxar[uu_normals_valid].shape)
    # uu_zmin = uu_zminmaxar[uu_normals_valid].min()
    # uu_zmax = uu_zminmaxar[uu_normals_valid].max()
    '''
    print("minz maxz",zmin,zmax)
    print("switchtest", 
          zminmaxar[normals_valid,0].min(),
          zminmaxar[normals_valid,1].max())
    # normals_valid = None
    '''
    return uu_zminmaxar, uu_normals_valid

case = 2

if case == 0:
    ppm = Ppm.loadPpm(pathlib.Path(r'C:\Vesuvius\Projects\large_frag_1846.khprj\orig obj\20230827161846.ppm'))
    ofpm = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\mesh.obj')
    ofpn = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\normals.obj')
elif case == 2:
    ppm = Ppm.loadPpm(pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\1846.ppm'))
    ofpm = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\mesh2.obj')
    ofpn = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\normals2.obj')
    ofl = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\layers')
    
vdir = pathlib.Path(r"H:\Vesuvius\Scroll1.volpkg\volumes\20230205180739")

print("vdir test",vdirname(vdir,25))
'''
im = read_tif(vdirname(vdir,25))
if im is None:
    print("im is None!")
else:
    print("im", im.shape)
'''

ppm.loadData()

'''
# pts = np.array(((3000.5, 3000.5),(-1.,-1.)))
pts = np.array(((3000.5, 3000.5),(-3000.,-3000.)))
# pts = np.array(((3000.5, 3000.5),))
print (ppm.ijk_interpolator(pts))
print (ppm.normal_interpolator(pts))
'''
print(ppm.data[2500,1500])
print(ppm.data[2500,1501])
print(ppm.data[2501,1500])
print(ppm.ijks.shape)
rr_ijks = ppm.ijks.astype(np.float32)
rr_normals = ppm.normals.astype(np.float32)
rr_h = rr_ijks.shape[0]
rr_w = rr_ijks.shape[1]
rr_uvs = np.mgrid[0:rr_h, 0:rr_w]
rr_uvs = rr_uvs.transpose(1,2,0)
print(rr_uvs.shape)

# uvks = np.concatenate((uvs, np.full((uvs.shape[0],uvs.shape[1],1),-1)), axis=2)
# print(uvks.shape, uvks.dtype)
rr_hrange = np.arange(rr_h)
rr_wrange = np.arange(rr_w)
rr_nmin = -32
rr_nmax = 32
rr_nrange = np.arange(rr_nmin, rr_nmax+1)
rr_nn = rr_nmax-rr_nmin+1

# too big!
# zs = np.full((h,w,nn), -1.0, dtype=np.float32)

rr_zminmaxar, rr_normals_valid = compute_zmin_zmax(rr_nrange, rr_ijks, rr_normals)
rr_zmin = rr_zminmaxar[rr_normals_valid].min()
rr_zmax = rr_zminmaxar[rr_normals_valid].max()

rr_volume = np.zeros((rr_nn,rr_h,rr_w), dtype=np.uint16)
# iv = np.zeros((2,h,w), dtype=np.float32)
# iv = None
# thrange = None
# twrange = None

rr_zstep = 200
# TODO: check tif directory and make sure zstart and
# zend are in range
rr_zstart = int(math.floor(rr_zmin/rr_zstep)*rr_zstep)
# zstart = max(zstart, 0)
rr_zend = int(math.ceil(rr_zmax))

# TODO: for testing
'''
rr_zstep = 50
rr_zstart = 9300
rr_zend = 9400
rr_zstep = 47
rr_zstart = 9300
rr_zend = 9400
# rr_zend = 9369
'''

rr_next_tif = read_tif(vdirname(vdir, rr_zstart))

for rr_z0 in range(rr_zstart, rr_zend, rr_zstep):
    rr_z1 = min(rr_zend, rr_z0+rr_zstep)
    rr_next_tif = process_z_band(rr_z0, rr_z1, vdir, rr_ijks, rr_normals, rr_zminmaxar, rr_nrange, rr_volume, rr_next_tif)

total = (rr_volume>0).sum()
print("total", total)

for i,layer in enumerate(rr_volume):
    print("layer", i)
    cv2.imwrite(str(ofl / ("%02d.tif"%i)), rr_volume[i,:,:].astype(np.uint16))
print("done")


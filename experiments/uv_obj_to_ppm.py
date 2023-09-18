import sys
import os
import pathlib
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import cv2
import skimage

ipath = pathlib.Path(r'C:\Vesuvius\Projects\large_frag_1846.khprj\orig obj\20230827161846')
opath = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\1846')

pobj = ipath.with_suffix('.obj')
ptif = ipath.with_suffix('.tif')

tif = cv2.imread(str(ptif), cv2.IMREAD_UNCHANGED)
print("tif", tif.shape)

fobj = pobj.open("r")

vrtl = []
tvrtl = []
nvrtl = []
trgl = []

for line in fobj:
    line = line.strip()
    words = line.split()
    if words[0][0] == '#':
        continue
    elif words[0] == 'v':
        vrtl.append([float(w) for w in words[1:4]])
    elif words[0] == 'vn':
        nvrtl.append([float(w) for w in words[1:4]])
    elif words[0] == 'vt':
        tvrtl.append([float(w) for w in words[1:3]])
    elif words[0] == 'f':
        trgl.append([int(w.split('/')[0])-1 for w in words[1:4]])

print("obj", len(vrtl), len(nvrtl), len(tvrtl), len(trgl))

ijks = np.array(vrtl)
normals = np.array(nvrtl)
uvs = np.array(tvrtl)
trgs = np.array(trgl)

h = tif.shape[0]
w = tif.shape[1]

iuvs = np.copy(uvs)
iuvs[:,0] *= (w-1)
# For compatibility with VC:
iuvs[:,1] = (h-1)*(1.-iuvs[:,1])

iuvtrgs = iuvs[trgs]
# print(iuvs.shape, trgs.shape, iuvtrgs.shape)

uvtrg = np.full((h, w), -1, dtype=np.int32)

print("start drawing triangles")
for i in range(len(iuvtrgs)):
    # cv2.fillConvexPolygon is at least 10 times
    # faster than skimage.draw.polygon
    # but unfortunately is not accurate enough!
    r = iuvtrgs[i,:,1]
    c = iuvtrgs[i,:,0]
    rr, cc = skimage.draw.polygon(r, c)
    uvtrg[rr,cc] = i
print("done drawing triangles")

# now uvtrgs is a h x w array with a triangle index at each
# grid point

potif = opath.with_suffix(".tif")
cv2.imwrite(str(potif), uvtrg.astype(np.uint16))

tco = (2500,1500)
tcox = (2501,1500)
tcoy = (2500,1501)
# print(uvtrg[tco])
# print(uvtrg[tcox])
# print(uvtrg[tcoy])

# adapted from https://stackoverflow.com/questions/31442826/increasing-efficiency-of-barycentric-coordinate-calculation-in-python
frombary = np.ones((iuvtrgs.shape[0],3,3), dtype=np.float64)
tc = (0,2,1)
frombary[:,(1,0),:] = iuvtrgs.transpose(tc)

# print("frombary", frombary.shape)
# print(frombary[uvtrg[tco]])

# converter from uv to barycentric coordinates, one 3x3
# conversion matrix per triangle
tobary = np.linalg.inv(frombary)

# print("tobary", tobary.shape)
# print(tobary[uvtrg[tco]])

ppmheader = '''width: %d
height: %d
dim: 6
ordered: true
type: double
version: 1
<>
'''%(w,h)

poppm = opath.with_suffix(".ppm")
fppm = poppm.open("wb")
fppm.write(ppmheader.encode('utf8'))
# print("header written")
# ppmdata.tofile(fppm)


# process output in strips of height "ssize" pixels, in order
# to reduce memory usage
ssize = 100
for curh in range(0,h,ssize):
    h0 = curh
    h1 = curh+ssize
    h1 = min(h,h1)
    dh = h1-h0
    print("strip",h0,h1)

    # dh by w array of uv values
    uvar = np.mgrid[h0:h1, :w]

    # windowed strip of uvtrg
    luvtrg = uvtrg[h0:h1]
    
    # print(uvar.shape)
    # ppmdata will hold the output data (ijks and normals),
    # in a layout that can be directly saved to a ppm file
    ppmdata = np.zeros((dh,w,2,3), dtype=np.float64)
    
    # for conversion, for each uv point need coords (u,v,1)
    uvar = np.vstack((uvar, np.ones((1, dh, w))))
    uvar = uvar.transpose(1,2,0)
    
    # print(uvar.shape)
    
    # b is a boolean that is true if the trg index is >= 0 at
    # the given uv (trg index < 0 means that this uv point is
    # outside the surface)
    b = luvtrg >= 0
    
    # print(b.shape)
    
    # tobaryar contains a 3x3 conversion matrix at
    # each uv point; the conversion matrix is selected
    # based on the trgl index at that point
    tobaryar = np.zeros((dh,w,3,3), dtype=np.float64)
    tobaryar[b] = tobary[luvtrg[b]]
    
    # print(tobaryar.shape, tobary.shape, uvtrg.shape, b.shape, tobary[uvtrg[b]].shape)
    
    # uvarr is uvar with an additional axis to make
    # further operations more convenient
    uvarr = uvar[:,:,np.newaxis,:]
    
    # print("uvarr", uvarr.shape)
    # print(uvar[tco])
    # print(uvar[tcox])
    # print(uvar[tcoy])
    # print("tobaryar", tobaryar.shape)
    # print(tobaryar[tco])
    # print(tobaryar[tcox])
    # print(tobaryar[tcoy])
    
    prod = tobaryar*uvarr
    
    # print("prod", prod.shape)
    # print(prod[tco])
    # print(prod[tcox])
    # print(prod[tcoy])
    
    # baryar contains the barycentric coordinates (3 numbers)
    # of each uv point.  The coordinates are relative to the
    # triangle whose index is contained in uvtrg at that uv point.
    baryar = prod.sum(axis=-1)
    
    # print("baryar", baryar.shape)
    # print(baryar[tco])
    # print(baryar[tcox])
    # print(baryar[tcoy])
    
    # large array containing the xyz (same as ijk, I'm not
    # consistent) points of the 3 vertices of the triangle,
    # one 3x3 array per uv point
    vxyzar = np.zeros((dh,w,3,3), dtype=np.float64)
    vxyzar[b] = ijks[trgs[luvtrg[b]]]
    
    # print("vxyzar", vxyzar.shape)
    # print(vxyzar[tco])
    # print(vxyzar[tcox])
    # print(vxyzar[tcoy])
    
    # multiplication step of the barycentric interpolation
    prod = baryar[:,:,:,np.newaxis]*vxyzar
    
    # print("prod", prod.shape)
    # print(prod[tco])
    
    # summation step of the barycentric interpolation
    # xyzar has the interpolated xyz coordinates of each uv point
    xyzar = prod.sum(axis=-2)
    # print(xyzar.shape, ppmdata.shape)
    # print(ppmdata[h0:h1,:,0,:].shape)
    ppmdata[:,:,0,:] = xyzar
    
    # print("xyzar", xyzar.shape)
    # print(xyzar[tco])
    # print(xyzar[tcox])
    # print(xyzar[tcoy])
    
    vnormar = np.zeros((dh,w,3,3), dtype=np.float64)
    vnormar[b] = normals[trgs[luvtrg[b]]]
    # multiplication step of the barycentric interpolation
    prod = baryar[:,:,:,np.newaxis]*vnormar
    
    # summation step of the barycentric interpolation.
    # normar has the interpolated (Phong interpolation) normal at each
    # uv point.  This normal has not yet been normalized
    normar = prod.sum(axis=-2)
    
    # print(normar.shape)
    # print(normar[tco])
    
    nsq = (normar*normar).sum(axis=-1)
    nsq = np.sqrt(nsq)
    b = (nsq != 0)
    nsq[b] = 1/nsq[b]
    
    # print(nsq.shape)
    
    normar = normar*nsq[:,:,np.newaxis]
    ppmdata[:,:,1,:] = normar
    ppmdata.tofile(fppm)
    
    # print(normar[tco])
    
# print(ppmdata[tco])


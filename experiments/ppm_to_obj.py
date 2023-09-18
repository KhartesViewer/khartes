import sys
import os
import pathlib
import numpy as np

sys.path.append(os.path.join(sys.path[0], '..'))
from ppm import Ppm

case = 2

if case == 0:
    ppm = Ppm.loadPpm(pathlib.Path(r'C:\Vesuvius\Projects\large_frag_1846.khprj\orig obj\20230827161846.ppm'))
    ofpm = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\mesh.obj')
    ofpn = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\normals.obj')
elif case == 2:
    ppm = Ppm.loadPpm(pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\1846.ppm'))
    ofpm = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\mesh2.obj')
    ofpn = pathlib.Path(r'C:\Vesuvius\Projects\ppmtest\normals2.obj')

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

alluvs = np.mgrid[0:ppm.ijks.shape[0], 0:ppm.ijks.shape[1]].astype(np.float64)
alluvs[0] /= ppm.ijks.shape[0]-1
alluvs[1] /= ppm.ijks.shape[1]-1
print("alluvs", alluvs.shape)
alluvs = alluvs.transpose(1,2,0)
print("alluvs transp", alluvs.shape)

center = (3800,1900)
height = 8
# height = 800
width = 600
hh = height//2
hw = width//2
x0 = center[0]-hw
y0 = center[1]-hh
x1 = center[0]+hw
y1 = center[1]+hh
####
'''
x0 = 8200
x1 = 8400
y0 = 4600
y1 = 4800
width = x1-x0
height = y1-y0
'''

ijks = ppm.ijks[y0:y1,x0:x1,:]
normals = ppm.normals[y0:y1,x0:x1,:]
uvs = alluvs[y0:y1,x0:x1]
# ijks = ijks.reshape(-1,3)
print("ijks", ijks.shape)
print("uvs", uvs.shape)

# ijks.transpose().shape = (3, width, height)
# tijks = ijks.transpose().reshape(3,-1)
tijks = ijks.reshape(-1,3)
print(tijks.shape)
tuvs = uvs.reshape(-1,2)
# tnormals = normals.transpose().reshape(3,-1)
tnormals = normals.reshape(-1,3)
wh = width*height
trgls0 = np.zeros((wh, 3), dtype=np.int32)
trgls0[:,0] = np.arange(wh)
trgls0[:,1] = np.arange(wh)+1
trgls0[:,2] = np.arange(wh)+width
trgls1 = np.zeros((wh, 3), dtype=np.int32)
trgls1[:,1] = np.arange(wh)
trgls1[:,0] = np.arange(wh)+1
trgls1[:,2] = np.arange(wh)-width+1
trgls = np.concatenate((trgls0, trgls1), axis=0)

trgls = trgls[(trgls>=0).all(axis=1)]
trgls = trgls[(trgls<wh).all(axis=1)]
rows = trgls // width
trgls = trgls[rows[:,0]==rows[:,1]]
# print("trgls", trgls.shape)

# trgls = np.concatenate((trgls, trgls2), axis=0)
print("trgls", trgls.shape)

b = (tnormals != 0.).any(axis=1)
nvalid = np.sum(b)
print("vertices",len(tijks),"valid",nvalid)
print("tijks", tijks.shape)
tijks = np.concatenate((tijks, np.full((tijks.shape[0],1), -1)), axis=1)
print("tijks", tijks.shape)
tijks[b,3] = np.arange(nvalid)

of = ofpm.open("w")
for ijk in tijks:
    if ijk[3] < 0:
        continue
    print("v %f %f %f"%(ijk[0], ijk[1], ijk[2]), file=of)

for i,uv in enumerate(tuvs):
    if tijks[i,3] < 0:
        continue
    print("vt %f %f"%(uv[0], uv[1]), file=of)

# trgls += 1
for trgl in trgls:
    ntrgl = tijks[trgl,3]
    if (ntrgl < 0).any():
        continue
    # obj files start counting at 1, not 0
    ntrgl += 1
    print("f %d %d %d"%(ntrgl[0],ntrgl[1],ntrgl[2]), file=of)

d = 8
sijks = ijks - 0.*d*normals
sijks2 = ijks + d*normals
sijks = np.concatenate((ijks, sijks, sijks2), axis=0)
stijks = sijks.reshape(-1,3)
nijks = tijks.shape[0]
stijks = np.concatenate((stijks, np.full((stijks.shape[0],1), -1)), axis=1)
stijks[0:nijks][b,3] = np.arange(nvalid)
stijks[nijks:2*nijks][b,3] = np.arange(nvalid,2*nvalid)
stijks[2*nijks:3*nijks][b,3] = np.arange(2*nvalid,3*nvalid)
trgls2 = np.zeros((wh, 3), dtype=np.int32)
trgls2[:,0] = np.arange(wh)
trgls2[:,1] = np.arange(wh)+wh
trgls2[:,2] = np.arange(wh)+2*wh
rows = trgls2 // width
trgls2 = trgls2[rows[:,0]==height//2]
of = ofpn.open("w")
for ijk in stijks:
    if ijk[3] < 0:
        continue
    print("v %f %f %f"%(ijk[0], ijk[1], ijk[2]), file=of)

# obj files start counting at 1, not 0
# trgls += 1
for trgl in trgls2:
    ntrgl = stijks[trgl,3]
    if (ntrgl < 0).any():
        continue
    # obj files start counting at 1, not 0
    ntrgl += 1
    print("f %d %d %d"%(ntrgl[0],ntrgl[1],ntrgl[2]), file=of)

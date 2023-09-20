import pathlib
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import math
import cv2
import matplotlib.pyplot as plt

'''
Based on:

Structure-oriented smoothing and semblance
D. Hale - Center for Wave Phenomena, 2009
https://scholar.google.com/scholar?cluster=15774580112029706695

Other relevant papers:

https://en.wikipedia.org/wiki/Structure_tensor

Horizon volumes with interpreted constraints
X Wu, D Hale - Geophysics, 2015
https://scholar.google.com/scholar?cluster=8389346155142410449

Estimators for orientation and anisotropy in digitized images
LJ Van Vliet, PW Verbeek - ASCI, 1995
https://scholar.google.com/scholar?cluster=8104629214698074825

Fast structural interpretation with structure-oriented filtering
Gijs C. Fehmers and Christian F. W. H Hocker - Geophysics, 2003
https://sci-hub.se/10.1190/1.1598121
'''

tifname = r"C:\Vesuvius\scroll 1 2000-2030\02015.tif"
outdir = pathlib.Path(r"C:\Vesuvius\Projects\tnorm")

tif = cv2.imread(tifname, cv2.IMREAD_UNCHANGED).astype(np.float32)
print("tif", tif.shape, tif.dtype)
# tif = tif[1500:2200, 3500:4200]
# tif = tif[1500:3500, 3500:5500]
tif = tif[:5500, :]
print("tif", tif.shape, tif.dtype)
print(np.min(tif), np.max(tif))
tif /= 65535.
print(np.min(tif), np.max(tif))
print(np.sum(tif==0.), np.sum(tif==1.))

# cv2.imshow doesn't work with my installation of opencv
# (some kind of anaconda problem...)
# plt.imshow(tif, cmap='gray')
# plt.show(block=True)
# plt.pause(1)

# gaussian partial derivative.
# currently using gaussian_filter1 from scipy.ndimage
# (order=1 means first-order derivative).
# Probably faster to use OpenCV, but a bit more coding:
# https://stackoverflow.com/questions/35012080/how-to-apply-a-partial-derivative-gaussian-kernel-to-an-image-with-opencv
sigma0 = 1.  # value used by Hale
sigma0 = 2.
gx = gaussian_filter1d(tif, sigma0, axis=1, order=1)
gx = gaussian_filter1d(gx, sigma0, axis=0, order=0)
print(np.min(gx), np.max(gx))
gy = gaussian_filter1d(tif, sigma0, axis=0, order=1)
gy = gaussian_filter1d(gy, sigma0, axis=1, order=0)
print(np.min(gy), np.max(gy))

# gaussian blur
# OpenCV function is several times faster than
# numpy equivalent
sigma1 = 8. # value used by Hale
sigma1 = 16.
# gx2 = gaussian_filter(gx*gx, sigma1)
gx2 = cv2.GaussianBlur(gx*gx, (0, 0), sigma1)
print(np.min(gx2), np.max(gx2))
# gy2 = gaussian_filter(gy*gy, sigma1)
gy2 = cv2.GaussianBlur(gy*gy, (0, 0), sigma1)
print(np.min(gy2), np.max(gy2))
# gxy = gaussian_filter(gx*gy, sigma1)
gxy = cv2.GaussianBlur(gx*gy, (0, 0), sigma1)
print(np.min(gxy), np.max(gxy))

gar = np.array(((gx2,gxy),(gxy,gy2)))
print("gar", gar.shape, gar.dtype)
gar = gar.transpose(2,3,0,1)
print("gar", gar.shape, gar.dtype)

# Explicitly calculate eigenvalue of 2x2 matrix instead of
# using the numpy linalg eigh function; 
# the explicit method is about 10 times faster.
# See https://www.soest.hawaii.edu/martel/Courses/GG303/Eigenvectors.pdf
# for a derivation

'''
eigvals, eigvecs = np.linalg.eigh(gar)
print("eigvals", eigvals.shape, eigvals.dtype)
print("eigvecs", eigvecs.shape, eigvecs.dtype)

# eigh presents eigenvalues in ascending order
lu = eigvals[:,:,1]
lv = eigvals[:,:,0]
'''

ad = gx2+gy2
sq = np.sqrt((gx2-gy2)**2+4*gxy**2)
lu = .5*(ad+sq)
lv = .5*(ad-sq)
# lv should never be < 0, but numerical issues
# apparently sometimes cause it to happen
lv[lv<0]=0

# End of explicit calculation of eigenvalues

# both eigenvalues are non-negative, and lu >= lv.
# lu is zero if gx2, gy2, and gxy are all zero
print("lu", np.min(lu), np.max(lu))
print("lv", np.min(lv), np.max(lv))
# if lu is 0., set lu and lv to 1.; this will give
# the correct values for isotropy (1.0) and linearity (0.0)
# for this case
lu0 = (lu==0)
print("lu==0", lu0.sum())
lu[lu0] = 1.
lv[lu0] = 1.

isotropy = lv/lu
linearity = (lu-lv)/lu
coherence = ((lu-lv)/(lu+lv))**2
print("linearity", linearity.shape, linearity.dtype)
print(np.min(linearity), np.max(linearity))
# plt.imshow(tif, cmap='gray')
# plt.figure()
# plt.imshow(linearity, cmap='gray')
# plt.show()

# explicitly calculate normalized eigenvectors
# eigenvector u
vu = np.concatenate((gxy, lu-gx2)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
vulen = np.sqrt((vu*vu).sum(axis=2))
vulen[vulen==0] = 1
vu /= vulen[:,:,np.newaxis]
print("vu", vu.shape, vu.dtype)

# eigenvector v
vv = np.concatenate((gxy, lv-gx2)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
vvlen = np.sqrt((vv*vv).sum(axis=2))
vvlen[vvlen==0] = 1
vv /= vvlen[:,:,np.newaxis]
print("vv", vv.shape, vv.dtype)

# All done with the calculations; time to draw
rgb = np.zeros((tif.shape[0], tif.shape[1], 3), dtype=np.uint8)
alpha = .75
# alpha = 1.0
rgb += (tif*alpha*255).astype(np.uint8)[:,:,np.newaxis]
# rgb = tif[:,:,np.newaxis]

# color highly linear areas red
# rgb[:,:,0] += (linearity*(1-alpha)*255).astype(np.uint8)

# color areas of zero gradient green
# rgb[:,:,1] += (lu0*(1-alpha)*255).astype(np.uint8)

# half-distance between grid points where dip lines will be drawn
dh = 25
drawpoints = np.mgrid[dh:rgb.shape[0]:2*dh, dh:rgb.shape[1]:2*dh].transpose(1,2,0)
print("drawpoints", drawpoints.shape, drawpoints.dtype)
print(drawpoints[0,0], drawpoints[-1,-1])

# half-length (in pixels) of line that will be drawn in the direction of
# linear features (this line length will be multiplied by
# "linearity")
linelen = 25.
# lvecs = linelen*vv*linearity[:,:,np.newaxis]
# try coherence instead
lvecs = linelen*vv*coherence[:,:,np.newaxis]
print("lvecs", lvecs.shape, lvecs.dtype)
dplvecs = lvecs[drawpoints[:,:,0], drawpoints[:,:,1],:]
# switch terms to change from y,x to x,y coordinates
drawpoints = drawpoints[:,:,::-1]

# x,y of one end of the line
x0 = (drawpoints-dplvecs)
print("x0", x0.shape, x0.dtype)
# x,y of other end of the line
x1 = (drawpoints+dplvecs)
# center of line
xc = drawpoints

lines = np.concatenate((x0,x1), axis=2)
lines = lines.reshape(-1,1,2,2).astype(np.int32)
points = xc.reshape(-1,1,1,2).astype(np.int32)
# print(lines[0])
# print(lines[1])
# print(lines[-1])
# draw lines
cv2.polylines(rgb, lines, False, (255,255,0), 2)
# draw center points
cv2.polylines(rgb, points, True, (0,255,255), 8)

# tif = tif[1500:3500, 3500:5500]
# show the inner part of the full image
plt.imshow(rgb[1500:3500, 3500:5500])
plt.show()

# save the full image as a jpeg
rgbout = str(outdir / 'test.jpg')
# the [:...] stuff is to convert rgb to bgr
cv2.imwrite(rgbout, rgb[:,:,::-1])

print("done")

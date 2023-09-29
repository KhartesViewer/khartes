import sys
import pathlib
import math

import cv2
import numpy as np
# from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.integrate import solve_ivp
import nrrd

'''
Structural Tensor code based on:

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

class ST(object):

    # assumes image is a floating-point numpy array
    def __init__(self, image):
        self.image = image
        self.lambda_u = None
        self.lambda_v = None
        self.vector_u = None
        self.vector_v = None
        self.isotropy = None
        self.linearity = None
        self.coherence = None


    def computeEigens(self):
        tif = self.image
        sigma0 = 1.  # value used by Hale
        sigma0 = 2.
        ksize = int(math.floor(.5+6*sigma0+1))
        hksize = ksize//2
        kernel = cv2.getGaussianKernel(ksize, sigma0)
        dkernel = kernel.copy()
        for i in range(ksize):
            x = i - hksize
            factor = x/(sigma0*sigma0)
            dkernel[i] *= factor
        # this gives a slightly wrong answer when tif is constant
        # (should give 0, but gives -1.5e-9)
        # gx = cv2.sepFilter2D(tif, -1, dkernel, kernel)
        # this is equivalent, and gives zero when it should
        gx = cv2.sepFilter2D(tif.transpose(), -1, kernel, dkernel).transpose()
        gy = cv2.sepFilter2D(tif, -1, kernel, dkernel)
        grad = np.concatenate((gx, gy)).reshape(2,gx.shape[0],gx.shape[1]).transpose(1,2,0)

        # gaussian blur
        # OpenCV function is several times faster than
        # numpy equivalent
        sigma1 = 8. # value used by Hale
        # sigma1 = 16.
        # gx2 = gaussian_filter(gx*gx, sigma1)
        gx2 = cv2.GaussianBlur(gx*gx, (0, 0), sigma1)
        # gy2 = gaussian_filter(gy*gy, sigma1)
        gy2 = cv2.GaussianBlur(gy*gy, (0, 0), sigma1)
        # gxy = gaussian_filter(gx*gy, sigma1)
        gxy = cv2.GaussianBlur(gx*gy, (0, 0), sigma1)

        gar = np.array(((gx2,gxy),(gxy,gy2)))
        gar = gar.transpose(2,3,0,1)

        # Explicitly calculate eigenvalue of 2x2 matrix instead of
        # using the numpy linalg eigh function; 
        # the explicit method is about 10 times faster.
        # See https://www.soest.hawaii.edu/martel/Courses/GG303/Eigenvectors.pdf
        # for a derivation
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

        # if lu is 0., set lu and lv to 1.; this will give
        # the correct values for isotropy (1.0) and linearity (0.0)
        # for this case

        lu0 = (lu==0)
        # print("lu==0", lu0.sum())
        lu[lu0] = 1.
        lv[lu0] = 1.

        isotropy = lv/lu
        linearity = (lu-lv)/lu
        coherence = ((lu-lv)/(lu+lv))**2

        # explicitly calculate normalized eigenvectors
        # eigenvector u
        vu = np.concatenate((gxy, lu-gx2)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
        vu[lu0,:] = 0.
        vulen = np.sqrt((vu*vu).sum(axis=2))
        vulen[vulen==0] = 1
        vu /= vulen[:,:,np.newaxis]
        # print("vu", vu.shape, vu.dtype)
        
        # eigenvector v
        vv = np.concatenate((gxy, lv-gx2)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
        vv[lu0,:] = 0.
        vvlen = np.sqrt((vv*vv).sum(axis=2))
        vvlen[vvlen==0] = 1
        vv /= vvlen[:,:,np.newaxis]
        
        self.lambda_u = lu
        self.lambda_v = lv
        self.vector_u = vu
        self.vector_v = vv
        self.grad = grad
        self.isotropy = isotropy
        self.linearity = linearity
        self.coherence = coherence

        self.lambda_u_interpolator = ST.createInterpolator(self.lambda_u)
        self.lambda_v_interpolator = ST.createInterpolator(self.lambda_v)
        self.vector_u_interpolator = ST.createInterpolator(self.vector_u)
        self.vector_v_interpolator = ST.createInterpolator(self.vector_v)
        self.grad_interpolator = ST.createInterpolator(self.grad)
        self.isotropy_interpolator = ST.createInterpolator(self.isotropy)
        self.linearity_interpolator = ST.createInterpolator(self.linearity)
        self.coherence_interpolator = ST.createInterpolator(self.coherence)

    def create_vel_func(self, xy, sign, nudge=0):
        x0 = np.array((xy))
        # vv0 = self.vector_v_interpolator((x0[::-1]))[0]
        # print(x0, vv0)
        def vf(t, y):
            # print("y", y.shape, y.dtype)
            # print(t,y)
            # vv stores vector at x,y in vv[y,x], but the
            # vector itself is returned in x,y order
            yr = y[::-1]
            vv = self.vector_v_interpolator((yr))[0]
            grad = self.grad_interpolator((yr))[0]
            # print(t,y,vv,grad)
            # print("vv", vv.shape, vv.dtype)
            # print(vv)
            # vvx = vv[0]
            # yx = y[1]
            # if vvx * (yx-x0) < 0:
            if t==0:
                if vv[0]*sign < 0:
                    vv *= -1
            elif (vv*(y-x0)).sum() < 0:
                vv *= -1
            # print(vv)
            # vv += 5*grad
            vv += nudge*grad
            return vv
        return vf

    # the idea of using Runge-Kutta (which solve_ivp uses)
    # was suggested by @TizzyTom
    def call_ivp(self, xy, sign, nudge=0):
        vel_func = self.create_vel_func(xy, sign, nudge)
        tmax = 400
        tmax = 500
        tsteps = np.linspace(0,tmax,tmax//10)
        # sol = solve_ivp(fun=vel_func, t_span=[0,tmax], t_eval=tsteps, atol=1.e-8, rtol=1.e-5, y0=xy)
        # sol = solve_ivp(fun=vel_func, t_span=[0,tmax], t_eval=tsteps, atol=1.e-8, y0=xy)
        # for testing
        # sol = solve_ivp(fun=vel_func, t_span=[0,tmax], rtol=1.e-5, atol=1.e-8, y0=xy)
        sol = solve_ivp(fun=vel_func, t_span=[0,tmax], max_step=2., y0=xy)
        # print("solution")
        # print(sol)
        return (sol.status, sol.y, sol.t, sol.nfev)

    # ix is rounding pixel position, ix0 is shift before rounding
    # output is transposed relative to y from call_ivp
    def evenly_spaced_result(self, xy, ix0, ix, sign, nudge=0):
        status, y, t, nfev = self.call_ivp(xy, sign, nudge)
        if status != 0:
            print("status", status)
            return None
        y = y.transpose()
        # print("y 1", y.shape)
        dx = np.diff(y[:,0])
        bad = (sign*dx <= 0)
        ibad = np.argmax(bad)
        if bad[0]:
            print("bad0")
            print(y)
            return None
        if ibad > 0:
            y = y[:ibad+1,:]
        # print("y 2", y.shape)

        xs = sign*y[:,0]
        ys = y[:,1]
        cs = CubicSpline(xs, ys)

        six0 = sign*ix0
        xmin = math.ceil((xs[0]+six0)/ix)*ix - six0
        xmax = math.floor((xs[-1]+six0)/ix)*ix - six0
        xrange = np.arange(xmin,xmax,ix)
        csys = cs(xrange)
        esy = np.stack((sign*xrange,csys), axis=1)
        return esy


    def sparse_result(self, xy, ix0, ix, sign, nudge=0):
        esy = self.evenly_spaced_result(xy, ix0, ix, sign, nudge)
        if esy is None:
            return None
        # print("esy", esy.shape)
        if len(esy) == 0:
            return None
        xs = sign*esy[:,0]
        ys = esy[:,1]
        # xmin = xs[0]
        # xmax = xs[-1]

        b = np.full(xs.shape, False)
        b[0] = True
        b[-1] = True
        tol = 1.
        # print("b",b.shape,b.dtype)
        # print("xrange",xrange.shape,xrange.dtype)
        while b.sum() < len(xs):
            idxs = b.nonzero()
            itx = xs[idxs]
            ity = ys[idxs]
            tcs = CubicSpline(itx,ity)
            oty = tcs(xs)
            diff = np.abs(oty-ys)
            # print("diff",diff.shape)
            midx = diff.argmax()
            # print("midx",midx)
            if b[midx]:
                break
            d = diff[midx]
            if d <= tol:
                break
            b[midx] = True

        idxs = b.nonzero()
        oy = esy[idxs]

        return oy


    # class function
    def createInterpolator(ar):
        interp = RegularGridInterpolator((np.arange(ar.shape[0]), np.arange(ar.shape[1])), ar, method='linear', bounds_error=False, fill_value=0.)
        return interp

    def saveEigens(self, fname):
        if self.lambda_u is None:
            print("saveEigens: eigenvalues not computed yet")
            return
        lu = self.lambda_u
        lv = self.lambda_v
        vu = self.vector_u
        vv = self.vector_v
        grad = self.grad
        # print(lu.shape, lu[np.newaxis,:,:].shape)
        # print(vu.shape)
        st_all = np.concatenate((lu[:,:,np.newaxis], lv[:,:,np.newaxis], vu, vv, grad), axis=2)
        # turn off the default gzip compression
        header = {"encoding": "raw",}
        nrrd.write(str(fname), st_all, header, index_order='C')

    def loadOrCreateEigens(self, fname):
        self.lambda_u = None
        print("loading eigens")
        self.loadEigens(fname)
        if self.lambda_u is None:
            print("calculating eigens")
            self.computeEigens()
            print("saving eigens")
            self.saveEigens(fname)

    def loadEigens(self, fname):
        try:
            data, data_header = nrrd.read(str(fname), index_order='C')
        except Exception as e:
            print("Error while loading",fname,e)
            return

        self.lambda_u = data[:,:,0]
        self.lambda_v = data[:,:,1]
        self.vector_u = data[:,:,2:4]
        self.vector_v = data[:,:,4:6]
        self.grad = data[:,:,6:8]
        lu = self.lambda_u
        lv = self.lambda_v
        self.isotropy = lv/lu
        self.linearity = (lu-lv)/lu
        self.coherence = ((lu-lv)/(lu+lv))**2

        # print("lambda_u", self.lambda_u.shape, self.lambda_u.dtype)
        # print("vector_u", self.vector_u.shape, self.vector_u.dtype)

        self.lambda_u_interpolator = ST.createInterpolator(self.lambda_u)
        self.lambda_v_interpolator = ST.createInterpolator(self.lambda_v)
        self.vector_u_interpolator = ST.createInterpolator(self.vector_u)
        self.vector_v_interpolator = ST.createInterpolator(self.vector_v)
        self.grad_interpolator = ST.createInterpolator(self.grad)
        self.isotropy_interpolator = ST.createInterpolator(self.isotropy)
        self.linearity_interpolator = ST.createInterpolator(self.linearity)
        self.coherence_interpolator = ST.createInterpolator(self.coherence)


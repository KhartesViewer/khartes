import sys
import pathlib
import math

import cv2
import numpy as np
import scipy
# from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator
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
        # return (sol.status, sol.y, sol.t, sol.nfev)
        if sol.status != 0:
            print("ivp status", sol.status)
            return None
        if sol.y is None:
            print("ivp y is None")
            return None
        return sol.y.transpose()

    def solve2d(self, xs, ys, constraints):
        # return ys.copy()

        # vv stores the vector at x,y in vv[y,x], but the
        # vector itself is returned in x,y order
        # yxs = np.stack((ys, xs), axis=1)
        # vvecs = self.vector_v_interpolator(yxs)
        # cohs = self.coherence_interpolator(yxs)

        # for each segment in xs,ys, find the midpoint
        mxs = .5*(xs[:-1]+xs[1:])
        # length (in x direction) of each segment
        lxs = xs[1:]-xs[:-1]
        mys = .5*(ys[:-1]+ys[1:])
        myxs = np.stack((mys, mxs), axis=1)
        # print("xs")
        # print(xs)
        # print("myxs")
        # print(myxs)
        # vv stores the vector at x,y in vv[y,x], but the
        # vector itself is returned in x,y order
        vvecs = self.vector_v_interpolator(myxs)
        # print("vvecs")
        # print(vvecs)
        cohs = self.coherence_interpolator(myxs)
        # cohs = self.linearity_interpolator(myxs)
        # cohs = np.full((myxs.shape[0]), 1.0, dtype=np.float64)
        grads = self.grad_interpolator(myxs)
        # make sure vvecs[:,0] is always > 0
        vvecs[vvecs[:,0] < 0] *= 1
        # nudge:
        # vvecs[:,1] += 5.*grads[:,1]
        vvzero = vvecs[:,0] == 0
        vslope = np.zeros((vvecs.shape[0]), dtype=np.float64)
        vslope[~vvzero] = vvecs[~vvzero,1]/vvecs[~vvzero,0]
        # print("vslope")
        # print(vslope)

        # compute pim1 = slope = vvecs[:,1]/vvecs[:,0]
        # compute dfidx = dy/dx from xs and ys
        # wim1 = cohs
        # print("cohs")
        # print(cohs)

        # xys = np.stack((xs, ys), axis=1)
        
        # Following the notation in Wu and Hale (2015):
        f = ys
        # print("f", f.shape)
        # print(f)
        # W = np.diag(cohs)
        G = np.diag(-1./lxs)
        # print(G.shape)
        G = np.append(G, np.zeros((G.shape[0]),dtype=np.float64)[:,np.newaxis], axis=1)
        # print(G.shape)
        G[:,1:] += np.diag(1./lxs)
        # print("G", G.shape)
        # print(G)
        # G = np.concatenate((G,G,G), axis=0)
        G = np.concatenate((G,G), axis=0)
        # print("G", G.shape)
        # print(G)

        # print("cohs", cohs.shape)
        rweight = .1
        rws = np.full(cohs.shape, rweight, dtype=np.float64)
        # print("rws", rws.shape)
        rvs = np.full(vslope.shape, 0., dtype=np.float64)
        # print("rvs", rvs.shape)
        # gweight = 0.1
        # gws = np.full(cohs.shape, gweight, dtype=np.float64)
        # # print("gws", gws.shape)
        # gvs = grads[:,1]
        # # print("gvs", gvs.shape)

        # W = np.diag(np.concatenate((cohs,rws,gws)))
        W = np.diag(np.concatenate((cohs,rws)))
        # print("W", W.shape)
        # v = np.concatenate((vslope, rvs, gvs))
        v = np.concatenate((vslope, rvs))
        # print("v", v.shape)

        wgtw = (W@G).T @ W
        A = wgtw@G
        # print("A", A.shape)
        # print(A)
        # print(A.sum(axis=1))
        b = wgtw@v
        # print("b", b.shape)
        # print(b)
        cons = np.array(constraints, dtype=np.float64)
        cidxs = cons[:,0].astype(np.int64)
        cys = cons[:,1]

        Z = np.identity(A.shape[0])
        # Z = Z[:,1:]
        Z = np.delete(Z, cidxs, axis=1)
        # print("Z", Z.shape)
        # print(np.linalg.inv(Z.T@A@Z))
        f0 = np.zeros(f.shape[0], dtype=np.float64)
        f0[cidxs] = cys
        # print("f0", f0.shape)
        # print(f0)

        try:
            p = np.linalg.inv(Z.T@A@Z)@Z.T@(b-A@f0)
        except:
            print("Singular matrix!")
            return ys.copy()

        newf = f0 + Z@p

        # print(np.linalg.inv(A))
        # print(scipy.linalg.inv(A))

        # newf = np.linalg.inv(A)@b
        # newf = np.linalg.inv(Z.T@A@Z)@b
        # print("newf")
        # print(newf)
        newys = newf + ys[0] - newf[0]
        return newys
        # return ys.copy()

    def interp2d(self, xy1, xy2, nudge=0.):
        print("interp2d", xy1, xy2)
        oxy1, oxy2 = xy1, xy2
        if xy1[0] > xy2[0]:
            oxy1,oxy2 = oxy2,oxy1
        ynudge = 0.
        x1 = oxy1[0]
        y1 = oxy1[1] + ynudge
        x2 = oxy2[0]
        y2 = oxy2[1] + ynudge
        nx = int(x2-x1)+1
        xs = np.linspace(x1,x2,nx, dtype=np.float64)
        # create list of constraints (each is (index, y))
        constraints = ((0, y1), (nx-1, y2))
        cons = np.array(constraints, dtype=np.float64)
        cidxs = cons[:,0].astype(np.int64)
        ncidxs=cidxs.shape[0]
        cys = cons[:,1]
        # create initial ys (line interpolating xy1 and xy2)
        y0s = np.linspace(y1,y2,nx, dtype=np.float64)

        # for each segment in xs,ys, find the midpoint
        mxs = .5*(xs[:-1]+xs[1:])
        # length (in x direction) of each segment
        lxs = xs[1:]-xs[:-1]
        # nx = xs.shape[0]
        ndx = nx-1
        # fs = np.zeros((2*ndx+cidxs.shape[0],))
        # print("fs", fs.shape)
        vslope = np.zeros((ndx), dtype=np.float64)
        rweight = .1
        # rweight = nudge
        # rweight = .001
        # nudge = 1.
        self.global_cohs = None
        # self.global_dcohs_dy = None
        def fun(iys):
            # print("iys", iys)
            oys = iys[cidxs]
            ys = iys.copy()
            ys[cidxs] = cys
            # print("ys", ys.shape)
            # print(ys)
            mys = .5*(ys[:-1]+ys[1:])
            myxs = np.stack((mys, mxs), axis=1)
            vvecs = self.vector_v_interpolator(myxs)
            # grads = self.grad_interpolator(myxs)
            # vvecs[:,1] += nudge*grads[:,1]
            cohs = self.coherence_interpolator(myxs)
            # cohs = self.linearity_interpolator(myxs)
            # myxsd = myxs.copy()
            # dyc = .01
            # myxsd[:,0] += dyc
            # cohsd = self.linearity_interpolator(myxsd)
            # TODO: Testing only!!
            # cohs = np.linspace(.8,.8+(ndx+1)*.01,ndx, dtype=np.float64)
            # self.global_dcohs_dy = (cohsd-cohs)/dyc
            # print("cohs", cohs)
            # make sure vvecs[:,0] is always > 0
            vvecs[vvecs[:,0] < 0] *= 1
            # nudge:
            # vvecs[:,1] += 5.*grads[:,1]
            vvzero = vvecs[:,0] == 0
            vslope[:] = 0.
            # vslope = np.zeros((vvecs.shape[0]), dtype=np.float64)
            vslope[~vvzero] = vvecs[~vvzero,1]/vvecs[~vvzero,0]
            # aslope = np.abs(vslope)**2
            # aslopegt1 = aslope > 1.
            # cohs[aslopegt1] /= aslope[aslopegt1]
            self.global_cohs = cohs

            yslope = (ys[1:]-ys[:-1])/lxs
            # fs = np.array((2*ndx+cidxs.shape[0],))
            # print("fs fun", fs.shape)
            fs = np.zeros((2*ndx+ncidxs,))
            fs[:ndx] = (vslope-yslope)*cohs
            fs[ndx:2*ndx] = rweight*yslope
            fs[2*ndx:] = cys-oys
            # print("fs", fs.shape)
            # print(fs)
            # print("fs", fs)
            # print()
            return fs

        def matvec(idys):
            # print("idys", idys.shape)
            # print("idys", idys)
            idys = idys.flatten()
            dys = idys.copy()
            odys = dys[cidxs]
            # print(dys)
            dys[cidxs] = 0.
            # print(cidxs, odys)
            dfs = np.zeros((2*ndx+ncidxs,))
            # print("dfs", dfs.shape)
            dfs[:ndx] = dys[:-1]
            dfs[:ndx] += -dys[1:]
            dfs[:ndx] *= self.global_cohs
            # dfs[:ndx] += .5*dys[:-1]*self.global_dcohs_dy
            # dfs[:ndx] += .5*dys[1:]*self.global_dcohs_dy
            dfs[:ndx] /= lxs
            dfs[ndx:2*ndx] = -rweight*dys[:-1]
            dfs[ndx:2*ndx] += rweight*dys[1:]
            dfs[ndx:2*ndx] /= lxs
            dfs[2*ndx:] = -odys
            # print("dfs", dfs)
            # print()
            return dfs

        def rmatvec(idfs):
            # print("idfs", idfs)
            dfs = idfs
            dys = np.zeros((nx,))
            cohs = self.global_cohs
            # dcohs_dy = self.global_dcohs_dy
            dys[:ndx] = dfs[:ndx]*cohs/lxs
            # dys[:ndx] += dcohs_dy
            dys[1:nx] += -dfs[:ndx]*cohs/lxs
            # dys[1:nx] += dcohs_dy
            dys[:ndx] += -rweight*dfs[ndx:2*ndx]/lxs
            dys[1:nx] += rweight*dfs[ndx:2*ndx]/lxs
            # Notice: not +=
            dys[cidxs] = -dfs[2*ndx:]
            # print("dys", dys)
            # print()
            return dys


        '''
        def matmat(idys):
            dys = idys.copy()
            odys = dys[cidxs,:]
            # print(dys)
            dys[cidxs,:] = 0.
            # print(cidxs, odys)
            dfs = np.zeros((2*ndx+ncidxs,dys.shape[1]))
            dfs[:ndx,:] = dys[:-1,:]
            dfs[:ndx,:] += -dys[1:,:]
            dfs[:ndx,:] *= self.global_cohs
            dfs[ndx:2*ndx,:] = -rweight*dys[:-1,:]
            dfs[ndx:2*ndx,:] += rweight*dys[1:,:]
            dfs[2*ndx:,:] = -odys
            return dfs
        '''

        def jac(ys):
            return LinearOperator((2*ndx+ncidxs, nx), matvec=matvec, rmatvec=rmatvec)

        '''
        f0 = fun(y0s)
        print(self.global_cohs)
        # print(self.global_dcohs_dy)
        # lo = jac(y0s)
        for i in range(nx):
            # for i in range(2):
            y1 = y0s.copy()
            y1[i] += .001
            f1 = fun(y1)
            print(f1-f0)
            # print(matvec(y1-y0s))
            print(matvec(y1-y0s))
            print()
        '''

        '''
        f0 = fun(y0s)
        for i in range(nx):
            y1 = np.zeros((y0s.shape))
            y1[i] += .001
            print(matvec(y1))
        print()

        for i in range(2*ndx+ncidxs):
            f1 = np.zeros((2*ndx+ncidxs))
            f1[i] += .001
            print(rmatvec(f1))
        '''

        r = least_squares(fun, y0s, jac=jac)

        # if self.global_cohs is not None:
        #     print("global_cohs")
        #     print(self.global_cohs)
        print("r", r.status, r.nfev, r.njev, r.cost)
        # print("r.grad")
        # print(r.grad)
        # print("r.x")
        # print(r.x)
        # return r.x
        xys = np.stack((xs, r.x), axis=1)
        print("xys", xys.shape)
        # print(xys)
        return xys

    def interp2dWH(self, xy1, xy2):
        print("interp2dWH", xy1, xy2)
        # order of xy1, xy2 shouldn't matter, except when creating xs
        # create xs (ordered list of x values; y is to be solved for)
        epsilon = .01
        oxy1, oxy2 = xy1, xy2
        if xy1[0] > xy2[0]:
            oxy1,oxy2 = oxy2,oxy1
        x1 = oxy1[0]
        y1 = oxy1[1]
        x2 = oxy2[0]
        y2 = oxy2[1]
        nx = int(x2-x1)+1
        xs = np.linspace(x1,x2,nx, dtype=np.float64)
        # create list of constraints (each is (index, y))
        constraints = ((0, y1), (nx-1, y2))
        # create initial ys (line interpolating xy1 and xy2)
        ys = np.linspace(y1,y2,nx, dtype=np.float64)
        prev_dys = -1.
        min_dys = -1.
        min_ys = None
        for _ in range(20):
            # call solver: given xs, ys, constraints, return new ys based on
            # solving a linear equation
            prev_ys = ys.copy()
            ys = self.solve2d(xs, ys, constraints)
            # print("new ys", ys)
            # keep calling solver until solution stabilizes
            # stop condition: average absolute update is less than epsilon
            dys = np.abs(ys-prev_ys)
            avg_dys = dys.sum()/nx
            print("avg_dys", avg_dys)
            if min_dys < 0. or avg_dys < min_dys:
                min_dys = avg_dys
                min_ys = ys.copy()
            if avg_dys < epsilon:
                break
            # if prev_dys > 0 and prev_dys < avg_dys:
            #     ys = prev_ys
            #     break
            prev_dys = avg_dys
        # xys = np.stack((xs, ys), axis=1)
        xys = np.stack((xs, min_ys), axis=1)
        print("xys", xys.shape)
        # print(xys)
        return xys

    # ix is rounding pixel position, ix0 is shift before rounding
    # output is transposed relative to y from call_ivp
    # def evenly_spaced_result(self, xy, ix0, ix, sign, nudge=0):
    def evenly_spaced_result(self, y, ix0, ix):
        # status, y, t, nfev = self.call_ivp(xy, sign, nudge)
        # if status != 0:
        #     print("status", status)
        #     return None
        # y = y.transpose()
        # print("y 1", y.shape)
        if y is None:
            return None
        if len(y) < 2:
            print("too few points", len(y))
            return None
        dx = np.diff(y[:,0])
        sign = 1
        if dx[0] < 0:
            sign = -1
        bad = (sign*dx <= 0)
        ibad = np.argmax(bad)
        # can't happen now; sign is based on dx[0]
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
        # esy = np.stack((sign*xrange,csys), axis=1)
        esy = np.stack((sign*xrange,csys), axis=1)
        return esy


    # def sparse_result(self, xy, ix0, ix, sign, nudge=0):
    def sparse_result(self, y, ix0, ix):
        # esy = self.evenly_spaced_result(xy, ix0, ix, sign, nudge)
        esy = self.evenly_spaced_result(y, ix0, ix)
        if esy is None:
            return None
        # print("esy", esy.shape)
        if len(esy) == 0:
            return None
        # xs = sign*esy[:,0]
        xs = esy[:,0].copy()
        if xs[0] > xs[-1]:
            xs *= -1
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
        # print("oy", oy)

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


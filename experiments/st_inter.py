import sys
import pathlib
import math

from PyQt5.QtWidgets import (
        QApplication,
        QGridLayout,
        QLabel,
        QMainWindow,
        QStatusBar,
        QVBoxLayout,
        QWidget,
        )
from PyQt5.QtCore import (
        QPoint,
        QSize,
        Qt,
        )
from PyQt5.QtGui import (
        QCursor,
        QImage,
        QPixmap,
        )

import cv2
import numpy as np
# from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.integrate import solve_ivp
import nrrd

class MainWindow(QMainWindow):

    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.app = app
        self.setMinimumSize(QSize(750,600))
        self.already_shown = False
        self.st = None
        grid = QGridLayout()
        widget = QWidget()
        widget.setLayout(grid)
        self.setCentralWidget(widget)
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.viewer = ImageViewer(self)
        # self.setCentralWidget(self.viewer)
        grid.addWidget(self.viewer, 0, 0)
        self.viewer.setDefaults()

        tifname = r"C:\Vesuvius\scroll 1 2000-2030\02015.tif"
        outdir = pathlib.Path(r"C:\Vesuvius\Projects\tnorm")
        nrrdname = outdir / '02015.nrrd'

        print("loading tif")
        self.viewer.loadTIFF(tifname)
        self.st = ST(self.viewer.image)
        '''
        compute_eigens = False
        if compute_eigens:
            print("calculating st")
            self.st.computeEigens()
            print("saving st")
            self.st.saveEigens(nrrdname)
        else:
            print("loading eigens")
            self.st.loadEigens(nrrdname)
        '''
        self.st.loadOrCreateEigens(nrrdname)

    def setStatusText(self, txt):
        self.status_bar.showMessage(txt)

    def showEvent(self, e):
        # print("show event")
        if self.already_shown:
            return
        self.viewer.setDefaults()
        self.viewer.drawImage()
        self.already_shown = True

    def resizeEvent(self, e):
        # print("resize event")
        self.viewer.drawImage()

    def keyPressEvent(self, e):
        self.viewer.keyPressEvent(e)

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
        # self.computeEigens(image)

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

    # ix is rounding pixel position
    # output is transposed relative to y from call_ivp
    def evenly_spaced_result(self, xy, ix, sign, nudge=0):
        status, y, t, nfev = self.call_ivp(xy, sign, nudge)
        if status != 0:
            return None
        # print("y 1", y.shape)
        y = y.transpose()
        dx = np.diff(y[:,0])
        bad = (sign*dx <= 0)
        ibad = np.argmax(bad)
        if bad[0]:
            # print("bad0")
            return None
        if ibad > 0:
            y = y[:ibad+1,:]
        # print("y 2", y.shape)

        xs = sign*y[:,0]
        ys = y[:,1]
        cs = CubicSpline(xs, ys)

        xmin = math.ceil(xs[0]/ix)*ix
        xmax = math.floor(xs[-1]/ix)*ix
        xrange = np.arange(xmin,xmax,ix)
        csys = cs(xrange)
        esy = np.stack((sign*xrange,csys), axis=1)
        return esy


    def sparse_result(self, xy, ix, sign, nudge=0):
        esy = self.evenly_spaced_result(xy, ix, sign, nudge)
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

    '''
    # Tried to write hand-rolled interpolator; still doesn't
    # work, despite complexity upon complexity
    def createInterpolator(ar):
        def interp(pts):
            i0 = np.floor(pts[...,0]).astype(np.int32)
            i1 = np.floor(pts[...,1]).astype(np.int32)
            f0 = pts[...,0]-i0
            f1 = pts[...,1]-i1

            inbounds = (pts[...,0]>=0) & (pts[...,0]<ar.shape[0]) & (pts[...,1]>=0) & (pts[...,1]<ar.shape[1])
            print("i0", i0.shape, i0.dtype)
            i0[~inbounds] = 0
            i1[~inbounds] = 0
            print("i0", i0.shape, i0.dtype)
            
            # i0n = i0[...,np.newaxis]
            # i1n = i1[...,np.newaxis]
            if len(ar.shape) > len(f0.shape):
                f0n = f0[...,np.newaxis]
                f1n = f1[...,np.newaxis]
            else:
                f0n = f0
                f1n = f1
            print("ar", ar.shape, ar.dtype)
            print("ar[i0,i1]", ar[i0,i1].shape)
            print("pts", pts.shape, pts.dtype)
            print("f0", f0.shape, f0.dtype)
            print("f0n", f0n.shape, f0n.dtype)
            res = ((1.-f0n)*((1.-f1n)*ar[i0,i1] + f1n*ar[i0,i1+1])+
            f0n*((1.-f1n)*ar[i0+1,i1] + f1n*ar[i0+1,i1+1]))
            vals = np.zeros((res.shape), dtype=pts.dtype)
            vals[inbounds] = res[inbounds]

            return vals

        return interp
    '''

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

class ImageViewer(QLabel):

    def __init__(self, main_window):
        super(ImageViewer, self).__init__()
        self.setMouseTracking(True)
        self.main_window = main_window
        self.image = None
        self.zoom = 1.
        self.center = (0,0)
        self.bar0 = (0,0)
        self.mouse_start_point = QPoint()
        self.center_start_point = None
        self.is_panning = False
        self.dip_bars_visible = True
        self.rays = []

    def mousePressEvent(self, e):
        if self.image is None:
            return
        if e.button() | Qt.LeftButton:
            modifiers = QApplication.keyboardModifiers()
            wpos = e.localPos()
            wxy = (wpos.x(), wpos.y())
            ixy = self.wxyToIxy(wxy)

            self.mouse_start_point = wpos
            self.center_start_point = self.center
            # print("ixys", ixy)
            self.is_panning = True

    def mouseMoveEvent(self, e):
        if self.image is None:
            return
        wpos = e.localPos()
        wxy = (wpos.x(), wpos.y())
        ixy = self.wxyToIxy(wxy)
        self.setStatusTextFromMousePosition()
        if self.is_panning:
            # print(wpos, self.mouse_start_point)
            delta = wpos - self.mouse_start_point
            dx,dy = delta.x(), delta.y()
            z = self.zoom
            # cx, cy = self.center
            six,siy = self.center_start_point
            self.center = (six-dx/z, siy-dy/z)
            self.drawImage()

    def mouseReleaseEvent(self, e):
        if e.button() | Qt.LeftButton:
            self.mouse_start_point = QPoint()
            self.center_start_point = None
            self.is_panning = False

    def leaveEvent(self, e):
        if self.image is None:
            return
        self.main_window.setStatusText("")

    def wheelEvent(self, e):
        if self.image is None:
            return
        self.setStatusTextFromMousePosition()
        d = e.angleDelta().y()
        z = self.zoom
        z *= 1.001**d
        self.setZoom(z)
        self.drawImage()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_T:
            wxy = self.mouseXy()
            ixy = self.wxyToIxy(wxy)
            print("t at",ixy)
            st = self.getST()
            if st is None:
                return
            # self.rays = []
            # for nudge in (0, 2., 5.):
            for nudge in (5.,):
                for sign in (1,-1):
                    '''
                    status,y,t,nfev = st.call_ivp(ixy, sign, nudge)
                    print(sign, "status", status, nfev)
                    # status 0 means success
                    if status == 0:
                        self.rays.append(y)
                    '''
                    y = st.sparse_result(ixy, 5, sign, nudge)
                    if y is not None:
                        self.rays.append(y)

            if len(self.rays) > 0:
                self.drawImage()
        elif e.key() == Qt.Key_C:
            if len(self.rays) == 0:
                return
            self.rays = []
            self.drawImage()
        elif e.key() == Qt.Key_V:
            self.dip_bars_visible = not self.dip_bars_visible
            self.drawImage()

    def getST(self):
        return self.main_window.st

    def mouseXy(self):
        pt = self.mapFromGlobal(QCursor.pos())
        return (pt.x(), pt.y())

    def setStatusTextFromMousePosition(self):
        wxy = self.mouseXy()
        ixy = self.wxyToIxy(wxy)
        self.setStatusText(ixy)

    def setStatusText(self, ixy):
        if self.image is None:
            return
        labels = ["X", "Y"]
        stxt = ""
        for i in (0,1):
            f = ixy[i]
            dtxt = "%.2f"%f
            if f < 0 or f > self.image.shape[1-i]-1:
                dtxt = "("+dtxt+")"
            stxt += "%s "%dtxt
        self.main_window.setStatusText(stxt)

    def ixyToWxy(self, ixy):
        ix,iy = ixy
        cx,cy = self.center
        z = self.zoom
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        wx = int(z*(ix-cx)) + wcx
        wy = int(z*(iy-cy)) + wcy
        return (wx,wy)

    def ixysToWxys(self, ixys):
        ww,wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        c = self.center
        z = self.zoom
        dxys = ixys.copy()
        dxys -= c
        dxys *= z
        dxys = dxys.astype(np.int32)
        dxys[...,0] += wcx
        dxys[...,1] += wcy
        return dxys

    def wxyToIxy(self, wxy):
        wx,wy = wxy
        ww,wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        dx,dy = wx-wcx, wy-wcy
        cx,cy = self.center
        z = self.zoom
        ix = cx + dx/z
        iy = cy + dy/z
        return (ix, iy)

    def wxysToIxys(self, wxys):
        ww,wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2

        # dxys = wx-wcx, wy-wcy
        dxys = wxys.copy()
        dxys[...,0] -= wcx
        dxys[...,1] -= wcy
        cx,cy = self.center
        z = self.zoom
        ixys = np.zeros(wxys.shape)
        ixys[...,0] = cx + dxys[...,0]/z
        ixys[...,1] = cy + dxys[...,1]/z
        return ixys

    def setDefaults(self):
        if self.image is None:
            return
        ww = self.width()
        wh = self.height()
        # print("ww,wh",ww,wh)
        iw = self.image.shape[1]
        ih = self.image.shape[0]
        self.center = (iw//2, ih//2)
        zw = ww/iw
        zh = wh/ih
        zoom = min(zw, zh)
        self.setZoom(zoom)
        # self.bar0 = self.center
        print("center",self.center[0],self.center[1],"zoom",self.zoom)

    def setZoom(self, zoom):
        # TODO: set min, max zoom
        prev = self.zoom
        self.zoom = zoom
        if prev != 0:
            bw,bh = self.bar0
            cw,ch = self.center
            # print(self.bar0, self.center)
            bw -= cw
            bh -= ch
            bw /= zoom/prev
            bh /= zoom/prev
            self.bar0 = (bw+cw, bh+ch)

    # class function
    def rectIntersection(ra, rb):
        (ax1, ay1, ax2, ay2) = ra
        (bx1, by1, bx2, by2) = rb
        # print(ra, rb)
        x1 = max(min(ax1, ax2), min(bx1, bx2))
        y1 = max(min(ay1, ay2), min(by1, by2))
        x2 = min(max(ax1, ax2), max(bx1, bx2))
        y2 = min(max(ay1, ay2), max(by1, by2))
        if (x1<x2) and (y1<y2):
            r = (x1, y1, x2, y2)
            # print(r)
            return r

    def loadTIFF(self, fname):
        try:
            image = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED).astype(np.float64)
            image /= 65535.
        except Exception as e:
            print("Error while loading",fname,e)
            return
        # clipping suggested by @TizzyTom
        # clip = 36000./65535.
        # image[image>clip] = clip
        self.image = image
        self.setDefaults()
        # print("loadTIFF")
        self.drawImage()

    def drawImage(self):
        if self.image is None:
            return
        iw = self.image.shape[1]
        ih = self.image.shape[0]
        z = self.zoom
        # zoomed image width, height:
        ziw = max(int(z*iw), 1)
        zih = max(int(z*ih), 1)
        # viewing window width, height:
        ww = self.width()
        wh = self.height()
        # print("di ww,wh",ww,wh)
        # viewing window half width
        whw = ww//2
        whh = wh//2
        cx,cy = self.center

        # Pasting zoomed data slice into viewing-area array, taking
        # panning into account.
        # Need to calculate the interesection
        # of the two rectangles: 1) the panned and zoomed slice, and 2) the
        # viewing window, before pasting
        ax1 = int(whw-z*cx)
        ay1 = int(whh-z*cy)
        ax2 = ax1+ziw
        ay2 = ay1+zih
        bx1 = 0
        by1 = 0
        bx2 = ww
        by2 = wh
        ri = ImageViewer.rectIntersection((ax1,ay1,ax2,ay2), (bx1,by1,bx2,by2))
        outrgb = np.zeros((wh,ww,3), dtype=np.uint8)
        alpha = .75
        if ri is not None:
            (x1,y1,x2,y2) = ri
            # zoomed data slice
            x1s = int((x1-ax1)/z)
            y1s = int((y1-ay1)/z)
            x2s = int((x2-ax1)/z)
            y2s = int((y2-ay1)/z)
            # print(sw,sh,ww,wh)
            # print(x1,y1,x2,y2)
            # print(x1s,y1s,x2s,y2s)
            zslc = cv2.resize(self.image[y1s:y2s,x1s:x2s], (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
            outrgb[y1:y2, x1:x2, :] = (zslc*alpha*255)[:,:,np.newaxis].astype(np.uint8)

        st = self.main_window.st
        if st is not None and self.dip_bars_visible:
            dh = 15
            '''
            w0i,h0i = self.wxyToIxy((dh,dh))
            dhi = 2*dh/self.zoom
            # print("a",w0i,h0i)
            w0i = int(math.floor(w0i/dhi))*dhi
            h0i = int(math.floor(h0i/dhi))*dhi
            # print("b",w0i,h0i)
            w0,h0 = self.ixyToWxy((w0i,h0i))
            # print("c",w0,h0)
            w0 -= dh
            h0 -= dh
            '''
            w0i,h0i = self.wxyToIxy((0,0))
            w0i -= self.bar0[0]
            h0i -= self.bar0[1]
            dhi = 2*dh/self.zoom
            w0i = int(math.floor(w0i/dhi))*dhi
            h0i = int(math.floor(h0i/dhi))*dhi
            w0i += self.bar0[0]
            h0i += self.bar0[1]
            w0,h0 = self.ixyToWxy((w0i,h0i))
            dpw = np.mgrid[h0:wh:2*dh, w0:ww:2*dh].transpose(1,2,0)
            # switch from y,x to x,y coordinates
            dpw = dpw[:,:,::-1]
            # print ("dpw", dpw.shape, dpw.dtype, dpw[0,5])
            dpi = self.wxysToIxys(dpw)
            # interpolators expect y,x ordering
            dpir = dpi[:,:,::-1]
            # print ("dpi", dpi.shape, dpi.dtype, dpi[0,5])
            vvs = st.vector_v_interpolator(dpir)
            # print("vvs", vvs.shape, vvs.dtype, vvs[0,5])
            coherence = st.coherence_interpolator(dpir)
            # print("coherence", coherence.shape, coherence.dtype, coherence[0,5])
            linelen = 25.
            lvecs = linelen*vvs*coherence[:,:,np.newaxis]
            # print("lvecs", lvecs.shape, lvecs.dtype, lvecs[5,5])
            x0 = dpw-lvecs
            x1 = dpw+lvecs

            lines = np.concatenate((x0,x1), axis=2)
            lines = lines.reshape(-1,1,2,2).astype(np.int32)
            cv2.polylines(outrgb, lines, False, (255,255,0), 1)

            points = dpw.reshape(-1,1,1,2).astype(np.int32)
            cv2.polylines(outrgb, points, True, (0,255,255), 3)

        for i,ray in enumerate(self.rays):
            # print("ray", self.ray)
            points = self.ixysToWxys(ray)
            # print("points", points)
            points = points.reshape(-1,1,1,2)
            # colors = ((255,255,255), (255,0,255), (0,255,0))
            colors = ((0,255,0),)
            color = colors[(i//2)%len(colors)]

            cv2.polylines(outrgb, points, True, color, 2)
            cv2.circle(outrgb, points[0,0,0], 3, (255,0,255), -1)

        bytesperline = 3*outrgb.shape[1]
        # print(outrgb.shape, outrgb.dtype)
        qimg = QImage(outrgb, outrgb.shape[1], outrgb.shape[0],
                      bytesperline, QImage.Format_RGB888)
        # print("created qimg")
        pixmap = QPixmap.fromImage(qimg)
        # print("created pixmap")
        self.setPixmap(pixmap)
        # print("set pixmap")

class Tinter():

    def __init__(self, app):
        window = MainWindow(app)
        self.app = app
        self.window = window
        window.show()

app = QApplication(sys.argv)

tinter = Tinter(app)
app.exec()

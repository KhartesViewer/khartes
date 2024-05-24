import math
import json
import numpy as np

from pathlib import Path
from collections import deque
from scipy.spatial import Delaunay
import cv2
'''
from scipy.interpolate import (
        LinearNDInterpolator,
        NearestNDInterpolator,
        )
'''

from utils import Utils
from base_fragment import BaseFragment, BaseFragmentView
from fragment import Fragment, FragmentView

from PyQt5.QtGui import QColor

class TrglFragment(BaseFragment):
    def __init__(self, name):
        super(TrglFragment, self).__init__(name)
        self.gpoints = np.zeros((0,3), dtype=np.float32)
        self.gtpoints = np.zeros((0,2), dtype=np.float32)
        self.trgls = np.zeros((0,3), dtype=np.int32)
        self.direction = 0
        '''
        self.name = name
        self.trgls = np.zeros((0,3), dtype=np.int32)
        self.create = Utils.timestamp()
        self.modified = Utils.timestamp()
        '''

    # class function
    # expected to return a list of fragments, but always
    # returns only one
    def load(obj_file):
        print("loading obj file", obj_file)
        pname = Path(obj_file)
        try:
            fd = pname.open("r")
        except:
            return None

        name = pname.stem
        
        vrtl = []
        tvrtl = []
        trgl = []
        
        created = ""
        frag_name = ""
        for line in fd:
            line = line.strip()
            words = line.split()
            if words[0][0] == '#':
                if len(words) > 2: 
                    if words[1] == "Created:":
                        created = words[2]
                    if words[1] == "Name:":
                        frag_name = words[2]
            elif words[0] == 'v':
                # len is 7 if the vrt has color attached
                # (color is ignored)
                if len(words) == 4 or len(words) == 7:
                    vrtl.append([float(w) for w in words[1:4]])
            elif words[0] == 'vt':
                if len(words) == 3:
                    tvrtl.append([float(w) for w in words[1:]])
            elif words[0] == 'f':
                if len(words) == 4:
                    # implicit assumption that v == vt
                    trgl.append([int(w.split('/')[0])-1 for w in words[1:]])
        print("tf obj reader", len(vrtl), len(tvrtl), len(trgl))
        
        if frag_name == "":
        #     frag_name = name.replace("_",":").replace("p",".")
            frag_name = name
        trgl_frag = TrglFragment(frag_name)
        trgl_frag.gpoints = np.array(vrtl, dtype=np.float32)
        trgl_frag.gtpoints = np.array(tvrtl, dtype=np.float32)
        if len(trgl) > 0:
            trgl_frag.trgls = np.array(trgl, dtype=np.int32)
        else:
            trgl_frag.trgls = np.zeros((0,3), dtype=np.int32)
        if created == "":
            ts = Utils.vcToTimestamp(name)
            if ts is not None:
                created = ts
        if created != "":
            trgl_frag.created = created
        trgl_frag.params = {}
        
        mname = pname.with_suffix(".mtl")
        fd = None
        color = None
        try:
            fd = mname.open("r")
        except:
            print("failed to open mtl file",mname.name)
            pass

        if fd is not None:
            for line in fd:
                words = line.split()
                # print("words[0]", words[0])
                if len(words) == 4 and words[0] == "Kd":
                    try:
                        # print("words", words)
                        r = float(words[1])
                        g = float(words[2])
                        b = float(words[3])
                    except:
                        continue
                    # print("rgb", r,g,b)
                    color = QColor.fromRgbF(r,g,b)
                    break

        if color is None:
            color = Utils.getNextColor()
        trgl_frag.setColor(color, no_notify=True)
        trgl_frag.valid = True
        trgl_frag.neighbors = BaseFragment.findNeighbors(trgl_frag.trgls)
        print(trgl_frag.name, trgl_frag.color.name(), trgl_frag.gpoints.shape, trgl_frag.gtpoints.shape, trgl_frag.trgls.shape)
        # print("tindexes", BaseFragment.trglsAroundPoint(100, trgl_frag.trgls))

        return [trgl_frag]

    def createView(self, project_view):
        return TrglFragmentView(project_view, self)

    def createCopy(self, name):
        frag = TrglFragment(name)
        frag.setColor(self.color, no_notify=True)
        frag.gpoints = np.copy(self.gpoints)
        frag.gtpoints = np.copy(self.gtpoints)
        frag.trgls = np.copy(self.trgls)
        frag.neighbors = np.copy(self.neighbors)
        frag.valid = True
        return frag

    # class function
    def saveList(frags, path, stem):
        # cfixed = self.created.replace(':',"_").replace('.',"p")
        for frag in frags:
            cfixed = Utils.timestampToVc(frag.created)
            if cfixed is None:
                print("Could not convert self.created", self.created, "to vc")
                cfixed = frag.created.replace(':',"_").replace('.',"p")
            print("tsl", frag.name)
            fpath = path / cfixed
            frag.save(fpath)

    # class function
    def saveListAsObjMesh(fvs, path, infill, ppm, class_count):
        print("TF slaom", len(fvs), class_count)
        name = path.name
        stem = path.stem
        for fv in fvs:
            frag = fv.fragment
            if class_count > 1 or len(fvs) > 1:
                newname = stem+"_"+frag.name
                opath = path.with_name(newname)
            else:
                opath = path
            print("TF slaom", opath)
            frag.save(opath, ppm, fv)

        return ""

    def save(self, fpath, ppm=None, fv=None):
        obj_path = fpath.with_suffix(".obj")
        name = fpath.name
        print("TF save", obj_path)
        of = obj_path.open("w")
        # print("hello", file=of)
        print("# Khartes OBJ File", file=of)
        print("# Created: %s"%self.created, file=of)
        print("# Name: %s"%self.name, file=of)
        print("# Vertices: %d"%len(self.gpoints), file=of)
        ns = BaseFragment.pointNormals(self.gpoints, self.trgls)
        vrts = self.gpoints
        if ppm is not None:
            vrts = ppm.layerIjksToScrollIjks(vrts)
        for i, pt in enumerate(vrts):
            print("v %f %f %f"%(pt[0], pt[1], pt[2]), file=of)
            if ns is not None:
                n = ns[i]
                print("vn %f %f %f"%(n[0], n[1], n[2]), file=of)
        print("# Color and texture information", file=of)
        # print("mtllib %s.mtl"%self.name, file=of)
        print("mtllib %s.mtl"%name, file=of)
        print("usemtl default", file=of)
        has_texture = (len(self.gtpoints) == len(self.gpoints))
        if has_texture:
            for i, pt in enumerate(self.gtpoints):
                print("vt %f %f"%(pt[0], pt[1]), file=of)
        print("# Faces: %d"%len(self.trgls), file=of)
        for trgl in self.trgls:
            ostr = "f"
            for i in range(3):
                v = trgl[i]+1
                if has_texture:
                    ostr += " %d/%d/%d"%(v,v,v)
                else:
                    ostr += " %d/%d"%(v,v)
            print(ostr, file=of)
        mtl_path = fpath.with_suffix(".mtl")
        try:
            of = mtl_path.open("w")
        except Exception as e:
            print("Could not open %s: %s"%(str(mtl_path), e))
            return
            
        print("newmtl default", file=of)
        rgb = self.color.getRgbF()
        print("Ka %f %f %f"%(rgb[0],rgb[1],rgb[2]), file=of)
        print("Kd %f %f %f"%(rgb[0],rgb[1],rgb[2]), file=of)
        print("Ks 0.0 0.0 0.0", file=of)
        print("illum 2", file=of)
        print("d 1.0", file=of)
        # TODO: print this only if TIFF file exists
        # if has_texture:
        #     print("map_Kd %s.tif"%name, file=of)

        if fv is not None:
            jfilename = fpath.with_suffix(".json")
            jdict = {}
            fdict = {}
            fdict["name"] = self.name
            area = fv.sqcm
            fdict["area_sq_cm"] = area
            fdict["n_vrts"] = len(self.gpoints)
            fdict["n_trgls"] = len(self.trgls)
            jdict[self.name] = fdict
            info_txt = json.dumps(jdict, indent=4)
            try:
                ofj = jfilename.open("w")
                print(info_txt, file=ofj)
            except Excepton as e:
                print("Could not open %s: %s"%(str(jfilename), e))
                return

    # find the intersections between a plane defined by axis and position,
    # and the triangles.  
    # The first return value is an array
    # with 6 columns (the x,y,z location of each of the two
    # intersections with a given trgl), and as many rows as
    # there are intersected triangles.
    # The second return value is a vector, as long as the first
    # array, with the trgl index of each intersected triangle.
    def findIntersections(pts, trgls, axis, position):
        gpts = pts
        # print("min", np.min(gpts, axis=0))
        # print("max", np.max(gpts, axis=0))
        # trgls = self.trgls
        # print("trgls", trgls.shape)
        # print(axis, position)

        # shift intersection plane slightly so that
        # no vertices lie on the plane
        while len(gpts[gpts[:,axis]==position]) > 0:
            position += .01
        
        # print(axis, position)
        
        # -1 or 1 depending on which side gpt is in relation
        # to the plane defined by axis and position
        gsgns = np.sign(gpts[:,axis] - position)
        # print("gsgns", gsgns.shape)
        # print(gsgns)
        # -1 or 1 for each vertex of each triangle
        trglsgns = gsgns[trgls]
        # sum of the signs for each trgl
        tssum = trglsgns.sum(axis=1)
        # if sum is -3 or 3, the trgl is entirely on one
        # side of the plane, and can be ignored from now on
        esor = (tssum != -3) & (tssum != 3)
        trglsgns = trglsgns[esor]
        trglvs = trgls[esor]
        # print("trglsgns", trglsgns.shape)
        # print(trglsgns)
        # print(trglvs)
        
        # shift the trglsgns by one, to compare each vertex to 
        # its adjacent vertex around the trgl
        trglroll = np.roll(trglsgns, 1, axis=1)
        # assuming vertices of each trgl are labeled 0,1,2,
        # for each trgl set a boolean showing whether each edge
        # of the trgl crosses the plane, in order:
        # 1 to 2, 2 to 0, 0 to 1
        es = np.roll((trglsgns != trglroll), 1, axis=1)
        
        # print(es)
        
        # repeat each column of es
        es2 = np.repeat(es, 2, axis=1)
        
        # for each trgl, assuming its vertices are numbered 0,1,2,
        # create a row of six vertices, corresponding to the
        # edge ordering in es, namely: 1,2,2,0,0,1
        vs0 = np.roll(np.repeat(trglvs, 2, axis=1), 3, axis=1)
        
        # if a given edge does NOT cross the plane, replace its
        # vertex numbers by -1
        vs0[~es2] = -1
        # print(vs0)
        
        # find all "-1" vertices in columns 0 and 1 and
        # roll them to columns 4 and 5
        m = vs0[:,0] == -1
        vs0[m] = np.roll(vs0[m], -2, axis=1)
        
        # find all "-1" vertices in columns 2 and 3 and
        # roll them to columns 4 and 5
        m = vs0[:,2] == -1
        vs0[m] = np.roll(vs0[m], 2, axis=1)
        
        # each row of vs contains two pairs of vertices specifying
        # the two edges of the triangle that cross the plane
        vs = vs0[:,:4]
        # print(vs)
        
        # There shouldn't be any "-1" values in vs at this point,
        # but if there are, filter them out
        vs = vs[vs[:,0] != -1]
        vs = vs[vs[:,2] != -1]
        
        # gpts projected on the axis
        gax = gpts[:,axis]
        
        # vsh has only one edge (vertex pair) per row
        vsh = vs.reshape(-1,2)
        # extract the two vertices of each edge pair
        v0 = vsh[:,0]
        v1 = vsh[:,1]
        
        # calculate the point where each edge intersects the plane.
        # d is always non-zero because v0 and v1 lie on opposite
        # sides of the plane
        d = gax[v1] - gax[v0]
        a = (position - gax[v0])/d
        a = a.reshape(-1,1)
        i = (1-a)*gpts[v0] + a*gpts[v1]
        
        # put the two intersection points, for the two edges of the single
        # triangle, back into a single row
        i01 = i.reshape(-1,6)
        # print(i01)
        # print("i01", i01.shape)

        trglist = np.indices((len(trgls),))[0]
        # print(trgls.shape, trglist.shape, esor.shape)
        trglist = trglist[esor]

        return i01, trglist


class TrglFragmentView(BaseFragmentView):
    def __init__(self, project_view, trgl_fragment):
        super(TrglFragmentView, self).__init__(project_view, trgl_fragment)
        # self.project_view = project_view
        # self.fragment = trgl_fragment
        # TODO fix:
        self.line = None
        self.setWorkingRegion(-1, 0.)
        self.has_working_non_working = (False, False)
        self.prev_pt_count = 0
        self.stpoints = None
        self.all_stpoints = None
        if len(trgl_fragment.trgls) == 0:
            self.mesh_visible = False

    # TODO: if cur_volume_view changed, unset working region
    # NOTE that Fragment.setLocalPoints sets stpoints,
    # but TrglFragment.setLocalPoints does not.
    def setLocalPoints(self, recursion_ok=True, always_update_zsurfs=True):
        self.local_points_modified = Utils.timestamp()
        if self.cur_volume_view is None:
            self.vpoints = np.zeros((0,4), dtype=np.float32)
            self.fpoints = self.vpoints
            # self.working_vpoints = np.zeros((0,4), dtype=np.float32)
            self.setWorkingRegion(-1, 0.)
            return
        self.vpoints = self.cur_volume_view.globalPositionsToTransposedIjks(self.fragment.gpoints)
        self.fpoints = self.vpoints
        # self.tpoints = self.fragment.gtpoints.copy()
        self.fragment.direction = self.cur_volume_view.direction
        npts = self.vpoints.shape[0]
        if npts > 0:
            indices = np.reshape(np.arange(npts), (npts,1))
            # print(self.vpoints.shape, indices.shape)
            self.vpoints = np.concatenate((self.vpoints, indices), axis=1)
        self.calculateSqCm()
        self.setScaledTexturePoints()
        # self.createTetras()
        # self.setWorkingRegion(35555, 60.)
        # TODO:
        # update positions of working points
        if self.working_fv is not None:
            self.working_fragment.gpoints = self.fragment.gpoints[self.working_vpoints]
            # recursion_ok=True causes crash due to FragmentView.setLocalPoints
            # looping over project_view.fragments
            # print("before wfv slp")
            self.working_fv.setLocalPoints(False)
            # print("after wfv slp")

    '''
    def createTetras(self):
        xyzs = self.vpoints[:,0:3]
        timer = Utils.Timer()
        self.tetras = Delaunay(xyzs)
        timer.time("delaunay")
        print(len(xyzs), len(self.tetras.simplices))
    '''

    '''
    
    Each vertex in a .obj file that is created by vc carries two
    pieces of information: the xyz location, and the texture (uv)
    coordinate.
    The xyz location is in scroll coordinates, but the uv coordinates
    are stretched or compressed so they both extend over the entire
    range 0.0 to 1.0.  The uv coordinates may also be rotated relative
    to the viewing angle we may prefer.
    The setScaledTexturePoints routine attepts to find a transformation 
    (scale, rotate, shift, aka an affine transformation) 
    that will convert uv coordinates into what I called "st" 
    (scaled texture) coordinates.  I denote the two resulting coordinates 
    as stx and sty.  
    Given the fundamental constraints (all stx and sty values are
    generated from u and v using the same affine transformation),
    the goals are:
        1) sty is, as much as possible, parallel to the original z axis;
        2) each triangle in st coordinates preserves, as much as possible,
           the area and angles of the original triangle in xyz space.
    
    The steps are:
        1) transform each triangle into a flattened xy space, exactly
        preserving areas and angles, and where the original alignment with
        the z axis is preserved;
        2) shift the triangle so that its center is at the origin of the
        flattened xy space;
        3) for each triangle, shift the uv coordinates of its vertices
        so that the center of the triangle in uv space is at the origin
        of uv space;
        4) solve a least-squares equation to find the coefficients
        (I call them a,b,c,d) of a rotation+scale transform (no
        shift), with the objective function being: the sum of the distances
        between the transformed uv points (transformed by the a,b,c,d
        matrix) and the recentered xy points be as small as possible.
    
    '''

    def setScaledTexturePoints(self):
        f = self.fragment
        # print("sstp")
        if self.stpoints is not None and len(f.gpoints) == self.prev_pt_count:
            # print("sstp returning")
            return
        self.stpoints = None
        self.all_stpoints = None
        # print("sstp set stpoints to None")
        self.prev_pt_count = len(f.gpoints)
        if len(f.gtpoints) != len(f.gpoints):
            return


        timer = Utils.Timer()
        # print("t, v",self.trgls().shape, self.vpoints.shape)

        # original xyzs
        oxyzs = self.vpoints[:,0:3]
        # oxyzs = f.gpoints[:]
        # txyzs is array[trgl #][trgl pt (0, 1, or 2)][pt xyz]
        txyzs = oxyzs[self.trgls()]

        # print(txyzs[0], txyzs[-1])

        # centers of triangles
        cxyzs = txyzs.sum(axis=1)/3
        # print(cxyzs.shape)
        cxyzs = cxyzs[:,np.newaxis,:]

        # print(cxyzs[0], cxyzs[-1])

        # Shift each trgl in txyzs so that the center of each
        # triangle is at the origin of the xyz coordinate system
        txyzs -= cxyzs

        # print(txyzs[0], txyzs[-1])
        # print("txyzs", txyzs.shape)
        t01 = txyzs[:,1]-txyzs[:,0]
        t02 = txyzs[:,2]-txyzs[:,0]
        # print("t10", t10.shape)
        # print(txyzs[0])
        # print(t10[0])

        # an array with the normal of each triangle
        # (not yet normalized)
        tnorm = np.cross(t01, t02)

        # print(tnorm.shape, weights.shape)

        # For each triangle, fxyaxis lies in the plane of the triangle,
        # and is perpendicular to the z axis
        # NOTE that in the local transposed coordinate
        # system, the global z axis is in the local y direction.
        fxyaxis = np.cross(tnorm, (0.,1.,0))
        # fxyaxis = np.cross(tnorm, (0.,0.,1))
        # print("ijk axis", self.iIndex, self.jIndex, self.kIndex)

        # For each triangle, calculate a weight based on the
        # cross product of the unnormalized triangle normal and
        # the z axis.
        # This weight will be used later in the least-squares process.
        # The idea is: triangles whose normal points in the z-axis
        # direction are not going to provide reliable information on
        # orientation.
        weights = np.sqrt((fxyaxis*fxyaxis).sum(axis=1))
        weights = weights.reshape(-1,1,1)

        # For each triangle, fzaxis lies in the plane of the triangle,
        # and is perpendicular to the triangle's normal and to
        # fxyaxis.  Note also that fzaxis lies in the plane formed
        # by the z axis, and the triangle's normal.
        fzaxis = np.cross(tnorm, fxyaxis)

        # normalize fzaxis
        norm = np.linalg.norm(fzaxis, axis=1, keepdims=True)
        norm[norm==0] = 1.
        fzaxis = fzaxis/norm

        # normalize fxyaxis
        norm = np.linalg.norm(fxyaxis, axis=1, keepdims=True)
        norm[norm==0] = 1.
        fxyaxis = fxyaxis/norm

        # print(fzaxis[0], fzaxis[-1])
        # re-center xyz to center of triangle, apply axes to get fxy, fz
        # compare this to re-centered u,v
        # print(fxyaxis.shape)
        fxyaxis = fxyaxis[:,np.newaxis,:]
        # print(fxyaxis.shape)

        # For each vertex of each triangle, calculate the vertex's
        # position in the new coordinate system formed by the
        # orthognal normalized axes fxyaxis and fzaxis.
        # Because these axes lie in the plane of the triangle,
        # each triangle in the new coordinate system will have the
        # same area and vertex angles as it did in the original xyz
        # coordinate system
        tfxy = (txyzs*fxyaxis).sum(axis=2)
        fzaxis = fzaxis[:,np.newaxis,:]
        # print(fxyaxis.shape)
        tfz = (txyzs*fzaxis).sum(axis=2)
        tfxy = np.stack((tfxy, tfz), axis=2)

        # print(tfxy.shape, tfz.shape, tfxyz.shape)
        # print(tfxy[0])
        # print(tfz[0])
        # print(tfxyz[0])
        # print(fxyaxis[0])
        # print(txyzs[0])
        # print((txyzs*fxyaxis)[0])
        # print(tfxy[0])
        # TODO: now do the same with fzaxis
        # then combine to form tfxys
        # mxyz = np.stack((fxyaxis,fzaxis), axis=0)
        # print(fxyaxis[0])
        # print(fzaxis[0])
        # print(mxyz[0])

        # tuvs contains the uv coordinates of each
        # vertex of each triangle.  Each row in tuvs
        # represents a single triangle, analogous to
        # txyzs at the top of this function
        tuvs = self.fragment.gtpoints[self.trgls()]
        # print("tuvs", tuvs.shape)
        # TODO: testing!
        # tuvs = tuvs[:,:,(1,0)]
        # print("tuvs", tuvs.shape)

        # Calculate the center of each triangle in uv coordinates.
        cuvs = tuvs.sum(axis=1)/3
        cuvs = cuvs[:,np.newaxis,:]
        # Shift each trgl in tuvs so that the center of the triangle
        # in uv space lies at the origin of the uv coordinate system.
        tfuv = tuvs-cuvs
        # print(tfxyz.shape, tfuv.shape)

        # apply the weights to the xy and uv coordinates
        tfxy *= weights
        tfxy = tfxy.reshape(-1,2)
        tfuv *= weights
        tfuv = tfuv.reshape(-1,2)
        # print(tfxy.shape, tfuv.shape)

        # extract the re-centered u and v for each vertex of each triangle
        u = tfuv[:,0]
        v = tfuv[:,1]
        # extract the re-centered, flattened x and y for each
        # vertex of each triangle
        x = tfxy[:,0]
        y = tfxy[:,1]

        # u, v, x, and y are each one-dimensional arrays whose
        # length is 3 times the number of triangles

        # n = len(u)

        '''
        # slc = slice(500000,600000)
        # slc = slice(300000,n)
        # slc = slice(n)
        # sn = len(range(*slc.indices(n)))
        print("sn", sn)
        uu = (u*u)[slc].sum()
        uv = (u*v)[slc].sum()
        vv = (v*v)[slc].sum()
        ux = (u*x)[slc].sum()
        uy = (u*y)[slc].sum()
        vx = (v*x)[slc].sum()
        vy = (v*y)[slc].sum()
        '''

        # Solve a least-squares problem.  The idea is to
        # find 4 numbers, (a,b,c,d), that minimize the error
        # in these two equations:
        # a*u + b*v = x
        # c*u + d*v = y
        # where u, v, x, y are the arrays computed above.

        # The math is not derived here.

        uu = (u*u).sum()
        uv = (u*v).sum()
        vv = (v*v).sum()
        ux = (u*x).sum()
        uy = (u*y).sum()
        vx = (v*x).sum()
        vy = (v*y).sum()

        # mden = uu+vv-2*uv

        # denominator of the inverse of A.t()@A
        mden = uu*vv - uv*uv
        # sn = n
        # print("mden", mden/sn)
        # print("uu vv uv", uu/sn, uv/sn, vv/sn)
        # print("uxy vxy", ux/sn, uy/sn, vx/sn, vy/sn)
        if mden == 0:
            return

        a = ( vv*ux - uv*vx)/mden
        b = (-uv*ux + uu*vx)/mden
        c = ( vv*uy - uv*vy)/mden
        d = (-uv*uy + uu*vy)/mden
        # print("abcd", a,b,c,d)

        # Now apply the coordinate transform defined
        # by a,b,c,d to the original uv points, to get
        # scaled transformed points
        gtp = self.fragment.gtpoints
        gu = gtp[:,0]
        gv = gtp[:,1]
        # TODO: testing!
        # gu = gtp[:,1]
        # gv = gtp[:,0]
        stp = np.zeros_like(gtp)
        stp[:,0] = a*gu + b*gv
        stp[:,1] = c*gu + d*gv
        # stp[:,0] -= stp[:,0].min()
        # stp[:,1] -= stp[:,1].min()
        self.st_abcd = (a,b,c,d)

        stmin = stp.min(axis=0)
        stmax = stp.max(axis=0)
        styc = .5*(stmin[1]+stmax[1])
        xyzmin = oxyzs.min(axis=0)
        xyzmax = oxyzs.max(axis=0)
        print("xyz min max", xyzmin, xyzmax)
        # zc = .5*(xyzmin[2]+xyzmax[2])
        # See note above; global z axis is in local y-axis direction
        zc = .5*(xyzmin[1]+xyzmax[1])
        print("styc zc", styc, zc)

        # shift the points so that the minimums are at the origin
        # self.st_shift = -stp.min(axis=0)
        self.st_shift = -stmin
        # but then center z
        self.st_shift[1] = zc-styc
        stp += self.st_shift
        timer.time("scale uv time")
        print("stx range", stp[:,0].min(), stp[:,0].max())
        print("sty range", stp[:,1].min(), stp[:,1].max())

        # at last, set the st ("scaled texture") points
        self.stpoints = stp
        # self.stmin = stmin
        # self.stmax = stmax
        self.stmin = stp.min(axis=0)
        self.stmax = stp.max(axis=0)
        self.xyzmin = xyzmin
        self.xyzmax = xyzmax

        stsize = self.stmax-self.stmin
        starea = (stsize*stsize).sum()
        ptarea = starea / len(self.stpoints)
        self.avg_st_len = math.sqrt(ptarea)
        # print(self.stmin, self.stmax, starea, ptarea, self.avg_len)
        self.outside_stpoints = self.outsidePoints(self.avg_st_len)
        print("stp", stp.shape, "outside", self.outside_stpoints.shape)
        self.all_stpoints = np.concatenate((stp, self.outside_stpoints), axis=0)
        self.retriangulateAll()
        timer.time("retriangulate time")
        
    def setVolumeViewDirection(self, direction):
        self.setWorkingRegion(-1, 0.)
        self.prev_pt_count = 0
        super(TrglFragmentView, self).setVolumeViewDirection(direction)
        
    def setVolumeView(self, vol_view):
        self.setWorkingRegion(-1, 0.)
        self.prev_pt_count = 0
        super(TrglFragmentView, self).setVolumeView(vol_view)

    def setWorkingRegion(self, index, max_angle):
        if index < 0:
            # self.working_trgls = np.zeros((0, 3), dtype=np.int32)
            # self.working_vpoints = np.zeros((0,4), dtype=np.float32)
            self.working_trgls = np.full((len(self.trgls()),), False)
            self.working_vpoints = np.full((len(self.fragment.gpoints),), False)
            self.working_fragment = None
            self.working_fv = None
            self.has_working_non_working = (False, True)
            self.working_fragment = None
            self.working_fv = None
            # print("swr cleared all")
            return
        tbn = self.regionByNormals(index, max_angle)
        # print("tbn", len(tbn))
        wt = np.full((len(self.trgls()),), False)
        wt[tbn] = True
        # self.working_trgls = self.trgls()[tbn]
        self.working_trgls = wt
        vs = np.unique(self.trgls()[tbn].flatten())
        wv = np.full((len(self.vpoints),), False)
        # print(vs.shape, wv.shape)
        # print(vs)
        wv[(vs)] = True
        self.working_vpoints = wv

        lworking = len(vs)
        lnonworking = len(self.vpoints)-lworking
        self.has_working_non_working = (lworking>0, lnonworking>0)
        # print("lwlnw wnw", lworking, lnonworking, self.has_working_non_working)
        # self.working_vpoints = self.vpoints[vs]
        # print("wvp", len(self.working_vpoints))
        # print("trgl_fragment set local points")

        self.working_fragment = Fragment("working", self.fragment.direction)
        self.working_fragment.setColor(self.fragment.color)
        # self.working_fragment.gpoints = np.copy(self.fragment.gpoints)
        self.working_fragment.gpoints = self.fragment.gpoints[self.working_vpoints]
        self.working_fv = FragmentView(None, self.working_fragment)
        self.working_fv.setVolumeView(self.cur_volume_view)
        # print("swr set all")

    def workingZsurf(self):
        if self.working_fv is not None:
            return self.working_fv.workingZsurf()

    def workingSsurf(self):
        if self.working_fv is not None:
            return self.working_fv.workingSsurf()

    def moveAlongNormalsSign(self):
        return -1.

    def workingVpoints(self):
        return self.working_vpoints

    def hasWorkingNonWorking(self):
        return self.has_working_non_working

    def workingTrgls(self):
        return self.working_trgls

    def calculateSqCm(self):
        pts = self.fragment.gpoints
        simps = self.fragment.trgls
        voxel_size_um = self.project_view.project.voxel_size_um
        sqcm = BaseFragment.calculateSqCm(pts, simps, voxel_size_um)
        self.sqcm = sqcm

    def calculateStArea(self):
        pts = self.stpoints
        simps = self.fragment.trgls
        voxel_size_um = 1000000/100 # 1 cm in um
        sqcm = BaseFragment.calculateSqCm(pts, simps, voxel_size_um)

    def getPointsOnSlice(self, axis, i):
        # matches = self.vpoints[(self.vpoints[:, axis] == i)]
        matches = self.vpoints[(self.vpoints[:, axis] >= i-.5001) & (self.vpoints[:, axis] < i+.5001)]
        return matches

    # outputs a list of lines; each line has two vertices
    # NOTE that input axis and position are in local tijk coordinates,
    # and that output vertices are in tijk coordinates
    def getLinesOnSlice(self, axis, axis_pos):
        '''
        tijk = [0,0,0]
        tijk[axis] = axis_pos
        gijk = self.cur_volume_view.transposedIjkToGlobalPosition(tijk)
        gaxis = self.cur_volume_view.globalAxisFromTransposedAxis(axis)
        gpos = gijk[gaxis]
        ints = self.fragment.findIntersections(gaxis, gpos)
        gpts = ints.reshape(-1, 3)
        vpts = self.cur_volume_view.globalPositionsToTransposedIjks(gpts)
        plines = vpts.reshape(-1,2,3)
        return plines
        '''
        ints, trglist = TrglFragment.findIntersections(self.fpoints, self.trgls(), axis, axis_pos)
        plines = ints.reshape(-1,2,3)
        return plines, trglist

    def aligned(self):
        return True

    def trgls(self):
        return self.fragment.trgls

    '''
    def localStAxesOld(self, index):
        axes = BaseFragment.pointThreeAxes(index, self.fragment.gpoints, self.stpoints, self.trgls())
        return axes

    # returns 3 axes: axis along increasing stx, axis along increasing sty,
    # normal.  The 3 axes are orthonormal.
    # TODO: the calculation of stxaxis and styaxis should take into
    # account the local stxy values (uvpts), intead of looking
    # at the z axis as a proxy
    def localStAxesHide(self, pt_index):
        xyzpts = self.fragment.gpoints
        uvpts = self.stpoints
        trgls = self.trgls()
        if uvpts is None or len(xyzpts) != len(uvpts):
            return None
        ltrgl_indexes = BaseFragment.trglsAroundPoint(pt_index, trgls)
        ltrgls = trgls[ltrgl_indexes]

        v0 = ltrgls[:,0]
        v1 = ltrgls[:,1]
        v2 = ltrgls[:,2]
        d01 = (xyzpts[v1] - xyzpts[v0]).astype(np.float64)
        d02 = (xyzpts[v2] - xyzpts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        npt = n3d.sum(axis=0)
        # print("npt", npt)
        l2 = np.sqrt(np.sum(npt*npt))
        if l2 == 0:
            return None
        normal = npt/l2
        # print("normal", normal)
        # In the global coordinate system, this represents
        # the axis along the scroll's original z axis.
        # This should be more or less aligned with the sty axis
        zaxis = np.array((0., 0., 1.), dtype=np.float32)
        stxaxis = np.cross(normal, zaxis)
        # stxaxis = np.cross(zaxis, normal)
        # stxaxis *= -1
        # print("stxaxis", stxaxis)
        l2 = np.sqrt(np.sum(stxaxis*stxaxis))
        if l2 == 0:
            return None
        stxaxis /= l2
        styaxis = np.cross(normal, stxaxis)
        # styaxis *= -1
        axes = np.array((stxaxis, styaxis, normal)).T
        # print(normal, axes)
        # return np.array((stxaxis, styaxis, normal)).T
        return axes
    '''

    def outsidePoints(self, ptstep):
        stp = self.stpoints
        stmin = self.stmin
        stmax = self.stmax
        minid = stmin/ptstep - 5
        maxid = stmax/ptstep + 5
        id0 = np.floor(minid).astype(np.int32)
        id1 = np.ceil(maxid).astype(np.int32)
        idn = id1-id0
        # We want to find all the points outside of the obj surface.
        # To do this, create an array, and set all the cells of
        # the array to 255.  Then set all cells that contain a
        # trgl vertex to 0.
        # Look for connected components, and take the connected
        # component that extends to 0,0; that is the outer region.
        # Note that connected components are created from non-zero
        # components, so need to make sure the points in the area 
        # we are interested in is non-zero.
        arr = np.full(idn, 255, dtype=np.uint8)
        istps = np.floor(stp/ptstep - id0).astype(np.int32)
        print("arr", id0, idn, arr.shape)
        # arr[istps[:,0], istps[:,1]] = 0
        arr[istps[:,0], istps[:,1]] = 0
        # cv2.imwrite("test.png", arr)
        ccoutput = cv2.connectedComponentsWithStats(arr, 4, cv2.CV_32S)
        (nlabels, labels, stats, centroids) = ccoutput
        label0 = labels[0,0]
        # print("nlabels", nlabels, "label0", label0)
        # print("stats", stats[label0])
        arr2 = np.full(idn, 255, dtype=np.uint8)
        # points that are outside are 0, points not in the outside are 255
        arr2[labels == label0] = 0
        # cv2.imwrite("test.png", arr2)
        kernel = np.ones((3,3), np.uint8)
        dilo = cv2.dilate(arr2, kernel, iterations=2)
        dili = cv2.dilate(arr2, kernel, iterations=1)
        diff = dilo-dili
        # cv2.imwrite("test.png", diff)
        pts = np.argwhere(diff)
        # print("pts", pts.shape, pts[0:5])
        pts = (pts+id0)*ptstep
        # print("pts", pts.shape, pts[0:5])
        return pts

    def retriangulateAll(self):
        # st_area = self.calculateStArea()
        # if self.fragment.trgls is None or len(self.fragment.trgls) == 0:
        #     return
        stp = self.stpoints
        if stp is None or len(stp) == 0:
            return
        '''
        stsize = self.stmax-self.stmin
        starea = (stsize*stsize).sum()
        ptarea = starea / len(self.stpoints)
        ptstep = math.sqrt(ptarea)
        print(self.stmin, self.stmax, starea, ptarea, ptstep)
        outside_pts = self.outsidePoints(ptstep)
        print("stp", stp.shape, "outside", outside_pts.shape)
        all_pts = np.concatenate((stp, outside_pts), axis=0)
        '''
        all_pts = self.all_stpoints

        all_trgls = None
        try:
            all_trgls = Delaunay(all_pts).simplices
        except Exception as err:
            err = str(err).splitlines()[0]
            print("movePoints triangulation error: %s"%err)
        if all_trgls is not None:
            new_trgls = all_trgls[(all_trgls < len(stp)).all(axis=1), :]
            # new_trgls = all_trgls
            self.fragment.trgls = TrglPointSet.rotateToMin(new_trgls)
            # self.fragment.trgls = new_trgls

    def retriangulateAndMoveOld(self, index, newst):
        oldst = self.stpoints[index]
        pass

    '''
    # Input: a trgls array (3 columns, n rows)
    # Output: the same array, but each row has been
    # rotated so that the smallest index of that row is
    # moved to the first column
    @staticmethod
    def rotateToMin(trgls):
        # print("rtm")
        mins = np.argmin(trgls, axis=1)
        otrgls = trgls.copy()
        otrgls[mins==1] = np.roll(trgls[mins==1], 2, axis=1)
        otrgls[mins==2] = np.roll(trgls[mins==2], 1, axis=1)
        return otrgls
    '''

    def pointIndexesInWindowOld(self, pt, half_width, include_outside_points):
        if include_outside_points:
            stp = self.all_stpoints
        else:
            stp = self.stpoints
        if stp is None or len(stp) == 0:
            return None
        ptsb = ((pt-half_width <= stp) & 
                (stp <= pt+half_width)).all(axis=1)
        # print("ptsb", ptsb.shape, ptsb.dtype, ptsb.sum())
        # return np.argwhere(ptsb)
        return np.nonzero(ptsb)[0]

    def trglDiffOld(self, old_pts, new_indexes):
        # old_pts_t = [tuple(pt.tolist()[0]) for pt in old_pts]
        old_pts_t = [tuple(pt.tolist()) for pt in old_pts]
        print("o_p_t", len(old_pts), len(new_indexes), old_pts_t[0])
        new_indexes = new_indexes.tolist()
        stp = self.all_stpoints
        if stp is None or len(stp) == 0:
            return None
        new_pts = stp[new_indexes]
        new_pts_t = [tuple(pt.tolist()) for pt in new_pts]
        # dictionary of stxy to new index
        print("stp[ind]", stp[new_indexes[0]], stp[new_indexes[0]].tolist())
        ptdict = {tuple(stp[ind].tolist()):ind for ind in new_indexes}
        # if an stxy location is only in old_pts, 
        # add it with an index of -1
        for opt in old_pts_t:
            # print("opt", opt)
            if opt not in ptdict:
                # print("not in dict", opt)
                # ptdict[opt] = -1
                if len(old_pts) == len(new_indexes): # point moved
                    # nindset = set(new_indexes)
                    # oindset = set(ptdict.values())
                    # missing = nindset-oindset
                    nstset = set(new_pts_t)
                    ostset = set(old_pts_t)
                    # print("missing", missing)
                    # print("diffs", nstset-ostset, ostset-nstset)
                    # print(nindset, oindset)
                    opt = (ostset-nstset).pop()
                    npt = (nstset-ostset).pop()
                    ptdict[opt] = ptdict[npt]

                    pass
                elif len(old_pts)-1 == len(new_indexes): # point deleted
                    ptdict[opt] = -1
                    pass
                else: # not sure how we got here
                    print("confused")
                    ptdict[opt] = -1
                    pass
        # new indexes of old points
        old_indexes = [ptdict[loc] for loc in old_pts_t]
        # print("a")

        old_trgls = None
        try:
            # print("a1")
            # print(old_pts)
            old_trgls = Delaunay(old_pts).simplices
            # print("a2")
        except Exception as err:
            err = str(err).splitlines()[0]
            print("trglDiff triangulation error with old pts: %s"%err)
        if old_trgls is None:
            return None
        # print("b")

        new_trgls = None
        try:
            new_trgls = Delaunay(new_pts).simplices
        except Exception as err:
            err = str(err).splitlines()[0]
            print("trglDiff triangulation error with new pts: %s"%err)
        if new_trgls is None:
            return None
        # print("c")

        # print("old", old_trgls.dtype)
        # print(old_trgls)
        # print(old_indexes)
        nst = len(self.stpoints)
        r_old_trgls = self.rotateToMin(np.array(old_indexes)[old_trgls])
        r_old_trgls = r_old_trgls[r_old_trgls[:,0] >= 0]
        r_old_trgls = r_old_trgls[(r_old_trgls < nst).all(axis=1)]
        r_new_trgls = self.rotateToMin(np.array(new_indexes)[new_trgls])
        r_new_trgls = r_new_trgls[r_new_trgls[:,0] >= 0]
        r_new_trgls = r_new_trgls[(r_new_trgls < nst).all(axis=1)]
        # print("old")
        # print(r_old_trgls)
        # print("new")
        # print(r_new_trgls)

        unique_old_trgls = Utils.setDiff2DIndex(r_old_trgls, r_new_trgls)
        unique_new_trgls = Utils.setDiff2DIndex(r_new_trgls, r_old_trgls)
        # print("uo", unique_old_trgls)
        # print("un", unique_new_trgls)
        return r_old_trgls[unique_old_trgls], r_new_trgls[unique_new_trgls]

    def replaceTrgls(self, uo, un):
        ltrgls = self.trgls().copy()
        if len(uo) > 0:
            orows = []
            for o in uo:
                row = (ltrgls == o).all(axis=1).nonzero()[0]
                # print("row", row, len(row))
                if len(row) == 0:
                    continue
                orows.append(row[0])
            # print("before", len(ltrgls))
            # print("orows", orows)
            ltrgls = np.delete(ltrgls, orows, 0)
            # print("after", len(ltrgls))
        if len(un) > 0:
            ltrgls = np.concatenate((ltrgls, un), axis=0)
        self.fragment.trgls = ltrgls


    def movePoint(self, index, new_vijk, update_xyz, update_st):
        # print("mp")
        vv = self.cur_volume_view
        new_gijk = vv.transposedIjkToGlobalPosition(new_vijk)
        new_uijk = vv.transposedIjkToIjk(new_vijk)
        old_vijk = self.vpoints[index, :3]
        # old_gijk = vv.transposedIjkToGlobalPosition(old_vijk)
        old_uijk = vv.transposedIjkToIjk(old_vijk)
        # dvijk = new_vijk-old_vijk
        # dgijk = [new_gijk[i]-old_gijk[i] for i in range(3)]
        duijk = [new_uijk[i]-old_uijk[i] for i in range(3)]
        # print("movePoint", dvijk, dgijk, duijk)
        # axes = self.fragment.pointThreeAxes(index, self.vpoints[:,:3], self.stpoints, self.trgls())
        # print("a")
        axes = self.localStAxes(index)
        # print("b")
        if axes is None:
            print("TrglFragmentView.movePoint: could not compute axes")
            return
        # rdvijk = (axes.T)@dvijk
        # rdgijk = (axes.T)@dgijk
        rduijk = (axes.T)@duijk
        # rdgijk = (axes)@dgijk
        old_stxy = self.all_stpoints[index]
        # new_stxy = old_stxy-rdvijk[:2]
        # new_stxy = old_stxy+rdvijk[:2]
        # new_stxy = old_stxy+rdgijk[:2]
        new_stxy = old_stxy+rduijk[:2]
        # new_stxy = old_stxy
        # new_gijk = self.cur_volume_view.transposedIjkToGlobalPosition(new_vijk)
        # print(self.fragment.gpoints)
        # print(match, new_gijk)
        if update_xyz:
            self.fragment.gpoints[index, :] = new_gijk
            # print("c")
            self.setLocalPoints(True, False)
            # print("d")
        if update_st:
            # call instead of setting self.stpoints
            # self.retriangulateAndMove(index, new_stxy)

            # TODO: don't hardwire half_width
            # half_width = 100
            # half_width = 20
            half_width = 3*self.avg_st_len
            # print("hw", half_width, self.avg_st_len)

            ops = TrglPointSet(self.all_stpoints, len(self.stpoints), new_stxy, half_width)
            self.stpoints[index, :] = new_stxy
            self.all_stpoints[index, :] = new_stxy
            nps = TrglPointSet(self.all_stpoints, len(self.stpoints), new_stxy, half_width)
            result = TrglPointSet.trglDiff(ops, nps)
            if result is not None:
                otrgls, ntrgls = result
                if len(otrgls) > 0 or len(ntrgls) > 0:
                    print("uo", otrgls)
                    print("un", ntrgls)
                    # self.replaceTrgls(otrgls, ntrgls)
                    self.fragment.trgls = TrglPointSet.replaceTrgls(self.fragment.trgls, otrgls, ntrgls)
            """
            old_indexes = self.pointIndexesInWindow(new_stxy, half_width, True)
            # print("oi", old_indexes.shape, old_indexes)
            '''
            print("ind", index)
            if self.stpoints is not None:
                print("lstp", len(self.stpoints))
            else:
                print("self.stpoints is None!")
                return
            '''

            old_pts = self.all_stpoints[old_indexes]
            self.stpoints[index, :] = new_stxy
            self.all_stpoints[index, :] = new_stxy
            new_indexes = self.pointIndexesInWindow(new_stxy, half_width, True)
            result = self.trglDiff(old_pts, new_indexes)
            if result is not None:
                otrgls, ntrgls = result
                if len(otrgls) > 0 or len(ntrgls) > 0:
                    print("uo", otrgls)
                    print("un", ntrgls)
                    self.replaceTrgls(otrgls, ntrgls)
            # self.retriangulateAll()
            """
        # print(self.fragment.gpoints)
        # modifyZsurf = False
        # if len(self.workingVpoints()) > 0 and self.workingVpoints()[index]:
        #     modifyZsurf = True

        self.fragment.notifyModified()
        return True

    def deletePointByIndex(self, index):
        if index < 0:
            return
        if index >= len(self.fragment.gpoints):
            return
        if self.stpoints is None or index >= len(self.stpoints):
            return
        half_width = 3*self.avg_st_len

        old_stxy = self.all_stpoints[index]
        ops = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        nps = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        nps.deletePoint(index)
        # Retriangulate before deleting point from self.stpoints etc
        result = TrglPointSet.trglDiff(ops, nps)
        if result is not None:
            otrgls, ntrgls = result
            if len(otrgls) > 0 or len(ntrgls) > 0:
                print("uo", otrgls)
                print("un", ntrgls)
                # self.replaceTrgls(otrgls, ntrgls)
                trgls = TrglPointSet.replaceTrgls(self.fragment.trgls, otrgls, ntrgls)
                trgls[trgls>index] -= 1
                self.fragment.trgls = trgls
        '''
        # print("hw", half_width, self.avg_st_len)
        old_stxy = self.all_stpoints[index]
        old_indexes = self.pointIndexesInWindow(old_stxy, half_width, True)
        # print("oi", old_indexes.shape, old_indexes)
        old_pts = self.all_stpoints[old_indexes]
        # self.stpoints[index, :] = new_stxy
        # self.all_stpoints[index, :] = new_stxy
        # new_indexes = self.pointIndexesInWindow(old_stxy, half_width, True)
        new_indexes = old_indexes[old_indexes != index]
        # print("o,n", len(old_indexes), len(new_indexes))
        result = self.trglDiff(old_pts, new_indexes)
        print("lt1", len(self.fragment.trgls))
        # deleted_trgls = (self.fragment.trgls == index).any(axis=1).nonzero()[0]
        # print("deleted", deleted_trgls)
        if result is not None:
            otrgls, ntrgls = result
            if len(otrgls) > 0 or len(ntrgls) > 0:
                print("uo", otrgls)
                print("un", ntrgls)
                self.replaceTrgls(otrgls, ntrgls)
        print("lt2", len(self.fragment.trgls))
        trgls = self.fragment.trgls.copy()
        print("lt3", len(trgls))
        deleted_trgls = (trgls == index).any(axis=1).nonzero()[0]
        print("deleted", deleted_trgls)
        trgls = np.delete(trgls, deleted_trgls, 0)
        print("lt4", len(trgls))
              # new_trgls = all_trgls[(all_trgls < len(stp)).all(axis=1), :]
        trgls[trgls>index] -= 1
        print("lt5", len(trgls))
        '''
        # self.fragment.trgls = trgls

        self.fragment.gpoints = np.delete(self.fragment.gpoints, index, 0)
        self.fragment.gtpoints = np.delete(self.fragment.gtpoints, index, 0)
        # self.vpoints = np.delete(self.vpoints, index, 0)
        # self.fpoints = np.delete(self.fpoints, index, 0)
        self.stpoints = np.delete(self.stpoints, index, 0)
        self.all_stpoints = np.delete(self.all_stpoints, index, 0)

        # prevent setScaledTexturePoints from running
        # when setLocalPoints is called
        self.prev_pt_count = len(self.fragment.gpoints)
        
        self.setLocalPoints(True, False)
        self.fragment.notifyModified()

        '''
        if self.stpoints is None:
            print("self.stpoints is None a")
        else:
            print("lstp a", len(self.stpoints))
        '''

    # returns list of trgl indexes
    def regionByNormals(self, ptind, max_angle):
        pts = self.fpoints
        trgls = self.fragment.trgls
        neighbors = self.fragment.neighbors
        minz = math.cos(math.radians(max_angle))
        # print("minz", minz)
        normals = BaseFragment.faceNormals(pts, trgls)
        trgl_stack = deque()
        tap = BaseFragment.trglsAroundPoint(ptind, trgls)
        zsgn = np.sum(normals[tap,2])
        # print("zsgn", zsgn)
        if zsgn < 0:
            zsgn = -1
        else:
            zsgn = 1
        # print("tap", tap)
        trgl_stack.extend(tap)
        done = set()
        out_trgls = []
        while len(trgl_stack) > 0:
            trgl = trgl_stack.pop()
            # print("1", trgl)
            if trgl in done:
                continue
            done.add(trgl)
            # print("2", trgl)
            n = normals[trgl]
            # print("n", n)
            # if abs(n[2]) < minz:
            if zsgn*n[2] < minz:
                continue
            # print("3", trgl)
            out_trgls.append(trgl)
            for neigh in neighbors[trgl]:
                # print("4", neigh)
                if neigh < 0:
                    continue
                # print("5", neigh)
                if neigh in done:
                    continue
                # print("6", neigh)
                trgl_stack.append(neigh)
            # print("ts", len(trgl_stack))
        return out_trgls


class TrglPointSet:

    def __init__(self, all_stpoints, nstpoints, pt, half_width):
        if all_stpoints is None or len(all_stpoints) == 0:
            return None
        ptsb = ((pt-half_width <= all_stpoints) & 
                (all_stpoints <= pt+half_width)).all(axis=1)
        self.indexes = np.nonzero(ptsb)[0]
        self.pts = all_stpoints[self.indexes]
        self.nstpoints = nstpoints
        # pts_t = [tuple(pt.tolist()) for pt in self.pts]
        # self.pt2ind = {
        #         tuple(all_stpoints[ind].tolist()):ind for ind in self.indexes}

    def deletePoint(self, index):
        row = (self.indexes == index).nonzero()[0]
        self.indexes = np.delete(self.indexes, row, 0)
        self.pts = np.delete(self.pts, row, 0)

    def addPoint(self, stxy, index):
        pass

    @staticmethod
    def trglDiff(oldps, newps):
        old_trgls = oldps.triangulate()
        new_trgls = newps.triangulate()

        if old_trgls is None or new_trgls is None:
            return None

        unique_old_trgls = Utils.setDiff2DIndex(old_trgls, new_trgls)
        unique_new_trgls = Utils.setDiff2DIndex(new_trgls, old_trgls)
        # print("uo", unique_old_trgls)
        # print("un", unique_new_trgls)
        return old_trgls[unique_old_trgls], new_trgls[unique_new_trgls]

    # Input: a trgls array (3 columns, n rows)
    # Output: the same array, but each row has been
    # rotated so that the smallest index of that row is
    # moved to the first column
    @staticmethod
    def rotateToMin(trgls):
        # print("rtm")
        mins = np.argmin(trgls, axis=1)
        otrgls = trgls.copy()
        otrgls[mins==1] = np.roll(trgls[mins==1], 2, axis=1)
        otrgls[mins==2] = np.roll(trgls[mins==2], 1, axis=1)
        return otrgls

    @staticmethod
    def replaceTrgls(trgls, uo, un):
        if len(uo) > 0:
            orows = []
            for o in uo:
                row = (trgls == o).all(axis=1).nonzero()[0]
                # print("row", row, len(row))
                if len(row) == 0:
                    continue
                orows.append(row[0])
            # print("before", len(trgls))
            # print("orows", orows)
            trgls = np.delete(trgls, orows, 0)
            # print("after", len(trgls))
        if len(un) > 0:
            trgls = np.concatenate((trgls, un), axis=0)
        return trgls

    def triangulate(self):
        trgls = None
        try:
            trgls = Delaunay(self.pts).simplices
        except Exception as err:
            err = str(err).splitlines()[0]
            print("trglDiff triangulation error: %s"%err)
            return None

        nst = self.nstpoints
        trgls = self.rotateToMin(np.array(self.indexes)[trgls])
        # trgls = trgls[trgls[:,0] >= 0]
        trgls = trgls[(trgls < nst).all(axis=1)]
        return trgls

    def pointIndexesInWindowOld(self, pt, half_width, include_outside_points):
        if include_outside_points:
            stp = self.all_stpoints
        else:
            stp = self.stpoints
        if stp is None or len(stp) == 0:
            return None
        ptsb = ((pt-half_width <= stp) & 
                (stp <= pt+half_width)).all(axis=1)
        # print("ptsb", ptsb.shape, ptsb.dtype, ptsb.sum())
        # return np.argwhere(ptsb)
        return np.nonzero(ptsb)[0]

    def trglDiffOld(self, old_pts, new_indexes):
        # old_pts_t = [tuple(pt.tolist()[0]) for pt in old_pts]
        old_pts_t = [tuple(pt.tolist()) for pt in old_pts]
        print("o_p_t", len(old_pts), len(new_indexes), old_pts_t[0])
        new_indexes = new_indexes.tolist()
        stp = self.all_stpoints
        if stp is None or len(stp) == 0:
            return None
        new_pts = stp[new_indexes]
        new_pts_t = [tuple(pt.tolist()) for pt in new_pts]
        # dictionary of stxy to new index
        print("stp[ind]", stp[new_indexes[0]], stp[new_indexes[0]].tolist())
        ptdict = {tuple(stp[ind].tolist()):ind for ind in new_indexes}
        # if an stxy location is only in old_pts, 
        # add it with an index of -1
        for opt in old_pts_t:
            # print("opt", opt)
            if opt not in ptdict:
                # print("not in dict", opt)
                # ptdict[opt] = -1
                if len(old_pts) == len(new_indexes): # point moved
                    # nindset = set(new_indexes)
                    # oindset = set(ptdict.values())
                    # missing = nindset-oindset
                    nstset = set(new_pts_t)
                    ostset = set(old_pts_t)
                    # print("missing", missing)
                    # print("diffs", nstset-ostset, ostset-nstset)
                    # print(nindset, oindset)
                    opt = (ostset-nstset).pop()
                    npt = (nstset-ostset).pop()
                    ptdict[opt] = ptdict[npt]

                    pass
                elif len(old_pts)-1 == len(new_indexes): # point deleted
                    ptdict[opt] = -1
                    pass
                else: # not sure how we got here
                    print("confused")
                    ptdict[opt] = -1
                    pass
        # new indexes of old points
        old_indexes = [ptdict[loc] for loc in old_pts_t]
        # print("a")

        old_trgls = None
        try:
            # print("a1")
            # print(old_pts)
            old_trgls = Delaunay(old_pts).simplices
            # print("a2")
        except Exception as err:
            err = str(err).splitlines()[0]
            print("trglDiff triangulation error with old pts: %s"%err)
        if old_trgls is None:
            return None
        # print("b")

        new_trgls = None
        try:
            new_trgls = Delaunay(new_pts).simplices
        except Exception as err:
            err = str(err).splitlines()[0]
            print("trglDiff triangulation error with new pts: %s"%err)
        if new_trgls is None:
            return None
        # print("c")

        # print("old", old_trgls.dtype)
        # print(old_trgls)
        # print(old_indexes)
        nst = len(self.stpoints)
        r_old_trgls = self.rotateToMin(np.array(old_indexes)[old_trgls])
        r_old_trgls = r_old_trgls[r_old_trgls[:,0] >= 0]
        r_old_trgls = r_old_trgls[(r_old_trgls < nst).all(axis=1)]
        r_new_trgls = self.rotateToMin(np.array(new_indexes)[new_trgls])
        r_new_trgls = r_new_trgls[r_new_trgls[:,0] >= 0]
        r_new_trgls = r_new_trgls[(r_new_trgls < nst).all(axis=1)]
        # print("old")
        # print(r_old_trgls)
        # print("new")
        # print(r_new_trgls)

        unique_old_trgls = Utils.setDiff2DIndex(r_old_trgls, r_new_trgls)
        unique_new_trgls = Utils.setDiff2DIndex(r_new_trgls, r_old_trgls)
        # print("uo", unique_old_trgls)
        # print("un", unique_new_trgls)
        return r_old_trgls[unique_old_trgls], r_new_trgls[unique_new_trgls]

    def replaceTrglsOld(self, uo, un):
        ltrgls = self.trgls().copy()
        if len(uo) > 0:
            orows = []
            for o in uo:
                row = (ltrgls == o).all(axis=1).nonzero()[0]
                # print("row", row, len(row))
                if len(row) == 0:
                    continue
                orows.append(row[0])
            # print("before", len(ltrgls))
            # print("orows", orows)
            ltrgls = np.delete(ltrgls, orows, 0)
            # print("after", len(ltrgls))
        if len(un) > 0:
            ltrgls = np.concatenate((ltrgls, un), axis=0)
        self.fragment.trgls = ltrgls







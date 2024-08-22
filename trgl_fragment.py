import math
import json
import numpy as np

from pathlib import Path
from collections import deque
import traceback
from scipy.spatial import Delaunay
import scipy
import cv2

from utils import Utils
from base_fragment import BaseFragment, BaseFragmentView
from fragment import Fragment, FragmentView
from uv_mapper import UVMapper

from PyQt5.QtGui import QColor

class TrglFragment(BaseFragment):
    def __init__(self, name):
        super(TrglFragment, self).__init__(name)
        self.gpoints = np.zeros((0,3), dtype=np.float32)
        self.gtpoints = np.zeros((0,2), dtype=np.float32)
        self.trgls = np.zeros((0,3), dtype=np.int32)
        self.direction = 0
        self.params = {}

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
            if words == []: # prevent crash on empty line
                continue
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
        if len(vrtl) > 0:
            trgl_frag.gpoints = np.array(vrtl, dtype=np.float32)
        else:
            trgl_frag.gpoints = np.zeros((0,3), dtype=np.float32)
        if len(tvrtl) > 0:
            trgl_frag.gtpoints = np.array(tvrtl, dtype=np.float32)
        else:
            trgl_frag.gtpoints = np.zeros((0,2), dtype=np.float32)
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
        if len(trgl_frag.gtpoints) > 0:
            tmp_fv = trgl_frag.createView(None)
            tmp_fv.setScaledTexturePoints(similar=False)
            trgl_frag.gtpoints = tmp_fv.stpoints

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
        stem = fpath.stem
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
        print("mtllib %s.mtl"%stem, file=of)
        print("usemtl default", file=of)
        image_file = ""
        has_texture = (len(self.gtpoints) == len(self.gpoints))
        if has_texture:
            tpts = self.gtpoints
            if fv is not None and fv.stpoints is not None and fv.map_corners is not None:
                image_ext = ".png"
                image_file = "%s%s"%(stem, image_ext)
                image_path = fpath.with_suffix(image_ext)
                cv2.imwrite(str(image_path), fv.map_image)
                c = fv.map_corners
                tpts = fv.stpoints.copy()
                st0 = np.array(c[0])
                st1 = np.array(c[1])
                # st0 = st0[::-1]
                dst = st1-st0
                # print(st0, st1, dst)
                if dst[0] != 0. and dst[1] != 0.:
                    # print("converting tpts")
                    tpts = (tpts-st0)/dst
                    tpts[:,1] = 1.-tpts[:,1]
                    tpts[tpts>1.] = -1.e+3
                    tpts[tpts<0.] = -1.e+3
                    tpts[(tpts<0).any(axis=1), :] = -1.e+3

            for i, pt in enumerate(tpts):
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

        # TODO: testing only!
        # print("map_Kd asdf.tif", file=of)
        if image_file != "":
            print("map_Kd", image_file, file=of)

        print("illum 2", file=of)
        print("d 1.0", file=of)
        # TODO: print this only if TIFF file exists
        # if has_texture:
        #     print("map_Kd %s.tif"%stem, file=of)

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
        self.stpoints = np.zeros((0,2))
        self.all_stpoints = np.zeros((0,2))
        self.setStxyDefaults()
        self.normals = None
        self.normal_offset = 0.
        self.half_width_multiplier = 10
        if len(trgl_fragment.trgls) == 0:
            self.mesh_visible = False

    def allowAutoExtrapolation(self):
        return False

    def allowAutoInterpolation(self):
        return False

    # TODO: if cur_volume_view changed, unset working region
    # NOTE that Fragment.setLocalPoints sets stpoints,
    # but TrglFragment.setLocalPoints does not.
    def setLocalPoints(self, recursion_ok=True, always_update_zsurfs=True):
        # print("set local points")
        self.local_points_modified = Utils.timestamp()
        if self.cur_volume_view is None:
            self.vpoints = np.zeros((0,4), dtype=np.float32)
            self.fpoints = self.vpoints
            # self.working_vpoints = np.zeros((0,4), dtype=np.float32)
            self.setWorkingRegion(-1, 0.)
            return
        # print("gpoints", self.fragment.gpoints)
        self.vpoints = self.cur_volume_view.globalPositionsToTransposedIjks(self.fragment.gpoints)
        self.fpoints = self.vpoints
        # self.tpoints = self.fragment.gtpoints.copy()
        self.fragment.direction = self.cur_volume_view.direction
        npts = self.vpoints.shape[0]
        '''
        if npts > 0:
            indices = np.reshape(np.arange(npts), (npts,1))
            # print(self.vpoints.shape, indices.shape)
            self.vpoints = np.concatenate((self.vpoints, indices), axis=1)
        '''
        indices = np.reshape(np.arange(npts), (npts,1))
        # print(self.vpoints.shape, indices.shape)
        self.vpoints = np.concatenate((self.vpoints, indices), axis=1)
        self.calculateSqCm()
        vv = self.cur_volume_view
        if vv.stxytf is not None:
            uvxytf = self.stxyToUv(vv.stxytf)
        self.setScaledTexturePoints()
        if vv.stxytf is not None:
            vv.stxytf = self.uvToStxy(uvxytf)
        self.mesh_visible = (len(self.trgls()) > 0)
        '''
        mapper = UVMapper(self.fragment.gtpoints, self.trgls())
        pt0, pt1 = mapper.getTwoAdjacentBoundaryPoints()
        mapper.constraints = np.array([[pt0, 0., 0.], [pt1, 1., 0.]], dtype=np.float64)
        uvs = mapper.computeUvsFromABF()
        self.fragment.gtpoints = uvs
        self.stpoints = None
        self.setScaledTexturePoints()
        '''
        # timer = Utils.Timer()
        # print("computing normals")
        # TODO: compute only modified normals
        self.normals = BaseFragment.pointNormals(self.vpoints[:,:3], self.trgls())
        # timer.time("normals")
        
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

    def setScaledTexturePoints(self, similar=True):
        # similar=False
        # traceback.print_stack()
        f = self.fragment
        # print("sstp")
        # if self.stpoints is not None and len(self.trgls()) >= 10 and len(f.gpoints) == self.prev_pt_count:
        if self.stpoints is not None and len(f.gpoints) == self.prev_pt_count:
            # print("sstp returning")
            return
        # self.stpoints = None
        # self.all_stpoints = None
        # print("sstp set stpoints to None")
        self.prev_pt_count = len(f.gpoints)
        if len(f.gtpoints) != len(f.gpoints):
            print("length mismatch", len(f.gtpoints), len(f.gpoints))
            return

        self.deleteDisconnectedComponents()
        # self.deleteFreePoints()
        timer = Utils.Timer()
        timer.active = False
        # print("t, v",self.trgls().shape, self.vpoints.shape)

        # original xyzs
        # TODO: should use gpoints instead of vpoints??
        # oxyzs = self.vpoints[:,0:3].astype(np.float64)
        oxyzs = self.fragment.gpoints.astype(np.float64)
        # oxyzs = f.gpoints[:]
        # txyzs is array[trgl #][trgl pt (0, 1, or 2)][pt xyz]
        txyzs = oxyzs[self.trgls()].astype(np.float64)

        gtps = self.fragment.gtpoints.astype(np.float64)
        # print("gtp range", gtps.min(axis=0), gtps.max(axis=0))

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
        # fxyaxis = np.cross(tnorm, (0.,1.,0))
        fxyaxis = np.cross(tnorm, (0.,0.,1))
        # print("ijk axis", self.iIndex, self.jIndex, self.kIndex)

        # For each triangle, calculate a weight based on the
        # cross product of the unnormalized triangle normal and
        # the z axis.
        # This weight will be used later in the least-squares process.
        # The idea is: triangles whose normal points in the z-axis
        # direction are not going to provide reliable information on
        # orientation.
        weights = np.sqrt((fxyaxis*fxyaxis).sum(axis=1))
        nw = len(weights)
        if nw > 10:
            wsort = np.argsort(weights)
            median_weight = weights[wsort[nw//2]]
            min_weight = weights[wsort[0]]
            max_weight = weights[wsort[-1]]
            weight90 = weights[wsort[(9*nw)//10]]
            print("weight range", min_weight, median_weight, weight90, max_weight)
            max_allowed_weight = 4*median_weight
            weights[weights > max_allowed_weight] = max_allowed_weight


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
        tuvs = gtps[self.trgls()].astype(np.float64)
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

        tfxynw = tfxy.copy()
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
        mden = uu*vv - uv*uv
        mden2 = uu+vv

        # sn = n
        # print("mden", mden/len(u))
        # print("uu vv uv", uu/sn, uv/sn, vv/sn)
        # print("uxy vxy", ux/sn, uy/sn, vx/sn, vy/sn)
        if mden == 0 or mden2 == 0:
            print("mden = 0")
            print("oxyzs,txyzs")
            print(oxyzs)
            print(txyzs)
            return

        # print("gt min max", gtp.min(axis=0), gtp.max(axis=0))
        gu = gtps[:,0]
        gv = gtps[:,1]
        stp = np.zeros_like(gtps)

        if similar:
            # print("similar")
            abcds = []

            a = (ux - vy)/mden2
            b = (vx + uy)/mden2
            c = b
            d = -a
            abcds.append((a,b,c,d))

            a = (ux + vy)/mden2
            b = (vx - uy)/mden2
            c = -b
            d = a
            abcds.append((a,b,c,d))

            # TODO: for testing only
            a = ( vv*ux - uv*vx)/mden
            b = (-uv*ux + uu*vx)/mden
            c = ( vv*uy - uv*vy)/mden
            d = (-uv*uy + uu*vy)/mden
            abcds.append((a,b,c,d))

            errors = []
            for a,b,c,d in abcds:
                stp[:,0] = a*gu + b*gv
                stp[:,1] = c*gu + d*gv
                tuvs = stp[self.trgls()].astype(np.float64)
    
                # Calculate the center of each triangle in uv coordinates.
                cuvs = tuvs.sum(axis=1)/3
                cuvs = cuvs[:,np.newaxis,:]
                # Shift each trgl in tuvs so that the center of the triangle
                # in uv space lies at the origin of the uv coordinate system.
                tfuv = tuvs-cuvs

                dd = tfxynw.flatten() - tfuv.flatten()
                error = np.sqrt((dd*dd).sum())/len(dd)
                errors.append(error)
                # print("abcd", a,b,c,d)
                # print("st error", error)
                # print(tfuv[0])
                # print(tfxynw[0:3])

            # print("st errors", errors)
            if errors[0] < errors[1]:
                a,b,c,d = abcds[0]
            else:
                a,b,c,d = abcds[1]

        else:
            print("affine")
            # denominator of the inverse of A.t()@A
            a = ( vv*ux - uv*vx)/mden
            b = (-uv*ux + uu*vx)/mden
            c = ( vv*uy - uv*vy)/mden
            d = (-uv*uy + uu*vy)/mden
            # print("abcd", a,b,c,d)

        # Now apply the coordinate transform defined
        # by a,b,c,d to the original uv points, to get
        # scaled transformed points
        '''
        sgu = np.sort(gu)
        sgv = np.sort(gv)
        ng = len(gtp)
        if ng > 10:
            print("median gtp", sgu[ng//2], sgv[ng//2])
            print("25% gtp", sgu[ng//4], sgv[ng//4])
            print("75% gtp", sgu[(3*ng)//4], sgv[(3*ng)//4])
            print("min gtp", sgu[0], sgv[0])
            print("max gtp", sgu[ng-1], sgv[ng-1])
        '''
        # TODO: testing!
        # gu = gtp[:,1]
        # gv = gtp[:,0]

        stp[:,0] = a*gu + b*gv
        stp[:,1] = c*gu + d*gv
        # stp[:,0] -= stp[:,0].min()
        # stp[:,1] -= stp[:,1].min()
        self.st_abcd = (a,b,c,d)

        stmin = stp.min(axis=0)
        stmax = stp.max(axis=0)
        # print("st min max", stmin, stmax)
        styc = .5*(stmin[1]+stmax[1])
        xyzmin = oxyzs.min(axis=0)
        xyzmax = oxyzs.max(axis=0)
        # print("xyz min max", xyzmin, xyzmax)
        # zc = .5*(xyzmin[2]+xyzmax[2])
        # See note above; global z axis is in local y-axis direction
        zc = .5*(xyzmin[1]+xyzmax[1])
        # print("styc zc", styc, zc)

        # print("stx range", stp[:,0].min(), stp[:,0].max())
        # print("sty range", stp[:,1].min(), stp[:,1].max())
        # shift the points so that the minimums are at the origin
        # self.st_shift = -stp.min(axis=0)
        self.st_shift = -stmin
        # but then center z
        self.st_shift[1] = zc-styc
        stp += self.st_shift
        timer.time("scale uv time")
        # print("stx range", stp[:,0].min(), stp[:,0].max())
        # print("sty range", stp[:,1].min(), stp[:,1].max())

        # at last, set the st ("scaled texture") points
        self.stpoints = stp
        # self.stmin = stmin
        # self.stmax = stmax
        self.stmin = stp.min(axis=0)
        self.stmax = stp.max(axis=0)
        # self.xyzmin = xyzmin
        # self.xyzmax = xyzmax

        # stsize = self.stmax-self.stmin
        # starea = (stsize*stsize).sum()
        # ptarea = starea / len(self.stpoints)
        # self.avg_st_len = math.sqrt(ptarea)
        lens = TrglPointSet.edgeLengths(self.stpoints, self.trgls())
        self.avg_st_len = 0.
        if len(lens) > 0:
            self.avg_st_len = lens.sum()/len(lens)
        # print(self.stmin, self.stmax, starea, ptarea, self.avg_st_len)
        print(self.stmin, self.stmax, self.avg_st_len)
        self.outside_stpoints = self.outsidePoints(self.avg_st_len)
        # print("stp", stp.shape, "outside", self.outside_stpoints.shape)
        self.all_stpoints = np.concatenate((stp, self.outside_stpoints), axis=0)
        self.retriangulateAll()
        timer.time("retriangulate time")

    def setStxyDefaults(self):
        self.st_abcd = (1.,0.,0.,1.)
        self.st_shift = np.zeros(2, dtype=np.float64)
        self.stmin = np.zeros(2, dtype=np.float64)
        self.stmax = np.zeros(2, dtype=np.float64)
        self.avg_st_len = 100

    def uvToStxy(self, uv):
        u, v = uv
        a,b,c,d = self.st_abcd
        sh = self.st_shift
        stxy = (a*u+b*v+sh[0], c*u+d*v+sh[1])
        return stxy

    def uvsToStxys(self, uvs):
        a,b,c,d = self.st_abcd
        # A = np.array(((a,b),(c,d)), dtype=np.float64)
        stxys = np.zeros((uvs.shape[0], 2), dtype=np.float64)
        sh = self.st_shift
        stxys[:,0] = a*uvs[:,0]+b*uvs[:,1]+sh[0]
        stxys[:,1] = c*uvs[:,0]+d*uvs[:,1]+sh[1]
        return stxys

    def stxyToUv(self, stxy):
        stxy = stxy - self.st_shift
        a,b,c,d = self.st_abcd
        det = a*d-b*c
        u = (d*stxy[0] - b*stxy[1])/det
        v = (-c*stxy[0] + a*stxy[1])/det
        return (u,v)
        
    def stxysToUvs(self, stxys):
        stxys = stxys.copy() - self.st_shift
        a,b,c,d = self.st_abcd
        print("stu", a,b,c,d)
        det = a*d-b*c
        uvs = np.zeros((stxys.shape[0], 2), dtype=np.float64)
        uvs[:,0] = (d*stxys[:,0] - b*stxys[:,1])/det
        uvs[:,1] = (-c*stxys[:,0] + a*stxys[:,1])/det
        return uvs
        
    def setVolumeViewDirection(self, direction):
        self.setWorkingRegion(-1, 0.)
        self.prev_pt_count = 0
        super(TrglFragmentView, self).setVolumeViewDirection(direction)
        
    def setVolumeView(self, vol_view):
        self.setWorkingRegion(-1, 0.)
        self.prev_pt_count = 0
        super(TrglFragmentView, self).setVolumeView(vol_view)

    def pushFragmentState(self):
        pass

    def popFragmentState(self):
        pass

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
        print("outsidePoints arr", id0, idn, arr.shape)
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
        all_pts = self.all_stpoints
        # print("ra", len(stp), len(all_pts), len(self.trgls()))

        all_trgls = None
        try:
            all_trgls = Delaunay(all_pts).simplices
        except Exception as err:
            err = str(err).splitlines()[0]
            print("retriangulateAll triangulation error: %s"%err)
        # print("allt", len(all_trgls))
        if all_trgls is not None:
            new_trgls = all_trgls[(all_trgls < len(stp)).all(axis=1), :]
            # print("newt", len(new_trgls))
            # new_trgls = all_trgls
            self.fragment.trgls = TrglPointSet.rotateToMin(new_trgls)
            # self.fragment.trgls = new_trgls
        # print("ra2", len(stp), len(all_pts), len(self.trgls()))

    def rebuildStPoints(self):
        # self.setLocalPoints(True, False)
        self.stpoints = None
        print("rsp set stpoints to None")
        self.setScaledTexturePoints()
        self.fragment.notifyModified()

    def reparameterize(self):
        self.stpoints = None
        print("rpm set stpoints to None")
        self.setScaledTexturePoints()
        xyzs = self.vpoints[:,0:3]
        trgls = self.trgls()
        # print("rt before")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        TrglPointSet.findSpikes(xyzs, trgls, "before reparameterize")
        # print("rt after findSpikes")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        self.disconnectColocatedPoints()
        # print("rt after coloc")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        self.deleteDisconnectedComponents()
        # print("rt after disconn")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        self.deleteFreePoints()
        # print("rt after free points")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        xyzs = self.vpoints[:,0:3]
        # txyzs is array[trgl #][trgl pt (0, 1, or 2)][pt xyz]
        trgls = self.trgls()
        # print("rt before mapper")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])
        mapper = UVMapper(xyzs, trgls)
        # print("rt after mapper")
        # print(self.trgls()[(self.trgls()==181).any(axis=1)])

        # set the floating points, and the window-boundary points,
        # as constraints (the floating points need to be constrained,
        # though they have no effect, otherwise the parameterizer
        # may complain of a singular matrix)
        # constraints = np.zeros((bpts.shape[0], 3), dtype=np.float64)
        # constraints[:, 0] = bpts
        # constraints[:, (1,2)] = pts[bpts]

        # mapper = UVMapper(self.fragment.gtpoints, self.trgls())
        ta = mapper.getTwoAdjacentBoundaryPoints()
        if ta is None:
            print("reparameterize: could not find boundary points!")
            return
        pt0, pt1 = ta
        mapper.constraints = np.array([[pt0, 0., 0.], [pt1, 1., 0.]], dtype=np.float64)
        weight = .000001
        mapper.ip_weights = np.full(self.stpoints.shape[0], weight)
        mapper.initial_points = self.fragment.gtpoints
        adjusted_sts = mapper.computeUvsFromABF()
        if adjusted_sts is None:
            print("reparameterize failed!")
            return
        self.fragment.gtpoints = adjusted_sts
        # print(adjusted_sts)
        self.stpoints = None
        # TrglPointSet.findSpikes(xyzs, trgls, "after reparam")
        # print("reparameterize set stpoints to None")
        self.setScaledTexturePoints()
        TrglPointSet.findSpikes(xyzs, trgls, "after reparameterize")
        # print(self.stpoints)
        self.fragment.notifyModified()


    # This depends on self.fragment.trgls being
    # up to date
    def adjustStPoints(self, index, half_width, stxy=None):
        if stxy is None:
            stxy = self.all_stpoints[index]
        # print("stxy", stxy)
        osts = TrglPointSet(self.all_stpoints, len(self.stpoints), stxy, half_width)
        # print("osts indexes", osts.indexes)
        retval = osts.adjustSts(self.fragment.gpoints, self.fragment.trgls, index)
        if retval is None:
            print("Adjustment failed")
            return None

        adjusted_inds, adjusted_sts, constrained = retval
        # print(len(adjusted_sts))
        self.stpoints[adjusted_inds] = adjusted_sts
        self.all_stpoints[adjusted_inds] = adjusted_sts
        adj_uvs = self.stxysToUvs(adjusted_sts)
        self.fragment.gtpoints[adjusted_inds] = adj_uvs
        print("Adjustment done")
        return constrained

    def movePoint(self, index, new_vijk, update_xyz, update_st):
        # print("mp")
        vv = self.cur_volume_view
        new_gijk = vv.transposedIjkToGlobalPosition(new_vijk)
        new_uijk = vv.transposedIjkToIjk(new_vijk)
        old_vijk = self.vpoints[index, :3]
        old_uijk = vv.transposedIjkToIjk(old_vijk)
        duijk = [new_uijk[i]-old_uijk[i] for i in range(3)]
        # print("movePoint", dvijk, dgijk, duijk)
        # print("a")
        axes = self.localStAxes(index)
        # print("b")
        if axes is None:
            print("TrglFragmentView.movePoint: could not compute axes")
            axes = np.zeros((3,3), dtype=np.float64)
            # return
        rduijk = (axes.T)@duijk
        old_stxy = self.all_stpoints[index]
        new_stxy = old_stxy+rduijk[:2]
        # print(self.fragment.gpoints)
        # print(match, new_gijk)

        if update_st and (new_stxy != old_stxy).all() and self.pointExists(new_stxy):
            print("move: point already exists")
            return

        if update_xyz:
            self.fragment.gpoints[index, :] = new_gijk
            # print("c")
            self.setLocalPoints(True, False)
            # print("d")

        if update_st:
            # call instead of setting self.stpoints
            # self.retriangulateAndMove(index, new_stxy)

            half_width = self.half_width_multiplier*self.avg_st_len
            # print("hw", half_width, self.avg_st_len)


            ops = TrglPointSet(self.all_stpoints, len(self.stpoints), new_stxy, half_width)

            self.stpoints[index, :] = new_stxy
            self.all_stpoints[index, :] = new_stxy
            uv = self.stxyToUv(new_stxy)
            self.fragment.gtpoints[index, :] = uv
            # self.stpoints[index, :] = new_stxy
            constrained = self.adjustStPoints(index, half_width)

            nps = TrglPointSet(self.all_stpoints, len(self.stpoints), new_stxy, half_width)
            self.applyTrglDiff(ops, nps)
            if not constrained:
                # self.stpoints = None
                # self.setLocalPoints(True, False)
                self.rebuildStPoints()

        self.fragment.notifyModified()
        return True

    def applyTrglDiff(self, ops, nps):
        result = TrglPointSet.trglDiff(ops, nps)
        if result is not None:
            otrgls, ntrgls = result
            print("otrgls", len(otrgls), self.maxEdgeLengthTrgls(otrgls))
            print("ntrgls", len(ntrgls), self.maxEdgeLengthTrgls(ntrgls))
            if len(otrgls) > 0 or len(ntrgls) > 0:
                # print("uo", otrgls)
                # print("un", ntrgls)
                # self.replaceTrgls(otrgls, ntrgls)
                self.fragment.trgls = TrglPointSet.replaceTrgls(self.fragment.trgls, otrgls, ntrgls)

    def pointExists(self, stxy):
        existing = np.nonzero((self.stpoints == stxy).all(axis=1))[0]
        return len(existing)

    def maxEdgeLengthTrgls(self, trgls):
        return TrglPointSet.maxEdgeLength(self.fragment.gpoints, trgls)

    def maxEdgeLength(self, tps):
        trgls = tps.triangulate()
        return TrglPointSet.maxEdgeLength(self.fragment.gpoints, trgls)

    def maxEdgeLengthAll(self):
        return TrglPointSet.maxEdgeLength(self.fragment.gpoints, self.fragment.trgls)

    def addPoint(self, tijk, stxy):
        # print("tf add point", tijk, stxy)
        if stxy is None:
            print("TrglFragment.addPoint failed because stxy not given")
            return
        if tijk is None:
            print("TrglFragment.addPoint failed because tijk not given")
            return
        print("a before", self.maxEdgeLengthAll())

        vv = self.cur_volume_view
        gijk = vv.transposedIjkToGlobalPosition(tijk)

        # existing = np.nonzero((self.stpoints == stxy).all(axis=1))[0]
        # if len(existing > 0):
        #     print("Point already exists at stxy", stxy)
        #     return
        if self.pointExists(stxy):
            print("Point already exists at stxy", stxy)
            return

        self.fragment.gpoints = np.append(self.fragment.gpoints, [gijk], axis=0)
        uv = self.stxyToUv(stxy)
        nstp = len(self.stpoints)
        self.fragment.gtpoints = np.append(self.fragment.gtpoints, [uv], axis=0)
        self.stpoints = np.append(self.stpoints, [stxy], axis=0)
        self.all_stpoints = np.insert(self.all_stpoints, nstp, stxy, axis=0)

        astxy = np.array(stxy)
        # agijk = np.array(gijk)
        half_width = self.half_width_multiplier*self.avg_st_len
        ops = TrglPointSet(self.all_stpoints, len(self.stpoints), astxy, half_width)
        ops.deletePoint(nstp)
        print("a ops", self.maxEdgeLength(ops))

        # Can't do this here; self.fragment.trgls is not
        # up to date yet (new point hasn't been added)
        # constrained = self.adjustStPoints(nstp, half_width)

        nps = TrglPointSet(self.all_stpoints, len(self.stpoints), astxy, half_width)
        print("a nps", self.maxEdgeLength(nps))
        # nps.addPointAtEnd(stxy)

        self.applyTrglDiff(ops, nps)
        constrained = self.adjustStPoints(nstp, half_width)
        nps2 = TrglPointSet(self.all_stpoints, len(self.stpoints), astxy, half_width)
        print("a nps2", self.maxEdgeLength(nps2))
        self.applyTrglDiff(nps, nps2)

        # tcount will be zero if the new point has no triangles,
        # non-zero otherwise
        tcount = (self.fragment.trgls==nstp).any(axis=1).sum()
        # tcount = (self.fragment.trgls==nstp).sum()
        # print("TrglFragment addPoint tcount", tcount)
        if tcount > 0 and constrained:
            # prevent setScaledTexturePoints from running
            # when setLocalPoints is called
            self.prev_pt_count = len(self.fragment.gpoints)

        self.setLocalPoints(True, False)
        print("a after", self.maxEdgeLengthAll())
        self.fragment.notifyModified()

    # If two points are colocated in xyz, disconnect
    # one of them from all the trgls, replacing it by
    # the other.
    # The disconnected point can be deleted in
    # another function
    def disconnectColocatedPoints(self):
        if self.stpoints is None:
            return
        trgls = self.fragment.trgls
        if len(trgls) == 0:
            return
        pts = self.fragment.gpoints
        npt = pts.shape[0]
        # print("npt", npt)
        # print("min edge length", TrglPointSet.minEdgeLength(pts,trgls))
        # This is wrong, and caused annoying pyramids:
        # lind = np.lexsort((pts[:,1], pts[:,0]))
        lind = np.lexsort((pts[:,2], pts[:,1], pts[:,0]))
        rlind = np.zeros(npt, dtype=np.int64)
        rlind[lind] = np.ogrid[:npt]
        sarr = pts[lind]
        value,inds,counts = np.unique(sarr, return_index=True, return_counts=True, axis=0)
        tinds = np.repeat(inds, counts)
        ndup = np.sum(counts-1)
        if ndup > 0:
            print("found",ndup,"colocated point(s)")
        dedup = lind[tinds[rlind]]
        # print("a", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        trgls = dedup[trgls]
        # print("b", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        trgls = trgls[trgls[:,0] != trgls[:,1]]
        # print("c", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        trgls = trgls[trgls[:,1] != trgls[:,2]]
        # print("d", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        trgls = trgls[trgls[:,2] != trgls[:,0]]
        # print("e", len(trgls))
        # print(trgls[(trgls==181).any(axis=1)])
        self.fragment.trgls = trgls

    # This will create free points by deleting the
    # triangles that hold them
    def deleteDisconnectedComponents(self):
        if self.stpoints is None:
            return
        trgls = self.fragment.trgls
        if len(trgls) < 3:
            return
        neighbors = BaseFragment.findNeighbors(trgls)
        '''
        gpoints = self.fragment.gpoints
        # testinds = np.nonzero(gpoints[:,2] == 10536)
        # print("testinds", testinds)
        # print(gpoints[testinds])
        testinds = np.nonzero(trgls==281)[0]
        print(testinds)
        print(trgls[testinds])
        print(neighbors[testinds])
        # print("trgls")
        # print(trgls[1774:1780])
        # print("neighbors")
        # print(neighbors[1774:1780])
        '''
        nt = trgls.shape[0]
        # print("t n", trgls.shape, neighbors.shape)
        # each triangle also has itself as a neighbor
        # (this is so that triangles with no neighbors will still
        # show up in the connectivity graph)
        neighbors = np.append(neighbors, np.ogrid[:nt,:0][0], axis=1)
        tindex = np.ogrid[:4*nt]//4
        nindex = neighbors.flatten()
        is_valid = (nindex > -1)
        tindex = tindex[is_valid]
        nindex = nindex[is_valid]
        ones = np.full(nindex.shape[0], 1)
        connections = scipy.sparse.csr_array((ones, (tindex, nindex)), shape=(nt,nt))
        # print("connections")
        # print(connections)
        nc, labels = scipy.sparse.csgraph.connected_components(connections, directed=False)
        if nc > 1:
            print("number of components", nc)
            hist = np.zeros(nc, dtype=np.int64)
            np.add.at(hist, labels, 1)
            print("hist", hist)
            mainind = np.argmax(hist)
            keep_trgl = (labels==mainind)
            self.fragment.trgls = trgls[keep_trgl]

    def deleteFreePoints(self):
        if self.stpoints is None:
            return
        trgls = self.fragment.trgls
        # Don't delete free points unless there is at
        # least one trgl
        if len(trgls) == 0:
            return
        npt = len(self.stpoints)
        free_flag = np.full(npt, True, dtype=np.bool_)
        free_flag[trgls.flatten()] = False
        nf = free_flag.sum()
        if nf == 0:
            return
        print(nf, "free points")
        nc = npt-nf
        o2n = np.full(npt, -1, dtype=np.int64)
        o2n[~free_flag] = np.ogrid[:nc]
        # fp_index = np.nonzero(free_flag)[0]
        self.fragment.trgls = o2n[trgls]
        old_outside = self.stpoints[npt:].copy()
        self.fragment.gpoints = self.fragment.gpoints[~free_flag]
        self.fragment.gtpoints = self.fragment.gtpoints[~free_flag]
        self.stpoints = self.stpoints[~free_flag]
        self.all_stpoints = np.concatenate((self.stpoints, old_outside))
        self.setLocalPoints(True, False)

    def deletePointByIndex(self, index):
        if index < 0:
            return
        if index >= len(self.fragment.gpoints):
            return
        if self.stpoints is None or index >= len(self.stpoints):
            return
        half_width = self.half_width_multiplier*self.avg_st_len

        old_stxy = self.all_stpoints[index]
        ops = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        nps = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        nps.deletePoint(index)
        # Retriangulate before deleting point from self.stpoints etc
        self.applyTrglDiff(ops, nps)
        self.fragment.trgls[self.fragment.trgls>index] -= 1

        self.fragment.gpoints = np.delete(self.fragment.gpoints, index, 0)
        self.fragment.gtpoints = np.delete(self.fragment.gtpoints, index, 0)
        self.stpoints = np.delete(self.stpoints, index, 0)
        self.all_stpoints = np.delete(self.all_stpoints, index, 0)

        ops = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        constrained = self.adjustStPoints(-1, half_width, old_stxy)
        nps = TrglPointSet(self.all_stpoints, len(self.stpoints), old_stxy, half_width)
        self.applyTrglDiff(ops, nps)

        # prevent setScaledTexturePoints from running
        # when setLocalPoints is called
        if constrained:
            self.prev_pt_count = len(self.fragment.gpoints)
        
        self.setLocalPoints(True, False)
        self.fragment.notifyModified()

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

    # Create a TrglPointSet that contains all stpoints
    # that lie in a square whose center and half-width are given
    def __init__(self, all_stpoints, nstpoints, pt, half_width):
        # TODO: this isn't valid!
        # if all_stpoints is None or len(all_stpoints) == 0:
        #     return None
        ptsb = ((pt-half_width <= all_stpoints) & 
                (all_stpoints <= pt+half_width)).all(axis=1)
        self.indexes = np.nonzero(ptsb)[0]
        self.reverse_indexes = np.full((len(all_stpoints)), -1, dtype=np.int64)
        self.reverse_indexes[ptsb] = np.ogrid[:ptsb.sum()]
        self.pts = all_stpoints[self.indexes]
        # number of normal (not outside) points in all_stpoints
        self.nstpoints = nstpoints
        # bs = np.nonzero(self.indexes < nstpoints)[0]
        # self.nipoints = bs[-1]+1
        # self.nipoints = bs.shape[0]

        # number of normal (not outside) points in self.indexes
        self.nipoints = (self.indexes < nstpoints).sum()


    def cutBoundaryPoints(self, trgls):
        ritrgls = self.reverse_indexes[trgls]
        ctrgls = ritrgls[((ritrgls<0).sum(axis=1)+1)//2 == 1]
        # print("ctrgls")
        # print(ctrgls)
        # print("cbp", trgls.shape, ritrgls.shape, ctrgls.shape)
        # print(ctrgls)
        bpts = ctrgls[ctrgls>=0].flatten()
        ubpts = np.unique(bpts)
        return ubpts

    def deletePoint(self, index):
        row = (self.indexes == index).nonzero()[0]
        # print("before", self.indexes)
        self.indexes = np.delete(self.indexes, row, 0)
        # print("after", self.indexes)
        self.pts = np.delete(self.pts, row, 0)

    @staticmethod
    def trglDiff(oldps, newps):
        # print("doing old trgls")
        old_trgls = oldps.triangulate()
        # print("doing new trgls")
        new_trgls = newps.triangulate()
        # print("None", old_trgls is None, new_trgls is None)

        # if old_trgls is None or new_trgls is None:
        #     return None
        if old_trgls is None:
            old_trgls = np.zeros((0,3), dtype=np.int64)
        if new_trgls is None:
            new_trgls = np.zeros((0,3), dtype=np.int64)

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

    @staticmethod
    def edgeLengths(pts, trgls):
        if trgls is None or len(trgls) == 0:
            return np.zeros((0,3), dtype=np.float64)
        tpts = pts[trgls]
        dtpts = tpts - np.roll(tpts, 1, axis=1)
        dsq = (dtpts*dtpts).sum(axis=2)
        return np.sqrt(dsq)

    # pts should be xyz points, not uv points
    @staticmethod
    def maxEdgeLength(pts, trgls):
        '''
        if trgls is None or len(trgls) == 0:
            return 0.
        tpts = pts[trgls]
        dtpts = tpts - np.roll(tpts, 1, axis=1)
        dsq = (dtpts*dtpts).sum(axis=2)
        maxdsq = np.max(dsq)
        return np.sqrt(maxdsq)
        '''
        lens = TrglPointSet.edgeLengths(pts, trgls)
        if len(lens) == 0:
            return 0.
        return np.max(lens)

    # pts should be xyz points, not uv points
    @staticmethod
    def minEdgeLength(pts, trgls):
        '''
        if trgls is None or len(trgls) == 0:
            return 0.
        tpts = pts[trgls]
        dtpts = tpts - np.roll(tpts, 1, axis=1)
        dsq = (dtpts*dtpts).sum(axis=2)
        mindsq = np.min(dsq)
        return np.sqrt(mindsq)
        '''
        lens = TrglPointSet.edgeLengths(pts, trgls)
        if len(lens) == 0:
            return 0.
        return np.min(lens)

    '''
    # pts should be xyz points, not uv points
    # returns ntrgl*3 array of the 3 angles around each trgl
    @staticmethod
    def computeAngles(self, pts, trgls):
        if pts is None or len(pts) < 3:
            return None
        if trgls is None or len(trgls) == 0:
            return None
        tpts = pts[trgls]
        # d02, d10, d21
        # i.e. vector from pt 0 to pt 2, etc
        tvecs = np.roll(tpts, 1, axis=1) - tpts
        tlens = np.sqrt((tvecs*tvecs).sum(axis=2))
        tlens[tlens[:,:]==0] = 1.

        # normalized vecs
        tnvecs = tvecs/tlens[:,:,np.newaxis]

        # dot products of normalized vectors
        tndps = (-tnvecs*np.roll(tnvecs, -1, axis=1)).sum(axis=2)
        angles = np.arccos(tndps)
        return angles

    @staticmethod
    def boundaryPoints(self, trgls):

    @staticmethod
    def computeAngleDeficits(self, pts, trgls):
        if pts is None or len(pts) < 3:
            return None
        if trgls is None or len(trgls) == 0:
            return None
        angles = self.computeAngles(pts, trgls)
        if angles is None:
            return None
        sums = np.zeros(len(points), dtpye=np.float64)
        is_on_boundary = self.onBoundaryArray()
    '''

    # pts should be xyz points, not uv points
    @staticmethod
    def findSpikes(pts, trgls, txt=""):
        mapper = UVMapper(pts, trgls)
        mapper.createAngles()
        sums = mapper.sumAnglesAroundPoints()
        on_boundary = mapper.onBoundaryArray()
        sums[on_boundary] = 100.
        inds = np.argsort(sums)
        minangle = 3.14
        dinds = inds[sums[inds] < minangle]
        if len(dinds) > 0:
            print("Spikes",txt)
            print(dinds)
            print(sums[dinds])
            print(pts[dinds])


    def triangulate(self):
        trgls = None
        try:
            trgls = Delaunay(self.pts).simplices
        except Exception as err:
            err = str(err).splitlines()[0]
            print("trglDiff triangulation error: %s"%err)
            return None

        nst = self.nstpoints
        # Besides rotating to min pt index, this line
        # replaces the local pt index by the global pt index
        # trgls = self.rotateToMin(np.array(self.indexes)[trgls])
        trgls = self.rotateToMin(self.indexes[trgls])
        # Remove trgls that contain 1 or more outside points
        trgls = trgls[(trgls < nst).all(axis=1)]
        return trgls

    # xyzpts is all the fragment points (gpoints),
    # trgls is all the trgls,
    # ptindex is the index (into the list of all fragment points)
    # of the point that is being moved.
    def adjustSts(self, xyzpts, trgls, ptindex):
        if trgls is None:
            return
        if len(self.pts) == 0:
            return
        # indices (relative to fragment point list) 
        # of windowed non-outside points
        inds = self.indexes[:self.nipoints]
        # stxy locations of windowed non-outside points
        pts = self.pts[:self.nipoints]
        # print("inds", inds)
        # print("xyzpts", xyzpts.shape)
        # xyz locations of windowed non-outside points
        localxyz = xyzpts[inds]
        # Don't triangulate here; the local convex
        # hull would not be desirable!
        # trgls = self.triangulate()
        # print(trgls)

        # trgls converted to use windowed-point indexing
        ltrgls = self.reverse_indexes[trgls]
        # eliminate trgls that have points outside of the
        # window 
        ltrgls = ltrgls[(ltrgls >= 0).all(axis=1)]
        """
        bounds_flag = mapper.onBoundaryArray()
        # input point is never used as a constraint
        if ptindex >= 0:
            bounds_flag[self.reverse_indexes[ptindex]] = False

        '''
        ptrgls = trgls.copy()
        ptrgls[bounds_flag[ptrgls]] *= -1
        print(ptrgls)
        '''

        nb = bounds_flag.sum()
        constraints = np.zeros((nb, 3), dtype=np.float64)

        # print(bounds_flag)
        # print(xyzpts.shape, localxyz.shape, self.pts.shape, pts.shape, constraints.shape)
        constraints[:, 0] = np.nonzero(bounds_flag)[0]
        constraints[:, (1,2)] = pts[bounds_flag]
        """

        """
        # floating points (points that are not part of
        # any trgl), using windowed-point indexing:
        # print("ltrgls")
        # print(ltrgls)
        floating = np.full(pts.shape[0], True, dtype=np.bool_)
        floating[ltrgls.flatten()] = False

        # floating = np.logical_not(np.isin(inds, ltrgls.flatten()))
        # print("floating", floating.sum(),"of",len(inds))
        print("floating", floating.sum(), np.nonzero(floating)[0])
        """

        # bpts contains points that are on the boundaries
        # of cut triangles.  The points use windowed-point indexing
        bpts = self.cutBoundaryPoints(trgls)

        """
        # print("bpts floating", bpts.shape, floating.shape, floating.sum())
        # add to "floating" the points on boundaries created
        # by windowing
        bfloating = floating.copy()
        bfloating[bpts] = True
        bpts = np.nonzero(bfloating)[0]
        # print("bpts floating again", bpts.shape, bfloating.shape, bfloating.sum())
        # print(bpts.shape)

        # print("bpts")
        # print(bpts)
        # lbpts = self.indexes[bpts]
        # print("lbpts")
        # print(lbpts)


        # set the floating points, and the window-boundary points,
        # as constraints (the floating points need to be constrained,
        # though they have no effect, otherwise the parameterizer
        # may complain of a singular matrix)
        """

        mapper = UVMapper(localxyz, ltrgls)

        if ptindex >= 0:
            bpts = bpts[bpts != self.reverse_indexes[ptindex]]
        print("cut bpts", len(bpts))
        constraints = np.zeros((bpts.shape[0], 3), dtype=np.float64)
        constraints[:, 0] = bpts
        constraints[:, (1,2)] = pts[bpts]

        mapper.constraints = constraints
        weight = .5
        mapper.ip_weights = np.full(pts.shape[0], weight)
        if ptindex >= 0:
            mapper.ip_weights[self.reverse_indexes[ptindex]] = 0.
        mapper.initial_points = pts
        # adjusted_sts = mapper.computeUvsFromABF()
        # print(pts.shape[0], nb)
        adjusted_sts = mapper.computeUvsFromXyzs()
        # Would prefer to this because it may handle flipped triangles
        # better than computeUvsFromXyzs(), but it
        # doesn't seem as stable
        # adjusted_sts = mapper.computeUvsFromAngles()
        if adjusted_sts is None:
            print("adjustSts: parameterization failed, no adjustment made")
            # return inds, pts
            return None
        # print("before", pts[~bounds_flag])
        # print("after", adjusted_sts[~bounds_flag])
        # print("delta", (adjusted_sts-pts)[~bounds_flag])
        # adjusted_sts[floating] = pts[floating]
        '''
        delta = adjusted_sts-pts
        delta = np.sqrt((delta*delta).sum(axis=1))
        delta[bpts] *= -1
        np.set_printoptions(suppress=True)
        # print(delta[np.abs(delta)>1000])
        # print(np.nonzero(np.abs(delta)>1000)[0])
        bigds = np.nonzero(np.abs(delta)>1000)[0]
        print("bigds", bigds)
        print(adjusted_sts[bigds])
        for bigd in bigds.tolist():
            print("  ", bigd)
            print(trgls[(trgls==bigd).any(axis=1)])
        '''
        return inds, adjusted_sts, len(bpts) > 1


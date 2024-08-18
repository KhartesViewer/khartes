import math
import json
import numpy as np
from pathlib import Path
from collections import deque
from utils import Utils
from base_fragment import BaseFragment, BaseFragmentView
from fragment import Fragment, FragmentView

from PyQt5.QtGui import QColor

class TrglFragment(BaseFragment):
    def __init__(self, name):
        super(TrglFragment, self).__init__(name)
        self.gpoints = np.zeros((0,3), dtype=np.float32)
        self.tpoints = np.zeros((0,2), dtype=np.float32)
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
        trgl_frag.gpoints = np.array(vrtl, dtype=np.float32)
        trgl_frag.tpoints = np.array(tvrtl, dtype=np.float32)
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
        print(trgl_frag.name, trgl_frag.color.name(), trgl_frag.gpoints.shape, trgl_frag.tpoints.shape, trgl_frag.trgls.shape)
        # print("tindexes", BaseFragment.trglsAroundPoint(100, trgl_frag.trgls))

        return [trgl_frag]

    def createView(self, project_view):
        return TrglFragmentView(project_view, self)

    def createCopy(self, name):
        frag = TrglFragment(name)
        frag.setColor(self.color, no_notify=True)
        frag.gpoints = np.copy(self.gpoints)
        frag.tpoints = np.copy(self.tpoints)
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
        has_texture = (len(self.tpoints) == len(self.gpoints))
        if has_texture:
            for i, pt in enumerate(self.tpoints):
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
        if len(trgl_fragment.trgls) == 0:
            self.mesh_visible = False

    # TODO: if cur_volume_view changed, unset working region
    def setLocalPoints(self, recursion_ok=True, always_update_zsurfs=True):
        if self.cur_volume_view is None:
            self.vpoints = np.zeros((0,4), dtype=np.float32)
            self.fpoints = self.vpoints
            # self.working_vpoints = np.zeros((0,4), dtype=np.float32)
            self.setWorkingRegion(-1, 0.)
            return
        self.vpoints = self.cur_volume_view.globalPositionsToTransposedIjks(self.fragment.gpoints)
        self.fpoints = self.vpoints
        self.fragment.direction = self.cur_volume_view.direction
        npts = self.vpoints.shape[0]
        if npts > 0:
            indices = np.reshape(np.arange(npts), (npts,1))
            # print(self.vpoints.shape, indices.shape)
            self.vpoints = np.concatenate((self.vpoints, indices), axis=1)
        self.calculateSqCm()
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
        
    def setVolumeViewDirection(self, direction):
        self.setWorkingRegion(-1, 0.)
        super(TrglFragmentView, self).setVolumeViewDirection(direction)
        
    def setVolumeView(self, vol_view):
        self.setWorkingRegion(-1, 0.)
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


    def getPointsOnSlice(self, axis, i):
        # matches = self.vpoints[(self.vpoints[:, axis] == i)]
        matches = self.vpoints[(self.vpoints[:, axis] >= i-.5) & (self.vpoints[:, axis] < i+.5)]
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

    def movePoint(self, index, new_vijk):
        new_gijk = self.cur_volume_view.transposedIjkToGlobalPosition(new_vijk)
        # print(self.fragment.gpoints)
        # print(match, new_gijk)
        self.fragment.gpoints[index, :] = new_gijk
        # print(self.fragment.gpoints)
        self.fragment.notifyModified()
        # modifyZsurf = False
        # if len(self.workingVpoints()) > 0 and self.workingVpoints()[index]:
        #     modifyZsurf = True
        self.setLocalPoints(True, False)
        return True

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









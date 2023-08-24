from utils import Utils
import numpy as np

from PyQt5.QtGui import QColor

class BaseFragment:
    def __init__(self, name):
        self.name = name
        self.color = QColor()
        self.cvcolor = (0,0,0,0)
        self.created = Utils.timestamp()
        self.modified = Utils.timestamp()
        self.valid = False
        self.project = None

    def notifyModified(self, tstamp=""):
        if tstamp == "":
            tstamp = Utils.timestamp()
        self.modified = tstamp
        # self.project can be None if BaseFragment is
        # a working Fragment of a TrglFragment
        if self.project is not None:
            self.project.notifyModified(tstamp)

    def setColor(self, qcolor, no_notify=False):
        self.color = qcolor
        rgba = qcolor.getRgbF()
        self.cvcolor = [int(65535*c) for c in rgba] 
        if not no_notify:
            self.notifyModified()

    def createView(self, project_view):
        print("BaseFragment: need to implement this class!")
        return None

    # class function
    def saveList(frags, path, stem):
        class_lists = {}
        for frag in frags:
            print("bsl", frag.name)
            # print(type(frag))
            t = type(frag)
            # t.asdf()
            l = class_lists.setdefault(t, [])
            l.append(frag)
        for cl, l in class_lists.items():
            cl.saveList(l, path, stem)

    def meshExportNeedsInfill(self):
        return False

    # class function
    def saveListAsObjMesh(fvs, path, infill):
        class_lists = {}
        for fv in fvs:
            frag = fv.fragment
            print("bsl", frag.name)
            # print(type(frag))
            t = type(frag)
            # t.asdf()
            l = class_lists.setdefault(t, [])
            l.append(fv)
        for cl, l in class_lists.items():
            err = cl.saveListAsObjMesh(l, path, infill, len(class_lists.items()))
            if err != "":
                return err
        return ""

    # class function
    # returns normals at points
    def pointNormals(pts, trgls):
        v0 = trgls[:,0]
        v1 = trgls[:,1]
        v2 = trgls[:,2]
        d01 = (pts[v1] - pts[v0]).astype(np.float64)
        d02 = (pts[v2] - pts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        ptn = np.zeros((len(pts), 3), np.float32)
        ptn[v0] += n3d
        ptn[v1] += n3d
        ptn[v2] += n3d
        l2 = np.sqrt(np.sum(ptn*ptn, axis=1)).reshape(-1,1)
        l2[l2==0] = 1.
        ptn /= l2
        return ptn

    # class function
    # returns normals of triangles
    def faceNormals(pts, trgls):
        v0 = trgls[:,0]
        v1 = trgls[:,1]
        v2 = trgls[:,2]
        d01 = (pts[v1] - pts[v0]).astype(np.float64)
        d02 = (pts[v2] - pts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        l2 = np.sqrt(np.sum(n3d*n3d, axis=1)).reshape(-1,1)
        l2[l2==0] = 1.
        n3d /= l2
        return n3d

    # class function
    def calculateSqCm(pts, trgls, voxel_size_um):
        v0 = trgls[:,0]
        v1 = trgls[:,1]
        v2 = trgls[:,2]
        d01 = (pts[v1] - pts[v0]).astype(np.float64)
        d02 = (pts[v2] - pts[v0]).astype(np.float64)
        n3d = np.cross(d01, d02)
        l2 = np.sqrt(np.sum(n3d*n3d, axis=1))
        area_sq_mm_trg = np.sum(l2)*voxel_size_um*voxel_size_um/(2*1000000)
        sqcm = area_sq_mm_trg/100.
        return sqcm

    # class function
    def findNeighbors(trgls):
        index = np.indices((len(trgls),1))[0]
        ones = np.ones((len(trgls),1), dtype=np.int32)
        zeros = np.zeros((len(trgls),1), dtype=np.int32)
        twos = 2*ones
        
        e01 = np.concatenate((trgls[:, (0,1)], index, twos, ones), axis=1)
        e12 = np.concatenate((trgls[:, (1,2)], index, zeros, ones), axis=1)
        e20 = np.concatenate((trgls[:, (2,0)], index, ones, ones), axis=1)
        
        edges = np.concatenate((e01,e12,e20), axis=0)
        rev = (edges[:,0] > edges[:,1])
        edges[rev,0:2] = edges[rev,1::-1]
        edges[rev,4] = -1
        edges = edges[edges[:,4].argsort()]
        edges = edges[edges[:,1].argsort(kind='mergesort')]
        edges = edges[edges[:,0].argsort(kind='mergesort')]
        
        ediff = np.diff(edges, axis=0)
        duprows = np.where(((ediff[:,0]==0) & (ediff[:,1]==0)))[0]
        duprows2 = np.sort(np.append(duprows, duprows+1))
        bdup = np.zeros((len(edges)), dtype=np.bool_)
        bdup[duprows2] = True
        
        neighbors = np.full((len(trgls), 3), -1, dtype=np.int32)
        
        eplus = edges[duprows+1,:4]
        eminus = edges[duprows,:4]
        # print(eplus)
        # print(eminus)
        neighbors[eplus[:,2],eplus[:,3]] = eminus[:,2]
        neighbors[eminus[:,2],eminus[:,3]] = eplus[:,2]
        return neighbors

    # returns list of indexes of those trgls that have pt_index as
    # a vertex
    def trglsAroundPoint(pt_index, trgls):
        bvec = (trgls[:,0] == pt_index) | (trgls[:,1] == pt_index) | (trgls[:,2] == pt_index)
        tindexes = np.where(bvec)[0]
        return tindexes.tolist()

class BaseFragmentView:
    def __init__(self, project_view, fragment):
        self.project_view = project_view
        self.fragment = fragment
        self.sqcm = 0.
        self.cur_volume_view = None
        self.visible = True
        self.active = False

    def setVolumeView(self, vol_view):
        if vol_view == self.cur_volume_view:
            return
        self.cur_volume_view = vol_view
        if vol_view is not None:
            self.setLocalPoints(False)

    def notifyModified(self, tstamp=""):
        if tstamp == "":
            tstamp = Utils.timestamp()
        self.modified = tstamp
        # print("fragment view", self.fragment.name,"modified", tstamp)
        self.project_view.notifyModified(tstamp)

    def getZsurfPoints(self, axis, axis_pos):
        return None

    def line(self):
        return None

    def getLinesOnSlice(self, axis, axis_pos):
        return None, None

    def triangulate(self):
        return None

    def addPoint(self, tijk):
        return None

    def deletePointByIndex(self, index):
        return None

    def setLiveZsurfUpdate(self, flag):
        return None

    def setWorkingRegion(self, index, max_angle):
        return None

    def workingZsurf(self):
        return None

    def workingSsurf(self):
        return None

    def workingVpoints(self):
        return np.zeros((0,4), dtype=np.bool_)

    def workingTrgls(self):
        return None

    def hasWorkingNonWorking(self):
        return (False, False)

    def workingLine(self):
        return None

    def workingLineAxis(self):
        return -1

    def workingLineAxisPosition(self):
        return 0

    def activeAndAligned(self):
        if not self.active:
            return False
        return self.aligned()

    # direction is not used here, but this notifies fragment view
    # to recompute things
    def setVolumeViewDirection(self, direction):
        self.setLocalPoints(False)
    
    def pointNormals(self):
        self.triangulate()
        trgls = self.trgls()
        if trgls is None:
            return
        pts3d = self.fpoints[:,:3]
        return BaseFragment.pointNormals(pts3d, trgls)
        '''
        # if self.tri is None or len(self.tri.simplices) == 0:
        #     return None
        # zpts = self.fpoints[:,2]
        # pts3d = np.append(self.tri.points, zpts.reshape(-1,1), axis=1)
        pts3d = self.fpoints[:,:3]
        # print("n",self.tri.points.shape, zpts.shape, pts3d.shape)
        v0 = trgls[:,0]
        v1 = trgls[:,1]
        v2 = trgls[:,2]
        d01 = pts3d[v1] - pts3d[v0]
        d02 = pts3d[v2] - pts3d[v0]
        n3d = np.cross(d01, d02)
        ptn3d = np.zeros((len(pts3d), 3), np.float32)
        ptn3d[v0] += n3d
        ptn3d[v1] += n3d
        ptn3d[v2] += n3d
        l2 = np.sqrt(np.sum(ptn3d*ptn3d, axis=1)).reshape(-1,1)
        l2[l2==0] = 1.
        ptn3d /= l2
        return ptn3d
        '''

    def moveAlongNormalsSign(self):
        return 1.

    def moveAlongNormals(self, step):
        ns = self.pointNormals()
        if ns is None:
            print("No normals found")
            return
        # print("man", self.fpoints.shape, ns.shape)
        # fpoints has 4 elements; the 4th is the index
        sgn = self.moveAlongNormalsSign()
        self.fpoints[:, :3] += sgn*step*ns
        self.fragment.gpoints = self.cur_volume_view.volume.transposedIjksToGlobalPositions(self.fpoints, self.fragment.direction)
        self.fragment.notifyModified()
        self.setLocalPoints(True)

    def moveInK(self, step):
        # if len(self.fpoints) > 0:
        #     print("before", self.fragment.gpoints[0], self.fpoints[0])
        self.fpoints[:,2] += step
        self.fragment.gpoints = self.cur_volume_view.volume.transposedIjksToGlobalPositions(self.fpoints, self.fragment.direction)
        # if len(self.fpoints) > 0:
        #     print("after", self.fragment.gpoints[0], self.fpoints[0])
        self.fragment.notifyModified()
        self.setLocalPoints(True)


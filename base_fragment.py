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
    
    def activeAndAligned(self):
        if not self.active:
            return False
        return self.aligned()

    # direction is not used here, but this notifies fragment view
    # to recompute things
    def setVolumeViewDirection(self, direction):
        self.setLocalPoints(False)
    
    def normals(self):
        self.triangulate()
        trgls = self.trgls()
        if trgls is None:
            return
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

    def moveAlongNormals(self, step):
        ns = self.normals()
        if ns is None:
            print("No normals found")
            return
        # print("man", self.fpoints.shape, ns.shape)
        # fpoints has 4 elements; the 4th is the index
        self.fpoints[:, :3] += step*ns
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


import numpy as np
from pathlib import Path
from utils import Utils
from base_fragment import BaseFragment, BaseFragmentView


class TrglFragment(BaseFragment):
    def __init__(self, name):
        super(TrglFragment, self).__init__(name)
        self.gpoints = np.zeros((0,3), dtype=np.float32)
        self.tpoints = np.zeros((0,2), dtype=np.float32)
        self.direction = 0
        '''
        self.name = name
        self.trgls = np.zeros((0,3), dtype=np.int32)
        self.create = Utils.timestamp()
        self.modified = Utils.timestamp()
        '''

    # class function
    def loadObjFile(obj_file):
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
        
        for line in fd:
            line = line.strip()
            if line[0] == '#':
                continue
            words = line.split()
            if words[0] == 'v':
                if len(words) != 4:
                    continue
                vrtl.append([float(w) for w in words[1:]])
            elif words[0] == 'vt':
                if len(words) != 3:
                    continue
                tvrtl.append([float(w) for w in words[1:]])
            elif words[0] == 'f':
                if len(words) != 4:
                    continue
                # implicit assumption that v == vt
                trgl.append([int(w.split('/')[0])-1 for w in words[1:]])
        
        
        trgl_frag = TrglFragment(name)
        trgl_frag.gpoints = np.array(vrtl, dtype=np.float32)
        trgl_frag.tpoints = np.array(tvrtl, dtype=np.float32)
        trgl_frag.trgls = np.array(trgl, dtype=np.int32)
        color = Utils.getNextColor()
        trgl_frag.setColor(color, no_notify=True)
        trgl_frag.valid = True
        print(trgl_frag.name, trgl_frag.color.name(), trgl_frag.gpoints.shape, trgl_frag.tpoints.shape, trgl_frag.trgls.shape)

        # TODO for testing (doesn't work; these go with TrglFragmentView!)
        # trgl_frag.active = False
        # trgl_frag.visible = False
        return trgl_frag

    def createView(self, project_view):
        return TrglFragmentView(project_view, self)

    def createCopy(self, name):
        frag = TrglFragment(name)
        frag.setColor(self.color, no_notify=True)
        frag.gpoints = np.copy(self.gpoints)
        frag.valid = True
        return frag

class TrglFragmentView(BaseFragmentView):
    def __init__(self, project_view, trgl_fragment):
        super(TrglFragmentView, self).__init__(project_view, trgl_fragment)
        # self.project_view = project_view
        # self.fragment = trgl_fragment
        # TODO fix:
        self.line = None

    def setLocalPoints(self, recursion_ok=True, always_update_zsurfs=True):
        if self.cur_volume_view is None:
            self.vpoints = np.zeros((0,4), dtype=np.float32)
            self.fpoints = self.vpoints
            return
        self.vpoints = self.cur_volume_view.volume.globalPositionsToTransposedIjks(self.fragment.gpoints, self.cur_volume_view.direction)
        self.fpoints = self.vpoints
        self.fragment.direction = self.cur_volume_view.direction
        npts = self.vpoints.shape[0]
        if npts > 0:
            indices = np.reshape(np.arange(npts), (npts,1))
            # print(self.vpoints.shape, indices.shape)
            self.vpoints = np.concatenate((self.vpoints, indices), axis=1)
        print("trgl_fragment set local points")

    def getZsurfPoints(self, axis, axis_pos):
        return None

    def getPointsOnSlice(self, axis, i):
        # matches = self.vpoints[(self.vpoints[:, axis] == i)]
        matches = self.vpoints[(self.vpoints[:, axis] >= i-.5) & (self.vpoints[:, axis] < i+.5)]
        return matches

    def aligned(self):
        return True

    def triangulate(self):
        return

    def trgls(self):
        return self.fragment.trgls

    def movePoint(self, index, new_vijk):
        new_gijk = self.cur_volume_view.transposedIjkToGlobalPosition(new_vijk)
        # print(self.fragment.gpoints)
        # print(match, new_gijk)
        self.fragment.gpoints[index, :] = new_gijk
        # print(self.fragment.gpoints)
        self.fragment.notifyModified()
        self.setLocalPoints(True, False)
        return True

    def addPoint(self, tijk):
        return

    def deletePointByIndex(self, index):
        return

    def setLiveZsurfUpdate(self, flag):
        return


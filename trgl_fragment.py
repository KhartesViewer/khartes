import numpy as np
from pathlib import Path
from utils import Utils
from base_fragment import BaseFragment, BaseFragmentView

from PyQt5.QtGui import QColor

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
        print(trgl_frag.name, trgl_frag.color.name(), trgl_frag.gpoints.shape, trgl_frag.tpoints.shape, trgl_frag.trgls.shape)

        return [trgl_frag]

    def createView(self, project_view):
        return TrglFragmentView(project_view, self)

    def createCopy(self, name):
        frag = TrglFragment(name)
        frag.setColor(self.color, no_notify=True)
        frag.gpoints = np.copy(self.gpoints)
        frag.valid = True
        return frag

    # class function
    def saveList(frags, path, stem):
        for frag in frags:
            print("tsl", frag.name)
            frag.save(path)

    def save(self, path):
        fpath = path / self.name
        obj_path = fpath.with_suffix(".obj")
        of = obj_path.open("w")
        # print("hello", file=of)
        print("# Khartes OBJ File", file=of)
        print("# Vertices: %d"%len(self.gpoints), file=of)
        ns = BaseFragment.normals(self.gpoints, self.trgls)
        for i, pt in enumerate(self.gpoints):
            print("v %f %f %f"%(pt[0], pt[1], pt[2]), file=of)
            if ns is not None:
                n = ns[i]
                print("vn %f %f %f"%(n[0], n[1], n[2]), file=of)
        print("# Color and texture information", file=of)
        print("mtllib %s.mtl"%self.name, file=of)
        print("usemtl default", file=of)
        has_texture = (len(self.tpoints) == len(self.gpoints))
        if has_texture:
            for i, pt in enumerate(self.gpoints):
                print("vt %f %f %f"%(pt[0], pt[1], pt[2]), file=of)
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
        of = mtl_path.open("w")
        print("newmtl default", file=of)
        rgb = self.color.getRgbF()
        print("Ka %f %f %f"%(rgb[0],rgb[1],rgb[2]), file=of)
        print("Kd %f %f %f"%(rgb[0],rgb[1],rgb[2]), file=of)
        print("Ks 0.0 0.0 0.0", file=of)
        print("illum 2", file=of)
        print("d 1.0", file=of)
        print("map_Kd %s.tif"%self.name, file=of)

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

    def getPointsOnSlice(self, axis, i):
        # matches = self.vpoints[(self.vpoints[:, axis] == i)]
        matches = self.vpoints[(self.vpoints[:, axis] >= i-.5) & (self.vpoints[:, axis] < i+.5)]
        return matches

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
        self.setLocalPoints(True, False)
        return True


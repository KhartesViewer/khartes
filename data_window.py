from PyQt5.QtGui import (
        QColor, QCursor, QFont, 
        QGuiApplication, QImage, QPalette, QPixmap,
        )
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtCore import QPoint, Qt
import numpy as np
import numpy.linalg as npla
import cv2

from utils import Utils
from st import ST
# import PIL
# import PIL.Image

# non-intuitively, QLabel is what is used to display pixmaps
class DataWindow(QLabel):

    def __init__(self, window, axis):
        super(DataWindow, self).__init__()
        self.window = window

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("lightgray"))
        self.setPalette(palette)

        self.axis = axis

        self.resetText()

        self.volume_view = None
        self.has_had_volume_view = False
        self.mouseStartPoint = None
        self.isPanning = False
        self.isMovingNode = False
        self.isMovingTiff = False
        self.fv2zpoints = {}
        self.localNearbyNodeIndex = -1
        self.maxNearbyNodeDistance = 10
        self.nearbyNodeDistance = -1
        self.nearby_tiff_corner = -1
        self.bounding_nodes = None
        self.bounding_nodes_fv = None
        # self.nearby_tiff_xy = None
        self.tfStartPoint = None
        self.nnStartPoint = None
        self.ntStartPoint = None
        self.cur_frag_pts_xyijk = None
        self.cur_frag_pts_fv = None
        self.setMouseTracking(True)
        self.zoomMult = 1.
        m = 65535
        self.nodeColor = (m,0,0,m)
        self.mutedNodeColor = ((3*m)//4,0,0,m)
        self.highlightNodeColor = (0,m,m,m)
        self.boundingNodeColor = (m,m//4,m//4,m)
        self.inactiveNodeColor = (m//2,m//4,m//4,m)
        self.triLineColor = (3*m//4,2*m//4,3*m//4,m)
        self.splineLineColor = self.triLineColor
        # self.triLineSize = 1
        # self.splineLineSize = self.triLineSize
        # self.inactiveSplineLineSize = 1
        self.inactiveSplineLineColor = self.triLineColor
        # self.crosshairSize = 2
        # self.defaultCursor = Qt.ArrowCursor
        # self.defaultCursor = Qt.OpenHandCursor
        self.defaultCursor = self.window.openhand
        self.nearNonMovingNodeCursor = self.window.openhand_transparent
        self.nearNonMovingNodeCursors = self.window.openhand_transparents
        self.addingCursor = Qt.CrossCursor
        self.movingNodeCursor = Qt.ArrowCursor
        self.panningCursor = Qt.ClosedHandCursor
        self.upperLeftCursor = Qt.SizeFDiagCursor
        self.upperRightCursor = Qt.SizeBDiagCursor
        self.waitCursor = Qt.WaitCursor
        # c = QCursor(Qt.CrossCursor)
        # px = c.pixmap()
        # print("pixmap", px)

    def getDrawWidth(self, name):
        return self.window.draw_settings[name]["width"]

    '''
    def getDrawOpacity(self, name):
        dsn = self.window.draw_settings[name]
        opacity = sdn["opacity"]
        apply = sdn["apply_opacity"]
        if name == "overlay":
            return opacity
        if not apply:
            return 1.0
        return opacity
    '''

    def getDrawOpacity(self, name):
        return self.window.draw_settings[name]["opacity"]

    def getDrawApplyOpacity(self, name):
        return self.window.draw_settings[name]["apply_opacity"]

    # def getDrawSetting(self, clss, name):
    #     return self.window.draw_settings[clss][name]

    def setVolumeView(self, vv):
        self.volume_view = vv
        if vv is None:
            return
        elif not self.has_had_volume_view:
            self.has_had_volume_view = True
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor("black"))
            self.setPalette(palette)
        # print("axis", axis)
        (self.iIndex, self.jIndex) = vv.volume.ijIndexesInPlaneOfSlice(self.axis)
        self.kIndex = self.axis

    def fragmentViews(self):
        return self.window.project_view.fragments.values()

    def currentFragmentView(self):
        return self.window.project_view.mainActiveVisibleFragmentView()

    def positionOnAxis(self):
        return self.volume_view.ijktf[self.kIndex]

    # slice ij position to tijk
    def ijToTijk(self, ij):
        i,j = ij
        tijk = [0,0,0]
        tijk[self.axis] = self.positionOnAxis()
        tijk[self.iIndex] = i
        tijk[self.jIndex] = j
        # print(ij, i, j, tijk)
        return tuple(tijk)

    # slice ijk position to tijk
    def ijkToTijk(self, ijk):
        i,j,k = ijk
        tijk = [0,0,0]
        tijk[self.axis] = k
        tijk[self.iIndex] = i
        tijk[self.jIndex] = j
        # print(ij, i, j, tijk)
        return tuple(tijk)

    def tijkToIj(self, tijk):
        i = tijk[self.iIndex]
        j = tijk[self.jIndex]
        return (i,j)

    def tijkToLocalIjk(self, tijk):
        i = tijk[self.iIndex]
        j = tijk[self.jIndex]
        k = tijk[self.axis]
        return (i,j,k)

    # window xy position to slice ij position
    def xyToIj(self, xy):
        x, y = xy
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        dx, dy = x-wcx, y-wcy
        tijk = list(self.volume_view.ijktf)
        # print("tf", tijk)
        zoom = self.getZoom()
        i = tijk[self.iIndex] + int(dx/zoom)
        j = tijk[self.jIndex] + int(dy/zoom)
        return (i, j)

    # slice ij position to window xy position
    def ijToXy(self, ij):
        i,j = ij
        zoom = self.getZoom()
        tijk = self.volume_view.ijktf
        ci = tijk[self.iIndex]
        cj = tijk[self.jIndex]
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        x = int(zoom*(i-ci)) + wcx
        y = int(zoom*(j-cj)) + wcy
        return (x,y)

    def tijksToIjs(self, tijks):
        ijs = np.stack((tijks[:,self.iIndex], tijks[:,self.jIndex]), axis=1)
        return ijs

    def ijsToXys(self, ijs):
        zoom = self.getZoom()
        tijk = self.volume_view.ijktf
        ci = tijk[self.iIndex]
        cj = tijk[self.jIndex]
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        cij = np.array((ci,cj))
        wc = np.array((wcx,wcy))
        # print(cij.shape, ijs.shape)
        xy = np.rint(zoom*(ijs-cij)+wc).astype(np.int32)
        return xy

    def getNearbyNodeIjk(self):
        xyijks = self.cur_frag_pts_xyijk
        nearbyNode = self.localNearbyNodeIndex
        if nearbyNode >= 0 and xyijks is not None and xyijks.shape[0] != 0:
            if nearbyNode >= xyijks.shape[0]:
                print("PROBLEM in getNearbyNodeIjk")
                print(xyijks.shape, nearbyNode)
                return None
            tijk = xyijks[nearbyNode, 2:]
            return self.tijkToLocalIjk(tijk)
        else:
            return None

    def setWorkingRegion(self):
        xyijks = self.cur_frag_pts_xyijk
        nearbyNode = self.localNearbyNodeIndex
        if nearbyNode >= 0 and xyijks is not None and xyijks.shape[0] != 0:
            # tijk = xyijks[nearbyNode, 2:5]
            index = int(xyijks[nearbyNode, 5])
            fv = self.cur_frag_pts_fv[nearbyNode]
            fv.setWorkingRegion(index, 60.)

    def setNearbyNodeIjk(self, ijk):
        xyijks = self.cur_frag_pts_xyijk
        nearbyNode = self.localNearbyNodeIndex
        if nearbyNode >= 0 and xyijks is not None and xyijks.shape[0] != 0:
            tijk = xyijks[nearbyNode, 2:5]
            index = int(xyijks[nearbyNode, 5])
            fv = self.cur_frag_pts_fv[nearbyNode]
            new_tijk = list(tijk)
            i,j,k = ijk
            new_tijk[self.iIndex] = i
            new_tijk[self.jIndex] = j
            new_tijk[self.axis] = k
            # True if successful
            # if fv.movePoint(index, new_tijk):
            if self.window.movePoint(fv, index, new_tijk):
                # wpos = e.localPos()
                # wxy = (wpos.x(), wpos.y())
                # nearbyNode = self.findNearbyNode(wxy)
                # if not self.setNearbyNode(nearbyNode):
                #     self.window.drawSlices()
                # Don't try to re-find the nearest node, since user probably
                # wants to continue using the key to move the node even
                # if the node moves out of "nearby" range
                self.window.drawSlices()

    # return True if nearby node changed, False otherwise
    def setNearbyNode(self, nearbyNode):
        pv = self.window.project_view
        old_global_node_index = pv.nearby_node_index
        old_global_node_fv = pv.nearby_node_fv
        new_global_node_index = -1
        new_global_node_fv = None
        xyijks = self.cur_frag_pts_xyijk
        xyijks_valid = (xyijks is not None and xyijks.shape[0] != 0)
        if nearbyNode >= 0 and xyijks_valid:
            new_global_node_index = int(xyijks[nearbyNode, 5])
            new_global_node_fv = self.cur_frag_pts_fv[nearbyNode]
        
        # print("ogni",old_global_node_index,"ngni",new_global_node_index)
        # Note special case when a node is deleted, 
        # MainWindow.deleteNearbyNode sets pv.nearby_node_fv to None
        # and pv.nearby_node_index to -1, before setNearbyNode is called,
        # so the global old vs new tests in this if statement fall through.
        # So need to compare old vs new local node indices as well
        if old_global_node_index != new_global_node_index or old_global_node_fv != new_global_node_fv or self.localNearbyNodeIndex != nearbyNode:
            # print("snn", self.curNearbyNode(), nearbyNode)
            if nearbyNode >= 0 and xyijks_valid:
                pv.nearby_node_fv = new_global_node_fv
                pv.nearby_node_index = new_global_node_index
            else:
                pv.nearby_node_fv = None
                pv.nearby_node_index = -1
            self.localNearbyNodeIndex = nearbyNode
            self.window.drawSlices()
            return True
        else:
            return False

    def inAddNodeMode(self):
        # modifiers = QApplication.keyboardModifiers()
        modifiers = QApplication.queryKeyboardModifiers()
        shift_pressed = (modifiers == Qt.ShiftModifier)
        if self.window.add_node_mode:
            shift_pressed = not shift_pressed
        # print("shift_pressed", shift_pressed)
        return shift_pressed

    def findBoundingNodes(self, xy):
        if self.axis == 2:
            # print("a")
            return None
        xyijks = self.cur_frag_pts_xyijk
        if xyijks is None:
            # print("b")
            return None
        if xyijks.shape[0] == 0:
            # print("c")
            return None
        xys = xyijks[:,0:2]
        curfv = self.currentFragmentView()

        d = 3
        ijl = self.xyToIj((xy[0]-d, xy[1]-d))
        ijg = self.xyToIj((xy[0]+d, xy[1]+d))
        line_found = False
        for fv, zpts in self.fv2zpoints.items():
            if len(zpts) == 0:
                continue
            if fv != curfv:
                continue
            matches = ((zpts >= ijl).all(axis=1) & (zpts <= ijg).all(axis=1)).nonzero()[0]
            if len(matches) > 0:
                line_found = True
                break

        if not line_found:
            # print("d")
            return None

        fvs = self.cur_frag_pts_fv
        flags = np.zeros((len(fvs)), dtype=np.bool_)
        for i in range(len(fvs)):
            flags[i] = (fvs[i] == curfv)

        ds = xys[:,0] - xy[0]
        big = 1000000
        dslt = ds.copy()
        dslt[dslt>0] = -big
        dslt[~flags] = -big
        dsgt = ds.copy()
        dsgt[dsgt<=0] = big
        dsgt[~flags] = big

        ilt = np.argmax(dslt)
        igt = np.argmin(dsgt)
        if ilt < 0 or igt < 0:
            # print("e")
            return None
        vlt = dslt[ilt]
        vgt = dsgt[igt]
        if vlt == -big or vgt == big:
            # print("f")
            return None
        xlt = xys[ilt,0]
        xgt = xys[igt,0]
        if xlt < 0 or xgt >= self.width():
            # print("g")
            return None
        filt = int(xyijks[ilt, 5])
        figt = int(xyijks[igt, 5])
        return (filt, figt)

    def findNearbyNode(self, xy):
        # if self.inAddNodeMode():
        #     return -1
        xyijks = self.cur_frag_pts_xyijk
        if xyijks is None:
            return -1
        if xyijks.shape[0] == 0:
            return -1
        xys = xyijks[:,0:2]
        # print(xys.dtype)
        # print("xy, xys, len", xy, xys, len(xys))
        # print("xys minus", xys-np.array(xy))
        ds = npla.norm(xys-np.array(xy), axis=1)
        # print(ds)
        imin = np.argmin(ds)
        vmin = ds[imin]
        self.nearbyNodeDistance = vmin
        if vmin > self.maxNearbyNodeDistance:
            self.nearbyNodeDistance = -1
            return -1

        # print("fnn", imin, index, xyijks[imin])
        # print("fnn", imin, index)

        # will be stored in self.localNearbyNodeIndex
        return imin

    def getZoom(self):
        return self.volume_view.zoom * self.zoomMult

    def allowMouseToDragNode(self):
        return True

    def mousePressEvent(self, e):
        if self.volume_view is None:
            return
        # print("press", e.button())
        if e.button() | Qt.LeftButton:
            modifiers = QApplication.keyboardModifiers()
            wpos = e.localPos()
            wxy = (wpos.x(), wpos.y())

            if self.inAddNodeMode():
                # print('Shift+Click')
                ij = self.xyToIj(wxy)
                tijk = self.ijToTijk(ij)
                # print("adding point at",tijk)
                if self.currentFragmentView() is not None:
                    self.setWaitCursor()
                    self.window.addPointToCurrentFragment(tijk)
                    nearbyNode = self.findNearbyNode(wxy)
                    if not self.setNearbyNode(nearbyNode):
                        self.window.drawSlices()
                
            else:
                # print("left mouse button down")
                self.mouseStartPoint = e.localPos()
                nearbyNode = self.findNearbyNode(wxy)
                tiffCorner = self.findNearbyTiffCorner(wxy)
                '''
                if nearbyNode < 0 or not self.allowMouseToDragNode():
                    self.tfStartPoint = self.volume_view.ijktf
                    self.isPanning = True
                    self.isMovingNode = False
                else:
                    self.nnStartPoint = self.getNearbyNodeIjk()
                    self.isPanning = False
                    self.isMovingNode = True
                '''
                self.tfStartPoint = self.volume_view.ijktf
                self.isPanning = True
                self.isMovingNode = False
                self.isMovingTiff = False
                if self.allowMouseToDragNode():
                    if tiffCorner >= 0:
                        self.ntStartPoint = self.getNearbyTiffIj()
                        self.isPanning = False
                        self.isMovingTiff = True
                    elif nearbyNode >= 0:
                        self.nnStartPoint = self.getNearbyNodeIjk()
                        self.isPanning = False
                        self.isMovingNode = True
        self.checkCursor()

    def drawNodeAtXy(self, outrgbx, xy, color, size):
        cv2.circle(outrgbx, xy, size, color, -1)

    def mouseReleaseEvent(self, e):
        if self.volume_view is None:
            return
        # print("release", e.button())
        if e.button() | Qt.LeftButton:
            self.mouseStartPoint = QPoint()
            self.tfStartPoint = None
            self.nnStartPoint = None
            self.isPanning = False
            self.isMovingNode = False
            self.isMovingTiff = False
            wpos = e.localPos()
            wxy = (wpos.x(), wpos.y())
            # nearbyNode = self.findNearbyNode(wxy)
            # self.setNearbyNode(nearbyNode)
            self.setNearbyTiffAndNode(wxy)
        self.checkCursor()

    def leaveEvent(self, e):
        if self.volume_view is None:
            return
        self.setNearbyNode(-1)
        self.setBoundingNodes(None)
        self.window.setStatusText("")
        self.window.setCursorPosition(None, None)
        self.checkCursor()

    def checkCursor(self):
        # if leaving:
        #     self.unsetCursor()
        #     return
        new_cursor = self.defaultCursor
        if self.isPanning:
            new_cursor = self.panningCursor
        elif self.inAddNodeMode():
            new_cursor = self.addingCursor
        # elif self.allowMouseToDragNode() and (self.isMovingNode or self.localNearbyNodeIndex >= 0):
        elif self.allowMouseToDragNode() and self.localNearbyNodeIndex >= 0:
            new_cursor = self.movingNodeCursor
        elif self.allowMouseToDragNode() and self.nearby_tiff_corner >= 0:
            c = self.nearby_tiff_corner
            if c == 0 or c == 3:
                new_cursor = self.upperLeftCursor
            else:
                new_cursor = self.upperRightCursor
        elif not self.allowMouseToDragNode() and self.localNearbyNodeIndex >= 0:
            cursors = self.nearNonMovingNodeCursors
            cursor = self.nearNonMovingNodeCursor
            if len(cursors) == 0:
                new_cursor = cursor
            else:
                pdist = self.nearbyNodeDistance/self.maxNearbyNodeDistance
                rdist = min(2.-2*pdist, 1)
                index = round((len(cursors)-1)*rdist)
                new_cursor = cursors[index]

        if new_cursor != self.cursor():
            self.setCursor(new_cursor)

    def setStatusTextFromMousePosition(self):
        pt = self.mapFromGlobal(QCursor.pos())
        mxy = (pt.x(), pt.y())
        ij = self.xyToIj(mxy)
        tijk = self.ijToTijk(ij)
        self.setStatusText(tijk)


    def setStatusText(self, ijk):
        if self.volume_view is None:
            return
        gijk = self.volume_view.transposedIjkToGlobalPosition(ijk)
        # gi,gj,gk = gijk
        vol = self.volume_view.volume

        labels = ["X", "Y", "Img"]
        axes = (2,0,1)
        if vol.from_vc_render:
            labels = ["X", "Img", "Y"]
            axes = (1,0,2)
        ranges = vol.getGlobalRanges()
        stxt = ""
        for i in axes:
            g = gijk[i]
            dtxt = "%d"%g
            mn = ranges[i][0]
            mx = ranges[i][1]
            if g < mn or g > mx:
                # dtxt = " --"
                dtxt = "("+dtxt+")"
            stxt += "%s %s   "%(labels[i], dtxt)

        ij = self.tijkToIj(ijk)
        xy = self.ijToXy(ij)
        d = 3
        ijl = self.xyToIj((xy[0]-d, xy[1]-d))
        ijg = self.xyToIj((xy[0]+d, xy[1]+d))
        # ijl = (ij[0]-d, ij[1]-d)
        # ijg = (ij[0]+d, ij[1]+d)
        line_found = False
        for fv, zpts in self.fv2zpoints.items():
            if len(zpts) == 0:
                continue
            # matches = (zpts == ij).all(axis=1).nonzero()[0]
            matches = ((zpts >= ijl).all(axis=1) & (zpts <= ijg).all(axis=1)).nonzero()[0]
            if len(matches) > 0:
                if not line_found:
                    stxt += "|  "
                stxt += " %s"%fv.fragment.name
                line_found = True
        if line_found:
            stxt += "   "


        nearbyNode = self.localNearbyNodeIndex
        # check nearbyNode < len to try and prevent intermittent crash
        # that sometimes occurs when creating active region and
        # then moving mouse quickly to another data slice
        if nearbyNode >= 0 and nearbyNode < len(self.cur_frag_pts_fv):
            fvs = self.cur_frag_pts_fv
            # print("axis",self.axis,"nn", nearbyNode)
            fv = fvs[nearbyNode]
            ijks = self.cur_frag_pts_xyijk[:, 2:5]
            index = int(self.cur_frag_pts_xyijk[nearbyNode, 5])
            stxt += "|  node: #%d %s " % (index, fv.fragment.name)
            tijk = ijks[nearbyNode]
            gijk = self.volume_view.transposedIjkToGlobalPosition(tijk)
            for i,a in enumerate(axes):
                g = gijk[a]
                dtxt = "%g"%round(g,2)
                mn = ranges[a][0]
                mx = ranges[a][1]
                if g < mn or g > mx:
                    # dtxt = " --"
                    dtxt = "("+dtxt+")"
                if i > 0:
                    stxt += " "
                stxt += "%s"%(dtxt)
            stxt += "   "


        self.window.setStatusText(stxt)

    def mouseMoveEvent(self, e):
        # print("move", e.localPos())
        if self.volume_view is None:
            return
        mxy = (e.localPos().x(), e.localPos().y())
        self.setStatusTextFromMousePosition()
        if self.isPanning:
            delta = e.localPos()-self.mouseStartPoint
            dx,dy = delta.x(), delta.y()
            # print("delta", dx, dy)
            tf = list(self.tfStartPoint)
            zoom = self.getZoom()
            tf[self.iIndex] -= int(dx/zoom)
            tf[self.jIndex] -= int(dy/zoom)
            self.volume_view.setIjkTf(tf)
            self.window.drawSlices()
        elif self.isMovingNode:
            # print("moving node")
            if self.nnStartPoint is None:
                print("nnStartPoint is None while moving node!")
                return
            delta = e.localPos()-self.mouseStartPoint
            dx,dy = delta.x(), delta.y()
            zoom = self.getZoom()
            di = int(dx/zoom)
            dj = int(dy/zoom)
            nij = list(self.nnStartPoint)
            nij[0] += di
            nij[1] += dj
            self.setNearbyNodeIjk(nij)
            self.setWaitCursor()
            self.window.drawSlices()
        elif self.isMovingTiff:
            if self.ntStartPoint is None:
                print("ntStartPoint is None while moving node!")
                return
            delta = e.localPos()-self.mouseStartPoint
            dx,dy = delta.x(), delta.y()
            zoom = self.getZoom()
            di = int(dx/zoom)
            dj = int(dy/zoom)
            nij = list(self.ntStartPoint)
            nij[0] += di
            nij[1] += dj
            self.setNearbyTiffIjk(nij)
            # self.window.drawSlices()
        else:
            mxy = (e.localPos().x(), e.localPos().y())
            self.setNearbyTiffAndNode(mxy)
            '''
            nearbyTiffCorner = self.findNearbyTiffCorner(mxy)
            self.setNearbyTiff(nearbyTiffCorner)
            nearbyNode = -1
            if nearbyTiffCorner < 0:
                nearbyNode = self.findNearbyNode(mxy)
            # print("mxy", mxy, nearbyNode)
            self.setNearbyNode(nearbyNode)
            '''
        ij = self.xyToIj(mxy)
        tijk = self.ijToTijk(ij)
        self.window.setCursorPosition(self, tijk)
        self.checkCursor()

    def setBoundingNodes(self, bns):
        prev = self.bounding_nodes
        if prev == bns:
            return
        self.bounding_nodes = bns
        curfv = self.currentFragmentView()
        if bns is None:
            curfv = None
        self.bounding_nodes_fv = curfv
        # if bns is not None:
            # print("bns", bns)
        self.drawSlice()

    def setNearbyTiff(self, nearby_tiff_corner):
        prev_corner = self.nearby_tiff_corner
        if prev_corner == nearby_tiff_corner:
            return
        self.nearby_tiff_corner = nearby_tiff_corner
        self.drawSlice()

    def getNearbyTiffIj(self):
        xyijks = self.cur_frag_pts_xyijk
        corner = self.nearby_tiff_corner
        if corner < 0:
            return None
        tiff_corners = self.window.tiff_loader.corners()
        cur_vol_view = self.window.project_view.cur_volume_view
        # cur_vol = cur_vol_view.cur_volume
        tijks = cur_vol_view.globalPositionsToTransposedIjks(tiff_corners)
        minij = self.tijkToIj(tijks[0])
        maxij = self.tijkToIj(tijks[1])
        mmijs = (minij, maxij)
        cx = corner%2
        cy = corner//2
        corner_ij = (mmijs[cx][0], mmijs[cy][1])
        # print("corner_ij", corner_ij, self.iIndex, self.jIndex)
        return corner_ij

    def setNearbyTiffIjk(self, nij):
        tijk = [0,0,0]
        tijk[self.axis] = 0
        tijk[self.iIndex] = nij[0]
        tijk[self.jIndex] = nij[1]
        vv = self.volume_view
        gijk = vv.transposedIjkToGlobalPosition(tijk)
        # print("gijk", gijk)
        iaxis = vv.globalAxisFromTransposedAxis(self.iIndex)
        jaxis = vv.globalAxisFromTransposedAxis(self.jIndex)
        corner = self.nearby_tiff_corner
        self.window.tiff_loader.setCornerValues(gijk, iaxis, jaxis, corner)

    def findNearbyTiffCorner(self, xy):
        # if self.inAddNodeMode():
        #     return -1
        tiff_corners = self.window.tiff_loader.corners()
        if tiff_corners is None:
            # print("no tiff corners")
            return -1
        minxy, maxxy, intersects_slice = self.cornersToXY(tiff_corners)
        if not intersects_slice:
            # print("no tiff intersect")
            return -1

        xys = np.array((
            (minxy[0],minxy[1]),
            (maxxy[0],minxy[1]),
            (minxy[0],maxxy[1]),
            (maxxy[0],maxxy[1]),
            ), dtype=np.float32)

        ds = npla.norm(xys-np.array(xy), axis=1)
        # print(ds)
        imin = np.argmin(ds)
        vmin = ds[imin]
        if vmin > self.maxNearbyNodeDistance:
            # print("tiff corner too far", vmin)
            self.nearby_tiff_corner = -1
            return -1

        self.nearby_tiff_corner = imin
        # self.nearby_tiff_xy = xys[imin].tolist()
        # print("nearby tiff", self.nearby_tiff_corner, self.nearby_tiff_xy)
        return imin

    def setNearbyTiffAndNode(self, xy):
        nearbyTiffCorner = self.findNearbyTiffCorner(xy)
        self.setNearbyTiff(nearbyTiffCorner)
        nearbyNode = -1
        if nearbyTiffCorner < 0:
            nearbyNode = self.findNearbyNode(xy)
        self.setNearbyNode(nearbyNode)
        bn = self.findBoundingNodes(xy)
        self.setBoundingNodes(bn)

    def wheelEvent(self, e):
        if self.volume_view is None:
            return
        # print("wheel", e.angleDelta(), e.pixelDelta())
        # print("wheel", e.angleDelta().y(), e.pixelDelta())
        self.setStatusTextFromMousePosition()
        d = e.angleDelta().y()
        z = self.volume_view.zoom
        z *= 1.001**d
        # print(d, z)
        self.volume_view.setZoom(z)
        mxy = (e.position().x(), e.position().y())
        self.setNearbyTiffAndNode(mxy)
        '''
        nearbyTiffCorner = self.findNearbyTiffCorner(mxy)
        self.setNearbyTiff(nearbyTiffCorner)
        nearbyNode = -1
        if nearbyTiffCorner < 0:
            nearbyNode = self.findNearbyNode(mxy)
        self.setNearbyNode(nearbyNode)
        '''
        self.window.drawSlices()
        # print("wheel", e.position())
        self.checkCursor()

    # SurfaceWindow subclass overrides this
    # Don't allow it in ordinary slices, because once node moves
    # out of the plane, the localNearbyNodeIndex is no longer valid
    def nodeMovementAllowedInK(self):
        return False

    def setWaitCursor(self):
        self.setCursor(self.waitCursor)
        # self.window.app.processEvents()

    # Note that this is called from MainWindow whenever MainWindow
    # catches a keyPressEvent; since the DataWindow widgets never
    # have focus, they never receive keyPressEvents directly
    def keyPressEvent(self, e):
        if self.volume_view is None:
            return
        key = e.key()
        # print(self.axis, key)
        sgn = 1
        opts = {
            Qt.Key_Left: (1*sgn,0,0),
            Qt.Key_A:    (1*sgn,0,0),
            Qt.Key_Right: (-1*sgn,0,0),
            Qt.Key_D:     (-1*sgn,0,0),
            Qt.Key_Up: (0,1*sgn,0),
            Qt.Key_W:  (0,1*sgn,0),
            Qt.Key_Down: (0,-1*sgn,0),
            Qt.Key_S:    (0,-1*sgn,0),
            Qt.Key_PageUp: (0,0,1*sgn),
            Qt.Key_E:      (0,0,1*sgn),
            Qt.Key_PageDown: (0,0,-1*sgn),
            Qt.Key_C:        (0,0,-1*sgn),
        }
        if key in opts:
            d = opts[key]
            # if self.inAddNodeMode() or (self.localNearbyNodeIndex < 0 and self.nearby_tiff_corner < 0):
            if self.localNearbyNodeIndex < 0 and self.nearby_tiff_corner < 0:
                # pan
                tfijk = list(self.volume_view.ijktf)
                # print(d)
                tfijk[self.iIndex] += d[0]
                tfijk[self.jIndex] += d[1]
                tfijk[self.axis] += d[2]
                self.volume_view.setIjkTf(tfijk)
                self.window.drawSlices()
            elif self.nearby_tiff_corner >= 0:
                # move nearby tiff corner
                # nij = list(self.ntStartPoint)
                nij = list(self.getNearbyTiffIj())
                nij = [round(x) for x in nij]
                if d[2] != 0:
                    return
                nij[0] -= d[0]
                nij[1] -= d[1]
                self.setNearbyTiffIjk(nij)
                # self.window.drawSlices()
            elif self.localNearbyNodeIndex >= 0:
                # move nearby node
                nij = list(self.getNearbyNodeIjk())
                nij = [round(x) for x in nij]
                d = opts[key]
                if d[2] != 0 and not self.nodeMovementAllowedInK():
                    return
                nij[0] -= d[0]
                nij[1] -= d[1]
                nij[2] -= d[2]
                self.setWaitCursor()
                self.setNearbyNodeIjk(nij)
        elif not self.isMovingNode and (key == Qt.Key_Backspace or key == Qt.Key_Delete):
            # print("backspace/delete")
            # ijk = self.getNearbyNodeIjk()
            # if ijk is None:
            #     return
            # print("ijk", ijk)
            # tijk = self.ijToTijk(ijk[0:2])
            # print("tijk", tijk)
            self.setWaitCursor()
            self.window.deleteNearbyNode()
            # this repopulates local node list
            self.drawSlice()
            pt = self.mapFromGlobal(QCursor.pos())
            mxy = (pt.x(), pt.y())
            # nearbyNode = self.findNearbyNode(mxy)
            # print("del nearby node", nearbyNode)
            # self.setNearbyNode(nearbyNode)
            self.setNearbyTiffAndNode(mxy)
            # print("del localNearbyNodeIndex", self.localNearbyNodeIndex)
            self.window.drawSlices()
        elif not self.isMovingNode and key == Qt.Key_X:
            # print("key X")
            ijk = self.getNearbyNodeIjk()
            if ijk is None:
                pt = self.mapFromGlobal(QCursor.pos())
                mxy = (pt.x(), pt.y())
                ij = self.xyToIj(mxy)
                tijk = self.ijToTijk(ij)
            else:
                # print("ijk", ijk)
                # tijk = self.ijToTijk(ijk[0:2])
                tijk = self.ijkToTijk(ijk)
            # print("tijk", tijk)
            self.volume_view.setIjkTf(tijk)
            self.window.drawSlices()
            # move cursor to cross hairs
            ij = self.tijkToIj(tijk)
            xy = self.ijToXy(ij)
            gxy = self.mapToGlobal(QPoint(*xy))
            QCursor.setPos(gxy)
        elif self.inAddNodeMode() and not self.isMovingNode and self.axis in (0,1) and key in (Qt.Key_BracketLeft, Qt.Key_BracketRight, Qt.Key_BraceLeft, Qt.Key_BraceRight):
            self.setWaitCursor()
            ijk = self.getNearbyNodeIjk()
            if ijk is None:
                pt = self.mapFromGlobal(QCursor.pos())
                mxy = (pt.x(), pt.y())
                ij = self.xyToIj(mxy)
            else:
                ij = ijk[:2]
            sign = 1
            if key in (Qt.Key_BracketLeft, Qt.Key_BraceLeft):
                sign = -1
            # self.window.drawSlices()
            self.autoExtrapolate(sign, ij)
        elif not self.isMovingNode and self.axis in (0,1) and key == Qt.Key_I:
            self.setWaitCursor()
            self.autoInterpolate()
        elif key == Qt.Key_T:
            pt = self.mapFromGlobal(QCursor.pos())
            mxy = (pt.x(), pt.y())
            ij = self.xyToIj(mxy)
            tijk = self.ijToTijk(ij)
            self.window.setCursorPosition(self, tijk)
        elif key == Qt.Key_V:
            pt = self.mapFromGlobal(QCursor.pos())
            mxy = (pt.x(), pt.y())
            self.setNearbyTiffAndNode(mxy)
        elif key == Qt.Key_R:
            self.setWaitCursor()
            self.setWorkingRegion()
            pt = self.mapFromGlobal(QCursor.pos())
            mxy = (pt.x(), pt.y())
            self.setNearbyTiffAndNode(mxy)
            self.window.drawSlices()
        elif key == Qt.Key_Shift:
            pt = self.mapFromGlobal(QCursor.pos())
            mxy = (pt.x(), pt.y())
            self.setNearbyTiffAndNode(mxy)
            self.tfStartPoint = None
            self.nnStartPoint = None
            self.mouseStartPoint = None
            self.isPanning = False
            self.isMovingNode = False
            self.isMovingTiff = False
        self.setStatusTextFromMousePosition()
        self.checkCursor()

    # Note that this is called from MainWindow whenever MainWindow
    # catches a keyReleaseEvent; since the DataWindow widgets never
    # have focus, they never receive keyReleaseEvents directly
    def keyReleaseEvent(self, e):
        if self.volume_view is None:
            return
        key = e.key()
        if key == Qt.Key_Shift:
            # print("shift released")
            pt = self.mapFromGlobal(QCursor.pos())
            mxy = (pt.x(), pt.y())
            self.setNearbyTiffAndNode(mxy)
            self.tfStartPoint = None
            self.nnStartPoint = None
            self.mouseStartPoint = None
            self.isPanning = False
            self.isMovingNode = False
            self.isMovingTiff = False
            self.checkCursor()

    '''
    Moved to utils.py
    # adapted from https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles/25068722#25068722
    # The C++ version of OpenCV provides operations, including intersection,
    # on rectangles, but the Python version doesn't.
    def rectIntersection(self, ra, rb):
        (ax1, ay1), (ax2, ay2) = ra
        (bx1, by1), (bx2, by2) = rb
        # print(ra, rb)
        x1 = max(min(ax1, ax2), min(bx1, bx2))
        y1 = max(min(ay1, ay2), min(by1, by2))
        x2 = min(max(ax1, ax2), max(bx1, bx2))
        y2 = min(max(ay1, ay2), max(by1, by2))
        if (x1<x2) and (y1<y2):
            r = ((x1, y1), (x2, y2))
            # print(r)
            return r
    '''

    def axisColor(self, axis):
        color = [0]*4
        # color[3] = int(65535*self.getDrawOpacity("axes"))
        color[axis] = 65535
        if axis == 1:
            # make green a less bright
            color[axis] *= 2/3
        if axis == 2:
            # make blue a bit brighter
            # f = 16384
            f = 24000
            color[(axis+1)%3] = f
            color[(axis+2)%3] = f
        return color

    def sliceGlobalLabel(self):
        gaxis = self.volume_view.globalAxisFromTransposedAxis(self.axis)
        labels = ["X", "Y", "IMG"]
        if self.volume_view.volume.from_vc_render:
            labels = ["X", "IMG", "Y"]
        return labels[gaxis]

    def sliceGlobalPosition(self):
        gxyz = self.volume_view.transposedIjkToGlobalPosition(self.volume_view.ijktf)
        gaxis = self.volume_view.globalAxisFromTransposedAxis(self.axis)
        return gxyz[gaxis]
    
    def autoInterpolate(self):
        if self.bounding_nodes is None:
            return
        volume = self.volume_view
        if volume is None :
            return
        curfv = self.currentFragmentView()
        if self.bounding_nodes_fv != curfv:
            return
        bns = self.bounding_nodes
        bia = bns[0]
        bib = bns[1]
        if bia >= len(curfv.vpoints) or bib >= len(curfv.vpoints):
            print("bia or bib out of range", bia, bib, len(curfv.vpoints))
            return
        tijka = curfv.vpoints[bia]
        tijkb = curfv.vpoints[bib]
        ija = self.tijkToIj(tijka)
        ijb = self.tijkToIj(tijkb)
        if ija[0] > ijb[0]:
            ija,ijb = ijb,ija
        ja = ija[1]
        jb = ijb[1]
        imin = ija[0]
        imax = ijb[0]
        jmin = min(ja,jb)
        jmax = max(ja,jb)

        ww = self.size().width()
        wh = self.size().height()
        ij0 = self.xyToIj((0,0))
        ij1 = self.xyToIj((ww,wh))
        ijo0 = ij0
        ijo1 = ij1
        z = self.getZoom()
        slc = volume.getSlice(self.axis, volume.ijktf)
        sw = slc.shape[1]
        sh = slc.shape[0]
        margin = 32
        ij0m = (ij0[0]-margin,ij0[1]-margin)
        ij1m = (ij1[0]+margin,ij1[1]+margin)
        ri = Utils.rectIntersection((ij0m,ij1m), ((0,0),(sw,sh)))
        if ri is None:
            return
        ij0 = ri[0]
        ij1 = ri[1]
        ri = Utils.rectIntersection((ij0,ij1), ((imin-500,jmin-500),(imax+500,jmax+500)))
        if ri is None:
            return
        ij0 = ri[0]
        ij1 = ri[1]
        print("autointerpolate", ija, ijb, ij0, ij1)
        if imin < ij0[0] or imax >= ij1[0]:
            return
        if jmin < ij0[1] or jmax >= ij1[1]:
            return
        st = ST((slc[int(ij0[1]):int(ij1[1]),int(ij0[0]):int(ij1[0])]).astype(np.float64)/65535.)
        print ("st created", st.image.shape)
        st.computeEigens()
        '''
        path = self.window.project_view.project.path
        st.saveImage(path / "st_debug.tif")
        st.saveEigens(path / "st_debug.nrrd")
        '''
        # print ("eigens computed")
        # dij = (ij[0]-ij0[0], ij[1]-ij0[1])
        dija = (ija[0]-ij0[0], ija[1]-ij0[1])
        dijb = (ijb[0]-ij0[0], ijb[1]-ij0[1])
        # min distance between computed auto-pick points
        # note this is in units of data-volume voxel size,
        # which differ from that of the original data,
        # due to subsampling
        min_delta = 5
        ijk = self.ijToTijk(ij0)
        gxyz = self.volume_view.transposedIjkToGlobalPosition(ijk)
        gaxis = self.volume_view.globalAxisFromTransposedAxis(self.iIndex)
        gstep = self.volume_view.volume.gijk_steps[gaxis]
        # attempt to align the computed points so that they
        # lie at global coordinates (along the appropriate axis)
        # that are mutiples of min_delta.  Seems to work
        # correctly in sub-sampled data volumes as well.
        min_delta_shift = (gxyz[gaxis]/gstep) % min_delta
        # y = st.call_ivp(dij, sign, 5.)
        # print("end points", dija, dijb)
        y = st.interp2dWH(dija, dijb)
        if y is None:
            print("no y values")
            return
        # pts = st.sparse_result(dij, min_delta_shift, min_delta, sign, 5.)
        pts = st.sparse_result(y, min_delta_shift, min_delta)
        if pts is None:
            print("no points")
            return
        # print (len(pts),"points returned")
        pts[:,0] += ij0[0]
        pts[:,1] += ij0[1]
        # print("xs",pts[:,0])
        zsurf_update = self.window.live_zsurf_update
        self.window.setLiveZsurfUpdate(False)
        for pt in pts:
            # pt = (dpt[0]+ij0[0], dpt[1]+ij0[1])
            if pt[0] < ijo0[0] or pt[0] >= ijo1[0] or pt[1] < ijo0[1] or pt[1] >= ijo1[1]:
                break
            tijk = self.ijToTijk(pt)
            # print("adding point at",tijk)
            self.window.addPointToCurrentFragment(tijk)
        self.window.setLiveZsurfUpdate(zsurf_update)
        mpt = self.mapFromGlobal(QCursor.pos())
        mxy = (mpt.x(), mpt.y())
        # nearbyNode = self.findNearbyNode(mxy)
        # self.setNearbyNode(nearbyNode)
        self.setNearbyTiffAndNode(mxy)
        self.window.drawSlices()


    def autoExtrapolate(self, sign, ij):
        volume = self.volume_view
        if volume is None :
            return
        if self.currentFragmentView() is None:
            return
        ww = self.size().width()
        wh = self.size().height()
        ij0 = self.xyToIj((0,0))
        ij1 = self.xyToIj((ww,wh))
        ijo0 = ij0
        ijo1 = ij1
        z = self.getZoom()
        slc = volume.getSlice(self.axis, volume.ijktf)
        sw = slc.shape[1]
        sh = slc.shape[0]
        margin = 32
        ij0m = (ij0[0]-margin,ij0[1]-margin)
        ij1m = (ij1[0]+margin,ij1[1]+margin)
        ri = Utils.rectIntersection((ij0m,ij1m), ((0,0),(sw,sh)))
        if ri is None:
            return
        ij0 = ri[0]
        ij1 = ri[1]
        ri = Utils.rectIntersection((ij0,ij1), ((ij[0]-500,ij[1]-500),(ij[0]+500,ij[1]+500)))
        if ri is None:
            return
        ij0 = ri[0]
        ij1 = ri[1]
        # print("autosegment", ij, sign, ij0, ij1)
        if ij[0] < ij0[0] or ij[0] >= ij1[0]:
            return
        if ij[1] < ij0[1] or ij[1] >= ij1[1]:
            return
        st = ST((slc[int(ij0[1]):int(ij1[1]),int(ij0[0]):int(ij1[0])]).astype(np.float64)/65535.)
        # print ("st created", st.image.shape)
        st.computeEigens()
        # print ("eigens computed")
        dij = (ij[0]-ij0[0], ij[1]-ij0[1])
        # min distance between computed auto-pick points
        # note this is in units of data-volume voxel size,
        # which differ from that of the original data,
        # due to subsampling
        min_delta = 5
        ijk = self.ijToTijk(ij0)
        gxyz = self.volume_view.transposedIjkToGlobalPosition(ijk)
        gaxis = self.volume_view.globalAxisFromTransposedAxis(self.iIndex)
        gstep = self.volume_view.volume.gijk_steps[gaxis]
        # attempt to align the computed points so that they
        # lie at global coordinates (along the appropriate axis)
        # that are mutiples of min_delta.  Seems to work
        # correctly in sub-sampled data volumes as well.
        min_delta_shift = (gxyz[gaxis]/gstep) % min_delta
        y = st.call_ivp(dij, sign, 5.)
        if y is None:
            print("no y values")
            return
        # pts = st.sparse_result(dij, min_delta_shift, min_delta, sign, 5.)
        pts = st.sparse_result(y, min_delta_shift, min_delta)
        if pts is None:
            print("no points")
            return
        # print (len(pts),"points returned")
        pts[:,0] += ij0[0]
        pts[:,1] += ij0[1]
        # print("xs",pts[:,0])
        zsurf_update = self.window.live_zsurf_update
        self.window.setLiveZsurfUpdate(False)
        for pt in pts:
            # pt = (dpt[0]+ij0[0], dpt[1]+ij0[1])
            if pt[0] < ijo0[0] or pt[0] >= ijo1[0] or pt[1] < ijo0[1] or pt[1] >= ijo1[1]:
                break
            tijk = self.ijToTijk(pt)
            # print("adding point at",tijk)
            self.window.addPointToCurrentFragment(tijk)
        self.window.setLiveZsurfUpdate(zsurf_update)
        mpt = self.mapFromGlobal(QCursor.pos())
        mxy = (mpt.x(), mpt.y())
        # nearbyNode = self.findNearbyNode(mxy)
        # self.setNearbyNode(nearbyNode)
        self.setNearbyTiffAndNode(mxy)
        self.window.drawSlices()

    def drawScaleBar(self, outrgbx):
        pixPerMm = 1000./self.volume_view.volume.apparentVoxelSize
        zoom = self.getZoom()
        length = zoom*pixPerMm
        unit = "mm"
        value = 100
        length *= value
        cnt = 0
        maxlen = 80
        maxlen = 40
        while length > maxlen:
            mod = cnt%3
            if mod == 0:
                length *= .5
                value *= .5
            elif mod == 1:
                length *= .4
                value *= .4
            else:
                length *= .5
                value *= .5
            cnt += 1

        length = int(length)
        wh = outrgbx.shape[0]
        y0 = wh - 10
        x0 = 10
        color = (65535,65535,65535,65535)
        text = "%g mm" % value
        cv2.line(outrgbx, (x0, y0), (x0+length, y0), color, 1)
        for i in range(0,2):
            xi = int(x0+i*length)
            cv2.line(outrgbx, (xi,y0-2), (xi, y0+2), color, 1)
        cv2.putText(outrgbx, text, (x0+length+10, y0+3), cv2.FONT_HERSHEY_PLAIN, .8, color)

    def innerResetText(self, text, ptsize):
        if ptsize > 0:
            font = self.font()
            font.setPointSize(ptsize)
            self.setFont(font)
        self.setMargin(10)
        # self.setAlignment(Qt.AlignCenter)
        self.setText(text)

    intro_text1 = '''
<p>
<b>Welcome to <code>khartes</code></b> (χάρτης)
<p>
To get started, go to the menu bar and select
<code>File/Open project...</code> 
to open an existing project, 
or select
<code>File/New project...</code> 
to create a new project.  
Note that when you create a new project, you will immediately be
required to specify a name and a storage location.
<p>
After you have created an empty project, 
you can use <code>File/Import TIFF files...</code> to create
data volumes from existing TIFF files. 
Note that the "Import TIFF Files" command creates a khartes
data volume from existing TIFF files that you alread have
on disk; it will not import TIFF files from elsewhere.
<p>
If you have an existing data volume that is in NRRD format,
you can use
<code>File/Import NRRD files...</code> to import it.
<p>
<b>Please be aware of memory limitations</b>.  The import-TIFF
dialog box shows you how large a volume you are about
to create; this entire volume will need to fit into your
computer's physical memory (RAM), so don't make your
volume too large.
<p>
To create a new fragment, go to the control panel in the lower right,
select the <code>Fragments</code> tab, and press the 
<code>Start New Fragment</code> button.
<p>
Once you have completed these 3 steps 
(created a new project, imported a sub-volume, 
and created a new fragment), you are ready to begin segmenting!
'''

    intro_text2 = '''
<p>
Congratulations!  You have succesfully created a new project.
<p>
Now that you have accomplished that, 
you need to go to the menu bar and 
select <code>File/Import TIFF files...</code> or 
<code>File/Import NRRD</code> to import a data volume.

Note that the "Import TIFF Files" command creates a khartes
data volume from existing TIFF files that you already have
on disk; it will not import TIFF files from elsewhere.
<p>
If you have an existing data volume that is in NRRD format,
you can use
<code>File/Import NRRD files...</code> to import it.
<p>
<b>Please be aware of memory limitations</b>.  The import-TIFF
dialog box shows you how large a volume you are about
to create; this entire volume will need to fit into your
computer's physical memory (RAM), so don't make your
volume too large.

<p>
<b>After you import a volume, 
this help message will disappear,</b>
so here are some next steps for you to keep in mind.
<p>
After you view the data volume, 
you may want to change its orientation.  To do this,
go to the control panel in the lower right and select
the <code>Volumes</code> tab.
In the <code>Dir</code> column, select your preferred viewing
orientation.
The orientation will determine what fragment shapes are allowed.
<p>
And remember that the next step after 
that is to press the <code>Start New Fragment</code>
in the <code>Fragments</code> tab to create a new fragment.
'''

    intro_text3 = '''
<p>
<b>Mousing and keying</b>
<p>
To pan within your current slice,
hold down the left mouse button while 
moving the mouse
Also, if your cursor is close to a fragment node, 
so that the node has turned cyan, 
you can use the mouse-plus-left-button combination to drag the node.
<p>
To create a new fragment node, click on the left mouse button while
holding down the shift key.
<p>
Use the mouse wheel to zoom in and out.
<p>
For finer control, you can use the arrow keys, 
as well as w-a-s-d, to navigate within a slice, 
and page-up / page-down (as well as the e and c keys) 
to move between slices.  
Likewise, the arrow keys can be used to move fragment 
nodes if the cursor is close to the node.
<p>
In the fragment-view window (the big window on the 
upper right), 
page-up and page-down (and the e and c keys) 
can be used to move fragment nodes 
into and out of the viewing plane.
'''


    def resetText(self, surface_window=False):
        text = ""
        ptsize = 12
        if surface_window:
            # text = ("Hello\nThis is a test")
            if self.window.project_view is None:
                text = DataWindow.intro_text1+DataWindow.intro_text3
            else:
                text = DataWindow.intro_text2+DataWindow.intro_text3
            ptsize = 10
            # print("inserting text")
            edit = self.window.edit
            font = edit.font()
            font.setPointSize(12)
            edit.setFont(font)

            # edit.setPlainText(text)
            edit.setText(text)
            return
        elif self.axis == 1:
            text = ("χάρτης")
            ptsize = 36
            self.setAlignment(Qt.AlignCenter)
        self.innerResetText(text, ptsize)

    # return xymin, xymax, intersects_slice
    def cornersToXY(self, corners):
        cur_vol_view = self.window.project_view.cur_volume_view
        # cur_vol = cur_vol_view.cur_volume
        tijks = cur_vol_view.globalPositionsToTransposedIjks(corners)
        mink = tijks[0][self.axis]
        maxk = tijks[1][self.axis]
        curk = self.positionOnAxis()
        intersects_slice = (mink <= curk <= maxk)

        minij = self.tijkToIj(tijks[0])
        maxij = self.tijkToIj(tijks[1])
        minxy = self.ijToXy(minij)
        maxxy = self.ijToXy(maxij)
        return minxy, maxxy, intersects_slice

    def drawTrackingCursor(self, canvas):
        cw = self.window.cursor_window
        if cw is None or cw == self:
            return
        cijk = self.window.cursor_tijk
        cij = self.tijkToIj(cijk)
        cxy = self.ijToXy(cij)
        r = 5
        thickness = 2
        # color = self.axisColor(cw.axis)
        white = (65535,65535,65535)
        color = (16000,55000,16000)
        # cv2.circle(canvas, cxy, 2*r+1, white, thickness+2)
        cv2.circle(canvas, cxy, 2*r+1, color, thickness)
        # self.ijToTijk is a virtual function that
        # sets the true k value in the fragment window
        # k = self.positionOnAxis()
        k = self.ijToTijk(cij)[self.axis]
        ck = cijk[self.axis]
        cx = cxy[0]
        cy = cxy[1]
        if k-ck != 0:
            cv2.line(canvas, (cx-r,cy), (cx+r,cy), white, thickness+2)
            cv2.line(canvas, (cx-r,cy), (cx+r,cy), color, thickness)
        if k < ck:
            cv2.line(canvas, (cx,cy-r), (cx,cy+r), white, thickness+2)
            cv2.line(canvas, (cx,cy-r), (cx,cy+r), color, thickness)

    def drawSlice(self):
        timera = Utils.Timer(False)
        volume = self.volume_view
        if volume is None :
            self.clear()
            if not self.has_had_volume_view:
                self.resetText()
            return
        opacity = self.getDrawOpacity("overlay")
        apply_labels_opacity = self.getDrawApplyOpacity("labels")
        apply_axes_opacity = self.getDrawApplyOpacity("axes")
        apply_borders_opacity = self.getDrawApplyOpacity("borders")
        apply_node_opacity = self.getDrawApplyOpacity("node")
        apply_free_node_opacity = self.getDrawApplyOpacity("free_node")
        apply_mesh_opacity = self.getDrawApplyOpacity("mesh")
        apply_line_opacity = self.getDrawApplyOpacity("line")
        self.setMargin(0)
        self.window.setFocus()
        # if self.axis == 1:
        z = self.getZoom()
        slc = volume.getSlice(self.axis, volume.ijktf)
        # timera.time("getslice")
        # slice width, height
        sw = slc.shape[1]
        sh = slc.shape[0]
        # label = self.sliceGlobalLabel()
        # gpos = self.sliceGlobalPosition()
        # print("%s %d %d"%(self.sliceGlobalLabel(), sw, sh))
        # zoomed slice width, height
        zsw = max(int(z*sw), 1)
        zsh = max(int(z*sh), 1)
        # viewing window width
        ww = self.size().width()
        wh = self.size().height()
        # viewing window half width
        whw = ww//2
        whh = wh//2
        # fi, fj = volume.ijInPlaneOfSlice(self.axis, volume.ijktf)
        fi, fj = self.tijkToIj(volume.ijktf)

        out = np.zeros((wh,ww), dtype=np.uint16)

        # Pasting zoomed data slice into viewing-area array, taking
        # panning into account.
        # In OpenCV, unlike PIL, need to calculate the interesection
        # of the two rectangles: 1) the panned and zoomed slice, and 2) the
        # viewing window, before pasting
        ax1 = int(whw-z*fi)
        ay1 = int(whh-z*fj)
        ax2 = ax1+zsw
        ay2 = ay1+zsh
        bx1 = 0
        by1 = 0
        bx2 = ww
        by2 = wh
        ri = Utils.rectIntersection(((ax1,ay1),(ax2,ay2)), ((bx1,by1),(bx2,by2)))
        if ri is not None:
            (x1,y1),(x2,y2) = ri
            # zoomed data slice
            x1s = int((x1-ax1)/z)
            y1s = int((y1-ay1)/z)
            x2s = int((x2-ax1)/z)
            y2s = int((y2-ay1)/z)
            # print(sw,sh,ww,wh)
            # print(x1,y1,x2,y2)
            # print(x1s,y1s,x2s,y2s)
            zslc = cv2.resize(slc[y1s:y2s,x1s:x2s], (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = zslc

            timera.time("resize")
        # timera.time("fit rect")

        # convert 16-bit (uint16) gray scale to 16-bit RGBX (X is like
        # alpha, but always has the value 65535)
        outrgbx = np.stack(((out,)*3), axis=-1)
        original = None
        # if opacity > 0 and opacity < 1:
        if opacity < 1:
            original = outrgbx.copy()
        else:
            apply_labels_opacity = True
            apply_axes_opacity = True
            apply_borders_opacity = True
            apply_node_opacity = True
            apply_free_node_opacity = True
            apply_line_opacity = True
            apply_mesh_opacity = True
        # outrgbx = np.stack(((out,)*4), axis=-1)
        # outrgbx[:,:,3] = 65535
        # draw a colored rectangle outline around the window, then
        # draw a thin black rectangle outline on top of that
        bw = self.getDrawWidth("borders")
        bwh = (bw-1)//2
        
        if bw > 0:
            cv2.rectangle(outrgbx, (bwh,bwh), (ww-bwh-1,wh-bwh-1), self.axisColor(self.axis), bw)
            cv2.rectangle(outrgbx, (0,0), (ww-1,wh-1), (0,0,0,65536), 1)
            if not apply_borders_opacity:
                cv2.rectangle(original, (bwh,bwh), (ww-bwh-1,wh-bwh-1), self.axisColor(self.axis), 5)
                cv2.rectangle(original, (0,0), (ww-1,wh-1), (0,0,0,65536), 1)

        fij = self.tijkToIj(volume.ijktf)
        fx,fy = self.ijToXy(fij)

        # size = self.crosshairSize
        size = self.getDrawWidth("axes")
        if size > 0:
            cv2.line(outrgbx, (fx,0), (fx,wh), self.axisColor(self.iIndex), size)
            cv2.line(outrgbx, (0,fy), (ww,fy), self.axisColor(self.jIndex), size)
            if not apply_axes_opacity:
                cv2.line(original, (fx,0), (fx,wh), self.axisColor(self.iIndex), size)
                cv2.line(original, (0,fy), (ww,fy), self.axisColor(self.jIndex), size)
        timera.time("draw cv2 underlay")

        self.cur_frag_pts_xyijk = None
        self.cur_frag_pts_fv = []
        xypts = []
        pv = self.window.project_view
        self.fv2zpoints = {}
        nearbyNode = (pv.nearby_node_fv, pv.nearby_node_index)
        splineLineSize = self.getDrawWidth("line")
        nodeSize = self.getDrawWidth("node")
        freeNodeSize = self.getDrawWidth("free_node")
        margin = 20
        oi0,oj0 = self.xyToIj((-margin, -margin))
        oi1,oj1 = self.xyToIj((ww+margin, wh+margin))
        wi0 = min(oi0, oi1)
        wi1 = max(oi0, oi1)
        wj0 = min(oj0, oj1)
        wj1 = max(oj0, oj1)

        for frag in self.fragmentViews():
            fragNodeSize = nodeSize
            apply_frag_node_opacity = apply_node_opacity
            if not frag.mesh_visible:
                fragNodeSize = freeNodeSize
                apply_frag_node_opacity = apply_free_node_opacity
            # if not frag.visible and frag != self.currentFragmentView():
            if not frag.visible or opacity == 0.:
                continue
            pts = frag.getZsurfPoints(self.axis, self.positionOnAxis())
            # if pts is not None:
            #     print(self.axis, len(pts))
            timera.time("get zsurf points")
            if pts is not None and splineLineSize > 0:
                # self.fv2zpoints[frag] = np.round(pts).astype(np.int32)
                pts = pts[(pts[:,0] >= wi0) & (pts[:,1] >= wj0) & (pts[:,0] < wi1) & (pts[:,1] < wj1)]
                self.fv2zpoints[frag] = pts
                # print(pts)
                color = frag.fragment.cvcolor
                # if frag == self.currentFragmentView():
                size = (3*splineLineSize)//2
                if frag.active:
                    # color = self.splineLineColor
                    # size = self.splineLineSize
                    if len(pts) == 1:
                        size += 2 
                else:
                    # size = splineLineSize
                    pass
                vrts = self.ijsToXys(pts)
                # vrts = vrts[(vrts[:,0] >= 0) & (vrts[:,1] >= 0) & (vrts[:,0] < ww) & (vrts[:,1] < wh)]
                # print(" ", len(vrts))
                vrts = vrts.reshape(-1,1,1,2).astype(np.int32)
                cv2.polylines(outrgbx, vrts, True, color, size)
                if not apply_line_opacity:
                    cv2.polylines(original, vrts, True, color, size)
                timera.time("draw zsurf points")

            lines, trglist = frag.getLinesOnSlice(self.axis, self.positionOnAxis())
            if lines is not None and splineLineSize > 0:
                trgls = frag.trgls()
                wtrgls = frag.workingTrgls()
                # working = np.full((len(trgls),), False)
                # working[wtrgls] = True
                # working_bools = working[trglist]
                working_bools = wtrgls[trglist]
                working_lines = lines[working_bools]
                non_working_lines = lines[~working_bools]
                # print(len(lines),"lines",len(working_lines),"working lines")
                llen = len(lines)
                wlen = len(working_lines)
                (has_working, has_non_working) = frag.hasWorkingNonWorking()
                # print("hwhnw", has_working, has_non_working)
                all_working = not has_non_working
                no_working = not has_working
                # all_working = (wlen == llen)
                # no_working = (wlen == 0)
                if all_working:
                    normal_lines = working_lines
                    thin_lines = None
                    # print("all working", len(normal_lines))
                elif no_working:
                    normal_lines = non_working_lines
                    thin_lines = None
                    # print("no working", len(normal_lines))
                else:
                    normal_lines = working_lines
                    thin_lines = non_working_lines
                    # print("normal thin", len(normal_lines), len(thin_lines))


                ijkpts = normal_lines.reshape(-1, 3)
                ijpts = self.tijksToIjs(ijkpts)
                xys = self.ijsToXys(ijpts)
                # TODO: figure out how to put a dense-enough set
                # of points in fv2zpoints, for the status bar
                # self.fv2zpoints[frag] = ijpts
                xys = xys.reshape(-1,1,2,2)
                color = frag.fragment.cvcolor
                size = splineLineSize
                cv2.polylines(outrgbx, xys, False, color, size)
                if not apply_line_opacity:
                    cv2.polylines(original, xys, False, color, size)

                size -= 1
                if thin_lines is not None and size > 0:
                    ijkpts = thin_lines.reshape(-1, 3)
                    ijpts = self.tijksToIjs(ijkpts)
                    xys = self.ijsToXys(ijpts)
                    # TODO: figure out how to put a dense-enough set
                    # of points in fv2zpoints, for the status bar
                    # self.fv2zpoints[frag] = ijpts
                    xys = xys.reshape(-1,1,2,2)
                    color = frag.fragment.cvcolor
                    size = splineLineSize-1
                    cv2.polylines(outrgbx, xys, False, color, size)
                    if not apply_line_opacity:
                        cv2.polylines(original, xys, False, color, size)

            pts = frag.getPointsOnSlice(self.axis, self.positionOnAxis())
            # working is an array of bools, one per pt
            working = frag.workingVpoints()
            '''
            working_bools = working[working_list]
            working_pts = pts[working_bools]
            non_working_pts = pts[~working_bools]
            llen = len(pts)
            wlen = len(working_pts)
            all_working = (wlen == llen)
            no_working = (wlen == 0)
            '''
            (has_working, has_non_working) = frag.hasWorkingNonWorking()
            all_working = not has_non_working
            none_working = not has_working
            # all_working = working.all()
            # none_working = (~working).all()
            emphasize = working
            if none_working:
                emphasize[:] = True

            timera.time("get nodes on slice")
            m = 65535
            self.nearbyNode = -1
            i0 = len(xypts)
            for i, pt in enumerate(pts):
                ij = self.tijkToIj(pt)
                xy = self.ijToXy(ij)
                xypts.append((xy[0], xy[1], pt[0], pt[1], pt[2], pt[3]))
                self.cur_frag_pts_fv.append(frag)
                # print("circle at",ij, xy)
                color = self.nodeColor
                size = fragNodeSize
                if not emphasize[int(pt[3])]:
                    # color = self.mutedNodeColor
                    size = fragNodeSize-1
                if not frag.active:
                    color = self.inactiveNodeColor
                if not frag.mesh_visible:
                    color = frag.fragment.cvcolor
                # print(pt, self.volume_view.nearbyNode)
                if self.bounding_nodes_fv == frag and self.bounding_nodes is not None and pt[3] in self.bounding_nodes:
                    color = self.boundingNodeColor
                    size += 1
                if (frag, pt[3]) == nearbyNode:
                    color = self.highlightNodeColor
                    self.nearbyNode = i0+i
                if size > 0:
                    self.drawNodeAtXy(outrgbx, xy, color, size)
                    if not apply_frag_node_opacity:
                        self.drawNodeAtXy(original, xy, color, size)
            timera.time("draw nodes on slice")

            m = 65535
        timera.time("draw zsurf points")
        self.cur_frag_pts_xyijk = np.array(xypts)

        # if self.window.project_view.settings['slices']['vol_boxes_visible']:
        if self.window.getVolBoxesVisible():
            cur_vol_view = self.window.project_view.cur_volume_view
            cur_vol = self.window.project_view.cur_volume
            for vol, vol_view in self.window.project_view.volumes.items():
                if vol == cur_vol:
                    continue
                gs = vol.corners()
                minxy, maxxy, intersects_slice = self.cornersToXY(gs)
                if not intersects_slice:
                    continue
                cv2.rectangle(outrgbx, minxy, maxxy, vol_view.cvcolor, 2)
                if not apply_labels_opacity:
                    cv2.rectangle(original, minxy, maxxy, vol_view.cvcolor, 2)
        tiff_corners = self.window.tiff_loader.corners()
        if tiff_corners is not None:
            # print("tiff corners", tiff_corners)

            minxy, maxxy, intersects_slice = self.cornersToXY(tiff_corners)
            if intersects_slice:
                # tcolor is a string
                tcolor = self.window.tiff_loader.color()
                qcolor = QColor(tcolor)
                rgba = qcolor.getRgbF()
                cvcolor = [int(65535*c) for c in rgba]
                cv2.rectangle(outrgbx, minxy, maxxy, cvcolor, 2)
                if not apply_labels_opacity:
                    cv2.rectangle(original, minxy, maxxy, cvcolor, 2)
        timera.time("draw frag")
        # print(self.cur_frag_pts_xyijk.shape)
        label = self.sliceGlobalLabel()
        gpos = self.sliceGlobalPosition()
        # print("label", self.axis, label, gpos)
        txt = "%s: %d" % (label, gpos)
        org = (10,20)
        size = 1.
        m = 16000
        gray = (m,m,m,65535)
        white = (65535,65535,65535,65535)
        if self.getDrawWidth("labels") > 0:
            cv2.putText(outrgbx, txt, org, cv2.FONT_HERSHEY_PLAIN, size, gray, 3)
            cv2.putText(outrgbx, txt, org, cv2.FONT_HERSHEY_PLAIN, size, white, 1)
            self.drawScaleBar(outrgbx)
            self.drawTrackingCursor(outrgbx)
            if not apply_labels_opacity:
                cv2.putText(original, txt, org, cv2.FONT_HERSHEY_PLAIN, size, gray, 3)
                cv2.putText(original, txt, org, cv2.FONT_HERSHEY_PLAIN, size, white, 1)
                self.drawScaleBar(original)
                self.drawTrackingCursor(original)


        if opacity > 0 and opacity < 1:
            outrgbx = cv2.addWeighted(outrgbx, opacity, original, 1.-opacity, 0)
        elif opacity == 0:
            outrgbx = original
        # outrgbx = np.append(outrgbx, np.full((wh,ww,1), 32000, dtype=np.uint16), axis=2)
        outrgbx = np.append(outrgbx, np.zeros((wh,ww,1), dtype=np.uint16), axis=2)

        bytesperline = 8*outrgbx.shape[1]
        qimg = QImage(outrgbx, outrgbx.shape[1], outrgbx.shape[0], 
                bytesperline, QImage.Format_RGBX64)
        pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(pixmap)
        timera.time("draw to qt")

class SurfaceWindow(DataWindow):

    def __init__(self, window):
        super(SurfaceWindow, self).__init__(window, 2)
        self.zoomMult = 1.5

    # see comments for this function in DataWindow
    def nodeMovementAllowedInK(self):
        return True

    def allowMouseToDragNode(self):
        return False

    # slice ij position to tijk
    def ijToTijk(self, ij):
        if self.axis != 2:
            return super(SurfaceWindow, self).ijToTijk(ij)
        i,j = ij
        tijk = [0,0,0]
        tijk[self.axis] = self.positionOnAxis()
        afvs = self.window.project_view.activeFragmentViews()
        afvs.reverse()
        for fv in afvs:
            zsurf = fv.workingZsurf()
            if zsurf is None:
                continue
            ri = round(i)
            rj = round(j)
            if rj >= 0 and rj < zsurf.shape[0] and ri >= 0 and ri < zsurf.shape[1]:
                z = zsurf[rj,ri]
                if not np.isnan(z):
                    tijk[self.axis] = np.rint(z)
                    break
        tijk[self.iIndex] = i
        tijk[self.jIndex] = j
        # print(ij, i, j, tijk)
        return tuple(tijk)

    def drawSlice(self):
        timera = Utils.Timer(False)
        volume = self.volume_view
        if volume is None:
            self.clear()
            if not self.has_had_volume_view:
                self.resetText(True)
                self.window.edit.show()
            return
        self.window.edit.hide()
        self.setMargin(0)
        curfv = self.currentFragmentView()
        # zoom by twice the usual amount
        z = self.getZoom()
        # viewing window width
        ww = self.size().width()
        wh = self.size().height()
        # viewing window half width
        # print("--------------------")
        # print("FRAGMENT", ww, wh)
        whw = ww//2
        whh = wh//2
        opacity = self.getDrawOpacity("overlay")
        apply_labels_opacity = self.getDrawApplyOpacity("labels")
        apply_axes_opacity = self.getDrawApplyOpacity("axes")
        apply_node_opacity = self.getDrawApplyOpacity("node")
        apply_free_node_opacity = self.getDrawApplyOpacity("free_node")
        apply_mesh_opacity = self.getDrawApplyOpacity("mesh")

        out = np.zeros((wh,ww), dtype=np.uint16)
        overout = None
        overout = np.zeros((wh,ww), dtype=np.uint16)
        overlay = None
        timera.time("zeros")
        # convert 16-bit (uint16) gray scale to 16-bit RGBX (X is like
        # alpha, but always has the value 65535)
        # outrgbx = np.zeros((wh,ww,4), dtype=np.uint16)
        # outrgbx[:,:,3] = 65535
        self.fv2zpoints = {}
        for frag in self.fragmentViews():
            # if not frag.activeAndAligned():
            if not frag.active:
                continue
            if frag.aligned() and frag.workingZsurf() is not None and frag.workingSsurf() is not None:
                slc = frag.workingSsurf()
                sw = slc.shape[1]
                sh = slc.shape[0]
                # zoomed slice width, height
                zsw = max(int(z*sw), 1)
                zsh = max(int(z*sh), 1)
                fi, fj = self.tijkToIj(volume.ijktf)
        
                # zoomed data slice
                timera.time("prep")
                # timera.time("resize")
                # viewing window
        
                # Pasting zoomed data slice into viewing-area array, taking
                # panning into account.
                # In OpenCV, unlike PIL, need to calculate the interesection
                # of the two rectangles: 1) the panned and zoomed slice, and 2) the
                # viewing window, before pasting
                ax1 = int(whw-z*fi)
                ay1 = int(whh-z*fj)
                ax2 = ax1+zsw
                ay2 = ay1+zsh
                bx1 = 0
                by1 = 0
                bx2 = ww
                by2 = wh
                ri = Utils.rectIntersection(((ax1,ay1),(ax2,ay2)), ((bx1,by1),(bx2,by2)))
                if ri is not None:
                    (x1,y1),(x2,y2) = ri
                    x1s = int((x1-ax1)/z)
                    y1s = int((y1-ay1)/z)
                    x2s = int((x2-ax1)/z)
                    y2s = int((y2-ay1)/z)
                    # print(sw,sh,ww,wh)
                    # print(x1,y1,x2,y2)
                    # print(x1s,y1s,x2s,y2s)
                    zslc = cv2.resize(slc[y1s:y2s,x1s:x2s], (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
                    out[y1:y2, x1:x2] = np.where(zslc==0,out[y1:y2, x1:x2], zslc)

                    timera.time("resize")
                ogrid = None
                if hasattr(curfv, 'osurf'):
                    ogrid = curfv.osurf
                if ogrid is not None and ri is not None:
                    overout = np.zeros((wh,ww), dtype=np.float32)
                    overout[:] = np.nan
                    zover = cv2.resize(ogrid, (zsw, zsh), interpolation=cv2.INTER_AREA)
                    (x1,y1,x2,y2) = ri
                    overout[y1:y2, x1:x2] = zover[(y1-ay1):(y2-ay1),(x1-ax1):(x2-ax1)]


        # convert 16-bit (uint16) gray scale to 16-bit RGBX (X is like
        # alpha, but always has the value 65535)
        # outrgbx = np.stack(((out,)*4), axis=-1)
        # outrgbx[:,:,3] = 65535
        outrgbx = np.stack(((out,)*3), axis=-1)
        original = None
        # if opacity > 0 and opacity < 1:
        if opacity < 1:
            original = outrgbx.copy()
        else:
            apply_labels_opacity = True
            apply_axes_opacity = True
            apply_node_opacity = True
            apply_free_node_opacity = True
            apply_mesh_opacity = True
        '''
        # Needs to be rewritten; outdated
        if overout is not None:
            outrgbx[:,:,0:3] //= 4
            outrgbx[:,:,0:3] *= 3
            # gt0 = curfv.gt0
            # lt0 = curfv.lt0
            mn = np.nanmin(overout)
            mx = np.nanmax(overout)
            amax = max(abs(mn),abs(mx))
            if amax > 0:
                overout /= amax
            gt0 = overout >= 0
            lt0 = overout < 0
            ogt0 = (65536*overout[gt0]).astype(np.uint16)
            olt0 = (-65536*overout[lt0]).astype(np.uint16)
            
            ogt0 //= 4
            olt0 //= 4

            outrgbx[gt0,0] += ogt0
            outrgbx[gt0,2] += ogt0
            outrgbx[lt0,1] += olt0
        '''

        timera.time("draw cv2 underlay")
        fij = self.tijkToIj(volume.ijktf)
        fx,fy = self.ijToXy(fij)

        # size = self.crosshairSize
        size = self.getDrawWidth("axes")
        if size > 0:
            cv2.line(outrgbx, (fx,0), (fx,wh), self.axisColor(self.iIndex), size)
            cv2.line(outrgbx, (0,fy), (ww,fy), self.axisColor(self.jIndex), size)
            if not apply_axes_opacity:
                cv2.line(original, (fx,0), (fx,wh), self.axisColor(self.iIndex), size)
                cv2.line(original, (0,fy), (ww,fy), self.axisColor(self.jIndex), size)

        self.cur_frag_pts_xyijk = None
        self.cur_frag_pts_fv = []

        pv = self.window.project_view
        xypts = []

        triLineSize = self.getDrawWidth("mesh")
        nodeSize = self.getDrawWidth("node")
        freeNodeSize = self.getDrawWidth("free_node")
        for frag in self.fragmentViews():
            # if not frag.activeAndAligned():
            if not frag.active:
                continue
            if not frag.visible or opacity == 0.:
                continue
            fragNodeSize = nodeSize
            apply_frag_node_opacity = apply_node_opacity
            if not frag.mesh_visible:
                fragNodeSize = freeNodeSize
                apply_frag_node_opacity = apply_free_node_opacity
            '''
            nni = -1
            if frag == pv.nearby_node_fv and pv.nearby_node_index >= 0 and nodeSize > 0:
                wpts = frag.workingVpoints()
                nni = pv.nearby_node_index
                if len(vpts) < nni or vpts[nni,3] != nni:
                    w = np.where(vpts[:,3] == nni)[0]
                    if len(w) == 0:
                        nni = -1
                    else:
                        nni = w[0]
            '''
            lineColor = frag.fragment.cvcolor
            self.nearbyNode = -1
            timer_active = False
            timer = Utils.Timer(timer_active)
            # if frag.tri is not None:
            # if frag.workingTrgls() is not None:
            if frag.workingTrgls().any():
                # pts = frag.tri.points
                wvflags = frag.workingVpoints()
                # pts = vpts[:, 0:2]
                allpts = frag.vpoints[:, 0:2]
                wpts = allpts[wvflags]
                # trgs = frag.tri.simplices
                wtflags = frag.workingTrgls()
                alltrgs = frag.trgls()
                wtrgs = alltrgs[wtflags]
                # allpts = frag.vpoints[:,:2]
                # vpts = allpts
                vrts = allpts[wtrgs]
                vrts = self.ijsToXys(vrts)
                vrts = vrts.reshape(-1,3,1,2).astype(np.int32)
                timer.time("compute lines")
                # True means closed line
                if triLineSize > 0:
                    cv2.polylines(outrgbx, vrts, True, lineColor, triLineSize)
                    if not apply_mesh_opacity:
                        cv2.polylines(original, vrts, True, lineColor, triLineSize)
                timer.time("draw lines")

                color = self.nodeColor
                # test not needed, by this point all frags are active
                # if not frag.activeAndAligned():
                #     color = self.inactiveNodeColor
                timer.time("compute points")
                vrts = self.ijsToXys(wpts)
                vrts = vrts.reshape(-1,1,1,2).astype(np.int32)
                if fragNodeSize > 0:
                    cv2.polylines(outrgbx, vrts, True, color, 2*fragNodeSize)
                    if not apply_frag_node_opacity:
                        cv2.polylines(original, vrts, True, color, 2*fragNodeSize)
                timer.time("draw points")

                if frag == pv.nearby_node_fv and pv.nearby_node_index >= 0 and fragNodeSize > 0:
                    nni = pv.nearby_node_index
                    if wvflags[nni]:
                        pt = allpts[nni]
                        xy = self.ijToXy(pt)
                        color = self.highlightNodeColor
                        self.drawNodeAtXy(outrgbx, xy, color, fragNodeSize)
                        if not apply_frag_node_opacity:
                            self.drawNodeAtXy(original, xy, color, fragNodeSize)

            elif frag.workingLine() is not None and frag.workingLineAxis() > -1:
                line = frag.workingLine()
                pts = np.zeros((line.shape[0],3), dtype=np.int32)
                # print(line.shape, pts.shape)
                axis = frag.workingLineAxis()
                pts[:,1-axis] = line[:,0]
                # print(pts.shape)
                pts[:,axis] = frag.workingLineAxisPosition()
                pts[:,2] = line[:,2]
                if triLineSize > 0:
                    for i in range(pts.shape[0]-1):
                        xy0 = self.ijToXy(pts[i,0:2])
                        xy1 = self.ijToXy(pts[i+1,0:2])
                        cv2.line(outrgbx, xy0, xy1, lineColor, triLineSize)
                        if not apply_mesh_opacity:
                            cv2.line(original, xy0, xy1, lineColor, triLineSize)

                # vpts = frag.workingVpoints()
                # pts = vpts[:, 0:2]
                wvflags = frag.workingVpoints()
                allpts = frag.vpoints
                vpts = allpts[wvflags]
                pts = vpts[:, 0:2]
                if fragNodeSize > 0:
                    for i,pt in enumerate(pts):
                        xy = self.ijToXy(pt[0:2])
                        color = self.nodeColor
                        # all frags are active at this point
                        # if not frag.activeAndAligned():
                        #     color = self.inactiveNodeColor
                        ijk = frag.vpoints[i]
                        if frag == pv.nearby_node_fv and vpts[i,3] == pv.nearby_node_index:
                            color = self.highlightNodeColor
                        self.drawNodeAtXy(outrgbx, xy, color, fragNodeSize)
                        if not apply_frag_node_opacity:
                            self.drawNodeAtXy(original, xy, color, fragNodeSize)

            elif fragNodeSize > 0:
                # pts = frag.fpoints[:, 0:2]
                # vpts = frag.workingVpoints()
                # pts = vpts[:, 0:2]
                wvflags = frag.workingVpoints()
                allpts = frag.vpoints[:, 0:2]
                wpts = allpts[wvflags]
                # print("pts shape", pts.shape)
                color = self.nodeColor
                size = fragNodeSize
                if not frag.mesh_visible:
                    color = frag.fragment.cvcolor
                # all frags are active at this point
                # if not frag.activeAndAligned():
                #     color = self.inactiveNodeColor
                vrts = self.ijsToXys(wpts)
                vrts = vrts.reshape(-1,1,1,2).astype(np.int32)
                cv2.polylines(outrgbx, vrts, True, color, 2*size)
                if not apply_frag_node_opacity:
                    cv2.polylines(original, vrts, True, color, 2*size)
                if frag == pv.nearby_node_fv and pv.nearby_node_index >= 0:
                    nni = pv.nearby_node_index
                    if wvflags[nni]:
                        pt = allpts[nni]
                        xy = self.ijToXy(pt)
                        color = self.highlightNodeColor
                        self.drawNodeAtXy(outrgbx, xy, color, size)
                        if not apply_frag_node_opacity:
                            self.drawNodeAtXy(original, xy, color, size)

            if frag.active:
                wvflags = frag.workingVpoints()
                allpts = frag.vpoints[:, 0:2]
                wpts = allpts[wvflags]
                i0 = len(xypts)
                for i,pt in enumerate(frag.vpoints[wvflags]):
                    ij = self.tijkToIj(pt)
                    xy = self.ijToXy(ij)
                    xypts.append((xy[0], xy[1], pt[0], pt[1], pt[2], pt[3]))
                    self.cur_frag_pts_fv.append(frag)
                    if frag == pv.nearby_node_fv and pt[3] == pv.nearby_node_index:
                        self.nearbyNode = i+i0
            timer.time("compute cur_frag_pts")
        timera.time("draw frag")
        self.cur_frag_pts_xyijk = np.array(xypts)

        if self.getDrawWidth("labels") > 0:
            self.drawScaleBar(outrgbx)
            self.drawTrackingCursor(outrgbx)
            if not apply_labels_opacity:
                self.drawScaleBar(original)
                self.drawTrackingCursor(original)

        if opacity > 0 and opacity < 1:
            outrgbx = cv2.addWeighted(outrgbx, opacity, original, 1.-opacity, 0)
        elif opacity == 0:
            outrgbx = original
        outrgbx = np.append(outrgbx, np.zeros((wh,ww,1), dtype=np.uint16), axis=2)
        # print("outrgbx", outrgbx.shape, outrgbx[250,250])

        bytesperline = 8*outrgbx.shape[1]
        qimg = QImage(outrgbx, outrgbx.shape[1], outrgbx.shape[0], 
                bytesperline, QImage.Format_RGBX64)
        pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(pixmap)
        timera.time("draw to qt")
        # print("--------------------")

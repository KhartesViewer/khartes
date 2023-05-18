import pathlib
import json
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.interpolate import CubicSpline
from utils import Utils
from PyQt5.QtGui import QColor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import cv2
import nrrd

class ColorSelectorDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, table, parent=None):
        super(ColorSelectorDelegate, self).__init__(parent)
        self.table = table
        # self.color = QColor()

    def createEditor(self, parent, option, index):
        # print("ce", index.row())
        cb = QtWidgets.QPushButton(parent)
        # cb.setFixedWidth()
        cb.setContentsMargins(5,5,5,5)
        cb.clicked.connect(lambda d: self.onClicked(d, cb, index))
        # self.index = index
        self.push_button = cb
        return cb

    def onClicked(self, cb_index, push_button, model_index):
        # self.table.model().setData(model_index, combo_box.currentText(), Qt.EditRole)
        old_color = push_button.palette().color(QtGui.QPalette.Window)
        # old_color = push_button.palette().background().color()
        new_color = QtWidgets.QColorDialog.getColor(old_color, self.table)
        # print("old_color",old_color.name(),"new_color",new_color.name())
        if new_color.isValid() and new_color != old_color:
            self.setColor(push_button, new_color.name())
            self.table.model().setData(model_index, new_color, Qt.EditRole)

    # this could be a class function
    def setColor(self, push_button, color):
        # color is a string, not a qcolor
        # print("pb setting color", color)
        push_button.setStyleSheet("background-color: %s"%color)

    def setEditorData(self, editor, index):
        # print("sed", index.row(), index.data(Qt.EditRole))
        # print("sed", index.row(), index.data(Qt.DisplayRole))
        color = index.data(Qt.DisplayRole)
        # color is a string
        if color:
            # editor.setCurrentIndex(cb_index)
            self.setColor(editor, color)

    def setModelData(self, editor, model, index):
        old_color = editor.palette().color(QtGui.QPalette.Window)
        # print("csd smd", old_color.name())

    def displayText(self, value, locale):
        return ""


class DirectionSelectorDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, table, parent=None):
        super(DirectionSelectorDelegate, self).__init__(parent)
        self.table = table

    def createEditor(self, parent, option, index):
        # print("ce", index.row())
        cb = QtWidgets.QComboBox(parent)
        # row = index.row()
        cb.addItem("X")
        cb.addItem("Y")
        cb.activated.connect(lambda d: self.onActivated(d, cb, index))
        # self.index = index
        self.combo_box = cb
        return cb

    def onActivated(self, cb_index, combo_box, model_index):
        self.table.model().setData(model_index, combo_box.currentText(), Qt.EditRole)

    def setEditorData(self, editor, index):
        # print("sed", index.row(), index.data(Qt.EditRole))
        # print("sed", index.row(), index.data(Qt.DisplayRole))
        cb_index = index.data(Qt.DisplayRole)
        if cb_index >= 0:
            editor.setCurrentIndex(cb_index)
        # return editor
        # for i in range(15):
        #     print(i, index.data(i))
        # otxt = index.data(Qt.EditRole)
        # cb_index = editor.findText(otxt)
        # if index >= 0:
        #     editor.setCurrentIndex(index)
        # return editor

    def setModelData(self, editor, model, index):
        pass
        # print("smd", editor.currentText())
        # do nothing, since onActivated handled it
        # model.setData(index, editor.currentText(), Qt.EditRole)

    # def updateEditorGeometry(self, editor, option, index):
    #     # print("ueg")
    #     editor.setGeometry(option.rect)


    '''
    void ButtonColumnDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    QPushButton button(index.data().toString());
    button.setGeometry(option.rect);
    painter->save();
    painter->translate(option.rect.topLeft());
    button.render(painter);
    painter->restore();
}
    '''

    '''
    def paint(self, painter, option, index):
        cb = QtWidgets.QComboBox()
        cb.addItem("X")
        cb.addItem("Y")
        cb.setGeometry(option.rect)
        painter.save()
        painter.translate(option.rect.topLeft())
        cb.render(painter)
        painter.restore()
    '''

class FragmentsModel(QtCore.QAbstractTableModel):
    def __init__(self, project_view, main_window):
        super(FragmentsModel, self).__init__()
        # note that self.project_view should not be
        # changed after initialization; instead, a new
        # instance of VolumesModel should be created
        # and attached to the QTableView
        self.project_view = project_view
        self.main_window = main_window

    columns = [
            "Active",
            "Visible",
            "Name",
            "Color",
            "Dir",
            "Pts"
            ]

    ctips = [
            "Select which fragment is active;\nclick box to select.\nNote that you can only select fragments\nwhich have the same direction (orientation)\nas the current volume view",
            "Select which fragments are visible;\nclick box to select",
            "Name of the fragment; click to edit",
            "Color of the fragment; click to edit",
            "Direction (orientation) of the fragment",
            "Number of points currently in fragment"
            ]
    
    def flags(self, index):
        col = index.column()
        # if col in (0,1,2):
        #     return Qt.ItemIsEditable|Qt.ItemIsEnabled
        oflags = super(FragmentsModel, self).flags(index)
        if col == 0:
            nflags = Qt.ItemNeverHasChildren
            row = index.row()
            fragments = self.project_view.fragments
            fragment = list(fragments.keys())[row]
            fragment_view = fragments[fragment]
            if self.project_view.cur_volume_view.direction == fragment.direction:
                nflags |= Qt.ItemIsUserCheckable
                nflags |= Qt.ItemIsEnabled
            return nflags
        elif col == 1:
            # print(col, int(oflags))
            nflags = Qt.ItemNeverHasChildren
            nflags |= Qt.ItemIsUserCheckable
            nflags |= Qt.ItemIsEnabled
            # nflags |= Qt.ItemIsEditable
            return nflags
        elif col== 2:
            nflags = Qt.ItemNeverHasChildren
            # nflags |= Qt.ItemIsUserCheckable
            nflags |= Qt.ItemIsEnabled
            nflags |= Qt.ItemIsEditable
            return nflags
        else:
            return Qt.ItemNeverHasChildren|Qt.ItemIsEnabled

    def headerData(self, section, orientation, role):
        if orientation != Qt.Horizontal:
            return
        
        if role == Qt.DisplayRole:
            if section == 0:
                # print("HD", self.rowCount())
                table = self.main_window.fragments_table
                # make sure the color button in column 3 is always open
                # (so no double-clicking required)
                for i in range(self.rowCount()):
                    index = self.createIndex(i, 3)
                    table.openPersistentEditor(index)

            return FragmentsModel.columns[section]
        elif role == Qt.ToolTipRole:
            return FragmentsModel.ctips[section]

    def columnCount(self, parent=None):
        return len(FragmentsModel.columns)

    def rowCount(self, parent=None):
        if self.project_view is None:
            return 0
        fragments = self.project_view.fragments
        # print("row count", len(fragments.keys()))
        return len(fragments.keys())

    # columns: name, color, direction, xmin, xmax, xstep, y..., img...

    def data(self, index, role):
        if self.project_view is None:
            return None
        if role == Qt.DisplayRole:
            return self.dataDisplayRole(index, role)
        elif role == Qt.TextAlignmentRole:
            return self.dataAlignmentRole(index, role)
        elif role == Qt.BackgroundRole:
            return self.dataBackgroundRole(index, role)
        elif role == Qt.CheckStateRole:
            return self.dataCheckStateRole(index, role)
        return None

    def dataCheckStateRole(self, index, role):
        column = index.column()
        row = index.row()
        fragments = self.project_view.fragments
        fragment = list(fragments.keys())[row]
        fragment_view = fragments[fragment]
        selected = (self.project_view.cur_fragment == fragment)
        if column == 0:
            if selected:
                return Qt.Checked
            else:
                return Qt.Unchecked
        if column == 1:
            if fragment_view.visible:
                return Qt.Checked
            else:
                return Qt.Unchecked

    def dataAlignmentRole(self, index, role):
        # column = index.column()
        # if column >= 4:
        #     return Qt.AlignVCenter + Qt.AlignRight
        return Qt.AlignVCenter + Qt.AlignRight

    def dataBackgroundRole(self, index, role):
        row = index.row()
        fragments = self.project_view.fragments
        fragment = list(fragments.keys())[row]
        # fragment_view = fragments[fragment]
        if self.project_view.cur_fragment == fragment:
            # return QtGui.QColor('lightgray')
            return QtGui.QColor('beige')

    def dataDisplayRole(self, index, role):
        row = index.row()
        column = index.column()
        fragments = self.project_view.fragments
        fragment = list(fragments.keys())[row]
        fragment_view = fragments[fragment]
        selected = (self.project_view.cur_fragment == fragment)
        # if column == 0:
        #     if selected:
        #         return Qt.Checked
        #     else:
        #         return Qt.Unchecked
        #     # return selected
        if column == 2:
            return fragment.name
        elif column == 3:
            # print("ddr", row, volume_view.color.name())
            return fragment.color.name()
        elif column == 4:
            # return "%s"%(('X','Y')[volume_view.direction])
            # print("data display role", row, volume_view.direction)
            return ('X','Y')[fragment.direction]
        elif column == 5:
            return len(fragment.gpoints)
        else:
            return None

    def setData(self, index, value, role):
        row = index.row()
        column = index.column()
        # print("setdata", row, column, value, role)
        # if role != Qt.EditRole:
        #     return False
        if role == Qt.CheckStateRole and column == 0:
            # print(row, value)
            if value != Qt.Checked:
                self.main_window.setCurrentFragment(None)
            else:
                fragments = self.project_view.fragments
                fragment = list(fragments.keys())[row]
                fragment_view = fragments[fragment]
                self.main_window.setCurrentFragment(fragment)
            return True
        elif role == Qt.CheckStateRole and column == 1:
            # print(row, value)
            fragments = self.project_view.fragments
            fragment = list(fragments.keys())[row]
            fragment_view = fragments[fragment]
            self.main_window.setFragmentVisibility(fragment, value==Qt.Checked)
            return True
        elif role == Qt.EditRole and column == 3:
            # print("sd color", value)
            fragments = self.project_view.fragments
            fragment = list(fragments.keys())[row]
            # fragment_view = fragments[fragment]
            # print("setdata", row, color.name())
            # fragment.setColor(color)
            self.main_window.setFragmentColor(fragment, value)
        elif role == Qt.EditRole and column == 2:
            # print("setdata", row, value)
            name = value
            # print("sd name", value)
            fragments = self.project_view.fragments
            fragment = list(fragments.keys())[row]
            # TODO: set fragment name
            # self.main_window.setDirection(volume, direction)

        return False

    def scrollToEnd(self):
        table = self.main_window.fragments_table
        rows = self.rowCount()
        print("rows", rows)
        index = self.createIndex(rows-1, 0)
        table.scrollTo(index)


    '''
    def setProjectView(pv):
        self.beginResetModel()
        project_view = pv
        self.endResetModel()
    '''

class VolumesModel(QtCore.QAbstractTableModel):
    def __init__(self, project_view, main_window):
        super(VolumesModel, self).__init__()
        # note that self.project_view should not be
        # changed after initialization; instead, a new
        # instance of VolumesModel should be created
        # and attached to the QTableView
        self.project_view = project_view
        self.main_window = main_window

    columns = [
            "Use",
            "Name",
            "Color",
            "Ld",
            "Dir",
            "X min",
            "X max",
            "dX",
            "Y min",
            "Y max",
            "dY",
            "Z min",
            "Z max",
            "dZ",
            ]

    ctips = [
            "Select which volume is visible;\nclick box to select",
            "Name of the volume",
            "Color of the volume outline drawn on slices;\nclick to edit",
            "Is volume currently loaded in memory\n(volumes that are not currently displayed\nare unloaded by default)",
            """Direction (orientation) of the volume;
X means that the X axis in the original tiff 
files is aligned with the vertical axes of the slice
displays; Y means that the Y axis in the original
tiff files is aligned with the slice vertical axes""",
            "Minimum X coordinate of the volume,\nrelative to tiff coordinates",
            "Maximum X coordinate of the volume,\nrelative to tiff coordinates",
            "X step (number of pixels stepped\nin the X direction in the tiff image\nfor each pixel in the slices)",
            "Minimum Y coordinate of the volume,\nrelative to tiff coordinates",
            "Maximum Y coordinate of the volume,\nrelative to tiff coordinates",
            "Y step (number of pixels stepped\nin the X direction in the tiff image\nfor each pixel in the slices)",
            "Minimum Z coordinate (image number) of the volume",
            "Maximum Z coordinate (image number) of the volume",
            "Z step (number of tiff images stepped for each slice image)",
            ]
    
    def flags(self, index):
        col = index.column()
        # if col in (0,1,2):
        #     return Qt.ItemIsEditable|Qt.ItemIsEnabled
        oflags = super(VolumesModel, self).flags(index)
        if col == 0:
            # print(col, int(oflags))
            nflags = Qt.ItemNeverHasChildren
            nflags |= Qt.ItemIsUserCheckable
            nflags |= Qt.ItemIsEnabled
            # nflags |= Qt.ItemIsEditable
            return nflags
        elif col== 2 or col == 4:
            nflags = Qt.ItemNeverHasChildren
            # nflags |= Qt.ItemIsUserCheckable
            nflags |= Qt.ItemIsEnabled
            # nflags |= Qt.ItemIsEditable
            return nflags
        else:
            return Qt.ItemNeverHasChildren|Qt.ItemIsEnabled

    def headerData(self, section, orientation, role):
        if orientation != Qt.Horizontal:
            return None
        
        if role == Qt.DisplayRole:
            if section == 0:
                # print("HD", self.rowCount())
                table = self.main_window.volumes_table
                # make sure the combo box in column 4 is always open
                # (so no double-clicking required)
                for i in range(self.rowCount()):
                    index = self.createIndex(i, 2)
                    table.openPersistentEditor(index)
                    index = self.createIndex(i, 4)
                    table.openPersistentEditor(index)

            return VolumesModel.columns[section]
        elif role == Qt.ToolTipRole:
            return VolumesModel.ctips[section]

    def columnCount(self, parent=None):
        return len(VolumesModel.columns)

    def rowCount(self, parent=None):
        if self.project_view is None:
            return 0
        volumes = self.project_view.volumes
        return len(volumes.keys())

    # columns: name, color, direction, xmin, xmax, xstep, y..., img...

    def data(self, index, role):
        if self.project_view is None:
            return None
        if role == Qt.DisplayRole:
            return self.dataDisplayRole(index, role)
        elif role == Qt.TextAlignmentRole:
            return self.dataAlignmentRole(index, role)
        elif role == Qt.BackgroundRole:
            return self.dataBackgroundRole(index, role)
        elif role == Qt.CheckStateRole:
            return self.dataCheckStateRole(index, role)
        return None

    def dataCheckStateRole(self, index, role):
        column = index.column()
        row = index.row()
        volumes = self.project_view.volumes
        volume = list(volumes.keys())[row]
        selected = (self.project_view.cur_volume == volume)
        if column == 0:
            if selected:
                return Qt.Checked
            else:
                return Qt.Unchecked

    def dataAlignmentRole(self, index, role):
        # column = index.column()
        # if column >= 4:
        #     return Qt.AlignVCenter + Qt.AlignRight
        return Qt.AlignVCenter + Qt.AlignRight

    def dataBackgroundRole(self, index, role):
        row = index.row()
        volumes = self.project_view.volumes
        volume = list(volumes.keys())[row]
        # volume_view = volumes[volume]
        if self.project_view.cur_volume == volume:
            # return QtGui.QColor('lightgray')
            return QtGui.QColor('beige')

    def dataDisplayRole(self, index, role):
        row = index.row()
        column = index.column()
        volumes = self.project_view.volumes
        volume = list(volumes.keys())[row]
        volume_view = volumes[volume]
        mins = volume.gijk_starts
        steps = volume.gijk_steps 
        sizes = volume.sizes
        selected = (self.project_view.cur_volume == volume)
        # if column == 0:
        #     if selected:
        #         return Qt.Checked
        #     else:
        #         return Qt.Unchecked
        #     # return selected
        if column == 1:
            return volume.name
        elif column == 2:
            # print("ddr", row, volume_view.color.name())
            return volume_view.color.name()
        elif column == 3:
            if volume.data is None:
                return 'No'
            else:
                return 'Yes'
        elif column == 4:
            # return "%s"%(('X','Y')[volume_view.direction])
            # print("data display role", row, volume_view.direction)
            return (0,1)[volume_view.direction]

        elif column >= 5:
            i3 = column-5
            i = i3//3
            j = i3 %3
            x0 = mins[i]
            dx = steps[i]
            nx = sizes[i]
            if j == 0:
                return x0
            elif j == 1:
                return x0+dx*(nx-1)
            else:
                return dx
        else:
            return None

    def setData(self, index, value, role):
        row = index.row()
        column = index.column()
        # print("setdata", row, column, value, role)
        # if role != Qt.EditRole:
        #     return False
        if role == Qt.CheckStateRole and column == 0:
            # print(row, value)
            if value != Qt.Checked:
                return False
            volumes = self.project_view.volumes
            volume = list(volumes.keys())[row]
            volume_view = volumes[volume]
            self.main_window.setVolume(volume)
            return True
        if role == Qt.EditRole and column == 2:
            color = value
            volumes = self.project_view.volumes
            volume = list(volumes.keys())[row]
            volume_view = volumes[volume]
            # print("setdata", row, color.name())
            # volume_view.setColor(color)
            self.main_window.setVolumeViewColor(volume_view, color)
        if role == Qt.EditRole and column == 4:
            # print("setdata", row, value)
            direction = 0
            if value == 'Y':
                direction = 1
            volumes = self.project_view.volumes
            volume = list(volumes.keys())[row]
            self.main_window.setDirection(volume, direction)

        return False


    '''
    def setProjectView(pv):
        self.beginResetModel()
        project_view = pv
        self.endResetModel()
    '''



class VolumeView():

    def __init__(self, project_view, volume):
        self.project_view = project_view
        self.volume = volume
        self.direction = 1
        self.trdata = None
        # itf, jtf, ktf are i,j,k of focus point (crosshairs)
        # in transposed-grid coordinates
        # so focus pixel = datatr[ktf, jtf, itf]
        self.ijktf = (0,0,0)
        self.zoom = 0.
        self.minZoom = 0.
        self.maxZoom = 5
        self.voxelSizeUm = 7.91
        self.apparentVoxelSize = self.voxelSizeUm
        gs = volume.gijk_steps
        # take the geometric mean of the step sizes
        sizemult = (gs[0]*gs[1]*gs[2])**(1./3.)
        self.apparentVoxelSize = sizemult*self.voxelSizeUm 
        # It would seem to make more sense to attach the
        # color to Volume, rather than to VolumeView, just as
        # a fragment's color is attached to Fragment rather
        # than to FragmentView.
        # However, if a user changes a volume's color, we
        # don't want to have to re-write the entire NRRD file.
        # So associate color with VolumeView instead.
        color = Utils.getNextColor()
        self.setColor(color)
        # self.color = QColor()
        # self.cvcolor = (0,0,0,0)

    def setColor(self, qcolor):
        self.color = qcolor
        rgba = qcolor.getRgbF()
        self.cvcolor = [int(65535*c) for c in rgba] 

    def setZoom(self, zoom):
        self.zoom = min(self.maxZoom, max(zoom, self.minZoom))

    # direction=0: "depth" slice is constant-x (y,z plane)
    # direction=1: "depth" slice is constant-y (z,x plane)
    # Another way to look at it is direction=0 means
    # that the X axis of the original tiff images is aligned
    # with the vertical axis in the top two slices;
    # direction=1 means that the Y axis of the original tiff
    # images is vertical.
    # original data is accessed as [slice, y, x]
    # note that origin of image is upper-left corner
    # image is accessed as [y, x]
    # call after data is loaded
    def setDirection(self, direction):
        if self.direction != direction:
            ijk = self.ijktf
            self.ijktf = (ijk[2], ijk[1], ijk[0])
        self.direction = direction
        if self.volume.data is not None:
            self.trdata = self.volume.trdatas[direction]
        else:
            print("warning, VolumeView.setDirection: volume data is not loaded")
            self.trdata = None

    def dataLoaded(self):
        self.trdata = self.volume.trdatas[self.direction]

    # call after direction is set
    def getDefaultZoom(self, window):
        depth, xline, inline = self.getSlices((0,0,0))
        minzoom = 0.
        for sh,sz in zip(
              [depth.shape, xline.shape, inline.shape],
              [window.depth.size(), window.xline.size(), window.inline.size()]):
            # print(sh,sz)
            shx = sh[1]
            szx = sz.width()
            shy = sh[0]
            szy = sz.height()
            mn = min(szx/shx, szy/shy)
            if minzoom == 0 or mn < minzoom:
                minzoom = mn
        # print("minzoom", minzoom)
        return minzoom

    def setDefaultMinZoom(self, window):
        dzoom = self.getDefaultZoom(window)
        self.minZoom = min(.05, .5*dzoom)

    # call after direction is set
    def setDefaultParameters(self, window):
        # zoom=1 means one pixel in grid should
        # be the same size as one pixel in viewing window
        self.zoom = self.getDefaultZoom(window)
        self.minZoom = .5*self.zoom
        sh = self.trdata.shape
        # itf, jtf, ktf are ijk of focus point in tranposed grid
        # value at focus point is trdata[ktf,jtf,itf]
        itf = int(sh[2]/2)
        jtf = int(sh[1]/2)
        ktf = int(sh[0]/2)
        self.ijktf = (itf, jtf, ktf)

        gi,gj,gk = self.transposedIjkToGlobalPosition(
                self.ijktf)
        print("global position x %d y %d image %d"%(gi,gj,gk))

    def setIjkTf(self, tf):
        o = [0,0,0]

        for i in range(0,3):
            t = tf[i]
            m = self.trdata.shape[2-i] - 1
            t = min(m, max(t,0))
            o[i] = t
        self.ijktf = tuple(o)

    def ijkToTransposedIjk(self, ijk):
        return self.volume.ijkToTransposedIjk(ijk, self.direction)

    def transposedIjkToIjk(self, ijkt):
        return self.volume.transposedIjkToIjk(ijkt, self.direction)

    def transposedIjkToGlobalPosition(self, ijkt):
        return self.volume.transposedIjkToGlobalPosition(ijkt, self.direction)

    def globalPositionsToTransposedIjks(self, gpoints):
        return self.volume.globalPositionsToTransposedIjks(gpoints, self.direction)

    def globalAxisFromTransposedAxis(self, axis):
        return self.volume.globalAxisFromTransposedAxis(axis, self.direction)

    def getSlice(self, axis, ijkt):
        return self.volume.getSlice(axis, ijkt, self.trdata)

    def getSlices(self, ijkt):
        return self.volume.getSlices(ijkt, self.trdata)

    def globalSlicePositionAlongAxis(axis, ijkt):
        return self.volume.globalSlicePositionAlongAxis(axis, ijkt, self.direction)


class Volume():

    def __init__(self):
        self.data = None
        # self.trdata = None
        self.trdatas = None
        self.data_header = None

        self.valid = False
        self.error = "no error message set"
        self.active_project_views = set()

    def createErrorVolume(err):
        vol = Volume()
        vol.error = err
        return vol

    # class function
    def sliceSize(start, stop, step):
        jmi = stop-start
        q = jmi // step
        r = jmi % step
        size = q
        if r != 0:
            size += 1
        return size

    # project is the project that the nrrd file will be added to
    # tiff_directory is the name of the directory containing the
    # tiff files
    # name is the stem of the output nrrd file 
    # ranges is:
    # [xmin, xmax, xstep], [ymin, ymax, ystep], [zmin, zmax, zstep]
    # (ints not strings)
    # where x and y correspond to positions in the tiff files,
    # and z is the tiff file number
    # these ranges are inclusive, so xmin=0 xmax=5 mean that
    # x values of 0,1,2,3,4,5 will be taken.
    # pattern is the format of the tiff file name, for instance 
    # "%05d.tif"
    # instead of setting pattern, a dictionary of the tiff filenames 
    # { int(z): filename, ...}
    # can be provided to filenames
    # callback, if specified, should take a string as argument, and return 
    # True to continue, False to stop
    # 
    # class function
    def createFromTiffs(project, tiff_directory, name, ranges, pattern, filenamedict=None, callback=None):
        # TODO: make sure 'name' is a valid file name
        # (lower case a-z, numbers, underscore, hyphen, space, dot 
        # (the last 3 not allowed to start or end a file name))
        xrange, yrange, zrange = ranges

        err = ""
        if xrange[0] < 0:
            err = "invalid x start value %d"%xrange[0]
        elif yrange[0] < 0:
            err = "invalid y start value %d"%yrange[0]
        elif xrange[0] >= xrange[1]:
            err = "x start value %d must be less than x end value %d"%(xrange[0],xrange[1])
        elif yrange[0] >= yrange[1]:
            err = "y start value %d must be less than y end value %d"%(yrange[0],yrange[1])
        elif xrange[2] <= 0:
            err = "x step %d must be greater than 0"%xrange[1]
        elif yrange[2] <= 0:
            err = "y step %d must be greater than 0"%yrange[1]
        elif zrange[2] <= 0:
            err = "image step %d must be greater than 0"%zrange[1]
        if err != "":
            print(err)
            return Volume.createErrorVolume(err)

        if pattern == "" and filenamedict is None:
            err = "need to specify either pattern or filenames"
            print(err)
            return Volume.createErrorVolume(err)

        tdir = pathlib.Path(tiff_directory)
        if not tdir.is_dir():
            err = "%s is not a directory"%tdir
            print(err)
            return Volume.createErrorVolume(err)
        oname = name
        if oname[-5:] != '.nrrd':
            oname += '.nrrd'
        ofilefull = project.volumes_path / oname
        if ofilefull.exists():
            err = "%s already exists"%ofilefull
            print(err)
            return Volume.createErrorVolume(err)
        try:
            open(ofilefull, 'wb')
        except Exception as e:
            # print("exception",e)
            err = "cannot write to file %s: %s"%(ofilefull, str(e))
            print(err)
            ofilefull.unlink(True)
            return Volume.createErrorVolume(err)
        wins = []
        xsize = Volume.sliceSize(xrange[0], xrange[1]+1, xrange[2])
        ysize = Volume.sliceSize(yrange[0], yrange[1]+1, yrange[2])
        zsize = Volume.sliceSize(zrange[0], zrange[1]+1, zrange[2])
        ocube = np.zeros((zsize, ysize, xsize), dtype=np.uint16)
        for i,z in enumerate(range(zrange[0], zrange[1]+1, zrange[2])):
            if pattern == "":
                if z not in filenamedict:
                    err = "file for image %d is missing"%z
                    print(err)
                    ofilefull.unlink(True)
                    return Volume.createErrorVolume(err)
                fname = filenamedict[z]
            else:
                fname = pattern%z
            imgf = tdir / fname
            # print(fname, imgf)
            if callback is not None and not callback("Reading %s"%fname):
                ofilefull.unlink(True)
                return Volume.createErrorVolume("Cancelled by user")
            iarr = None
            try:
                # note that imread doesn't throw an exception
                # when it cannot find the file, it simply returns
                # None
                iarr = cv2.imread(str(imgf), cv2.IMREAD_UNCHANGED)
            except cv2.error as e:
                err = "could not read file %s: %s"%(imgf, str(e))
                print(err)
                ofilefull.unlink(True)
                return Volume.createErrorVolume(err)
            if iarr is None:
                err = "failed to read file %s"%(imgf)
                print(err)
                ofilefull.unlink(True)
                return Volume.createErrorVolume(err)

            if xrange[1]+1 > iarr.shape[1]:
                err = "max requested x value %d is outside x range %d of image %s"%(xrange[1],iarr.shape[1]-1, fname)
                print(err)
                ofilefull.unlink(True)
                return Volume.createErrorVolume(err)
            if yrange[1]+1 > iarr.shape[0]:
                err = "requested y range %d to %d is outside y range %d of image %s"%(yrange[1], iarr.shape[0]-1, fname)
                ofilefull.unlink(True)
                return Volume.createErrorVolume(err)
            ocube[i] = np.copy(
                    iarr[yrange[0]:yrange[1]+1:yrange[2], 
                        xrange[0]:xrange[1]+1:xrange[2]])
            # ocube[i] = iarr[yrange[0]:yrange[1]+1:yrange[2], 
            #             xrange[0]:xrange[1]+1:xrange[2]]
            # wins.append(oarr)

        print("beginning stack")
        if callback is not None and not callback("Stacking images"):
            ofilefull.unlink(True)
            return Volume.createErrorVolume("Cancelled by user")
        # ocube = np.stack(wins)
        timestamp = Utils.timestamp()
        header = {
                "khartes_xyz_starts": "%d %d %d"%(xrange[0], yrange[0], zrange[0]),
                "khartes_xyz_steps": "%d %d %d"%(xrange[2], yrange[2], zrange[2]),
                "khartes_version": "1.0",
                "khartes_created": timestamp,
                "khartes_modified": timestamp,
                # turns off the default gzip compression (scroll images
                # don't compress well, so compression only slows down
                # the I/O speed)
                "encoding": "raw",
                }
        print("beginning transpose")
        tocube = np.transpose(ocube)
        print("beginning write to %s"%ofilefull)
        if callback is not None and not callback("Beginning write to %s"%ofilefull):
            return Volume.createErrorVolume("Cancelled by user")
        # the option index_order='C' seems to make nrrd.write
        # run very slowly, so better to pass a transposed matrix
        # to nrrd.write
        # nrrd.write(str(ofilefull), ocube, header, index_order='C')
        nrrd.write(str(ofilefull), tocube, header)
        print("file %s saved"%ofilefull)
        volume = Volume.loadNRRD(ofilefull)
        project.addVolume(volume)
        
        return volume

    def loadData(self, project_view):
        if self.data is not None:
            return
        print("reading data from",self.path,"for",self.name)
        # data = nrrd.read_data(str(self.path), index_order='C')
        # need to call nrrd.read rather than nrrd.read_data,
        # because nrrd.read_data has complicated prerequisites
        data, data_header = nrrd.read(str(self.path), index_order='C')
        print("finished reading")
        self.data = data
        self.createTransposedData()
        self.active_project_views.add(project_view)
        # self.setDirection(0)
        print(self.data.shape, self.trdatas[0].shape, self.trdatas[1].shape)

    def unloadData(self, project_view):
        self.active_project_views.discard(project_view)
        l = len(self.active_project_views)
        if l > 0:
            print("unloadData: still %d project views using this volume"%l)
            return
        print("unloading data for", self.name)
        self.data = None
        self.trdatas = None
        self.trdata = None

    def loadNRRD(filename, missing_allowed=False):
        try:
            print("reading header for",filename)
            data_header = nrrd.read_header(str(filename))
            print("finished reading")
            gijk_starts = [0,0,0]
            gijk_steps = [1,1,1]
            gijk_sizes = [0,0,0]
            version = 0.0
            if "khartes_version" in data_header:
                vstr = data_header["khartes_version"]
                try:
                    version = float(vstr)
                except:
                    pass
            if "khartes_xyz_starts" in data_header:
                # x, y, imageNumber
                xyzstr = data_header['khartes_xyz_starts']
                try:
                    gijk_starts = tuple(
                        int(x) for x in xyzstr.split())
                    print("gijk_starts", gijk_starts)
                except:
                    err="nrrd file could not parse khartes_xyz_starts string '%s'"%xyzstr
                    print(err)
                    return Volume.createErrorVolume(err)
            elif not missing_allowed:
                err="nrrd file %s is missing header 'khartes_xyz_starts'"%filename
                print(err)
                return Volume.createErrorVolume(err)
            if "khartes_xyz_steps" in data_header:
                xyzstr = data_header['khartes_xyz_steps']
                try:
                    gijk_steps = tuple(
                        int(x) for x in xyzstr.split())
                except:
                    err="nrrd file could not parse khartes_xyz_steps string '%s'"%xyzstr
                    print(err)
                    return Volume.createErrorVolume(err)
            elif not missing_allowed:
                err="nrrd file %s is missing header 'khartes_xyz_steps'"%filename
                print(err)
                return Volume.createErrorVolume(err)
            if "sizes" in data_header:
                sizes = data_header['sizes']
            else:
                err="nrrd file %s is missing header 'sizes'"%filename
                print(err)
                return Volume.createErrorVolume(err)
            created = data_header.get("khartes_created", "")
            modified = data_header.get("khartes_modified", "")
        except Exception as e:
            err = "Failed to read nrrd file %s: %s"%(filename,e)
            print(err)
            return Volume.createErrorVolume(err)
        volume = Volume()
        volume.data_header = data_header
        volume.gijk_starts = gijk_starts
        volume.gijk_steps = gijk_steps
        volume.version = version
        volume.created = created
        volume.modified = modified
        volume.valid = True
        volume.path = filename
        volume.name = filename.stem
        volume.data = None
        # convert np.int32 to int
        volume.sizes = tuple(int(sz) for sz in sizes)
        print(version, created, modified, volume.sizes)
        # print(data_header)
        # print(volume.data.shape, volume.trdatas[0].shape, volume.trdatas[1].shape)
        return volume

    def createTransposedData(self):
        self.trdatas = []
        self.trdatas.append(self.data.transpose(2,0,1))
        self.trdatas.append(self.data.transpose(1,0,2))

    def ijkToTransposedIjk(self, ijk, direction):
        i,j,k = ijk
        if direction == 0:
            return (j,k,i)
        else:
            # return (k,i,j)
            return (i,k,j)

    def transposedIjkToIjk(self, ijkt, direction):
        it,jt,kt = ijkt
        if direction == 0:
            return (kt,it,jt)
        else:
            # return (jt,kt,it)
            return (it,kt,jt)

    def transposedIjkToGlobalPosition(self, ijkt, direction):
        (i,j,k) = self.transposedIjkToIjk(ijkt, direction)
        # steps = self.data_header["steps"]
        # zeros = self.data_header["zeros"]
        steps = self.gijk_steps
        zeros = self.gijk_starts
        gi = zeros[0]+i*steps[0]
        gj = zeros[1]+j*steps[1]
        gk = zeros[2]+k*steps[2]
        return (gi, gj, gk)

    # gpoints is an array of ints.  Each row consists of gx, gy, gz
    # where gx and gy correspond to x and y on the tif image
    # and gz is the number of the image
    def globalPositionsToTransposedIjks(self, gpoints, direction):
        g0 = np.array(self.gijk_starts)
        dg = np.array(self.gijk_steps)
        ijks = (gpoints-g0)/dg
        if direction == 0:
            tijks = ijks[:,(1,2,0)]
        else:
            tijks = ijks[:,(0,2,1)]
        return tijks

    # returns range as [[imin,imax], [jmin,jmax], [kmin,kmax]]
    def getGlobalRanges(self):
        arr = []
        # steps = self.data_header["steps"]
        # zeros = self.data_header["zeros"]
        steps = self.gijk_steps
        zeros = self.gijk_starts
        for i in range(3):
            mn = zeros[i]
            n = self.data.shape[2-i]
            mx = mn+(n-1)*steps[i]
            arr.append([mn, mx])
        return arr

    def globalAxisFromTransposedAxis(self, axis, direction):
        if direction == 0:
            return (axis+1)%3
        else:
            # return (axis+2)%3
            # return (2,0,1)[axis]
            return (0,2,1)[axis]

    # it, jt, kt are ijk in transposed grid
    # it (fast, axis 0) is inline coord
    # jt (med, axis 1) is xline coord
    # kt (slow, axis 2) is depth coord
    # trdata is laid out in C style, so 
    # value at it,ij,ik is indexed trdata[kt,jt,it]
    def getSlice(self, axis, ijkt, trdata):
        (it,jt,kt) = ijkt
        if axis == 2: # depth
            return trdata[kt,:,:]
        elif axis == 1: # xline
            return trdata[:,jt,:]
        else: # inline
            return trdata[:,:,it]

    def ijIndexesInPlaneOfSlice(self, axis):
        return ((1,2), (0,2), (0,1))[axis]

    def ijInPlaneOfSlice(self, axis, ijkt):
        inds = self.ijIndexesInPlaneOfSlice(axis)
        return(ijkt[inds[0]], ijkt[inds[1]])

    def globalSlicePositionAlongAxis(self, axis, ijkt, direction):
        gi,gj,gk = self.transposedIjkToGlobalPosition(ijkt, direction)
        return (gi, gj, gk)[axis]

    # call after direction is set
    # it, jt, kt are ijk in transposed grid
    # it (fast, axis 0) is inline coord
    # jt (med, axis 1) is xline coord
    # kt (slow, axis 2) is depth coord
    # trdata is laid out in C style, so 
    # value at it,ij,ik is indexed trdata[kt,jt,it]
    def getSlices(self, ijkt, trdata):
        # print(self.trdata.shape, ijkt)
        depth = self.getSlice(2, ijkt, trdata)
        xline = self.getSlice(1, ijkt, trdata)
        inline = self.getSlice(0, ijkt, trdata)

        return (depth, xline, inline)


# note that FragmentView is defined after Fragment
class Fragment:

    def __init__(self, name, direction):
        self.direction = direction
        # self.color = QColor("green")
        self.color = QColor()
        self.cvcolor = (0,0,0,0)
        self.name = name
        # fragment points in global coordinates
        self.gpoints = np.zeros((0,3), dtype=np.int32)
        self.valid = False
        self.created = Utils.timestamp()
        self.modified = Utils.timestamp()

    # TODO: need "created" and "modified" timestamps
    def save(self, path):
        info = {}
        info['name'] = self.name
        info['direction'] = self.direction
        info['color'] = self.color.name()
        info['gpoints'] = self.gpoints.tolist()
        # print(info)
        info_txt = json.dumps(info, indent=4)
        file = path / (self.name + ".json")
        print("writing to",file)
        file.write_text(info_txt, encoding="utf8")

    def createErrorFragment():
        frag = Fragment("", -1)
        frag.error = err
        return frag

    # TODO: need "created" and "modified" timestamps
    def load(json_file):
        try:
            json_txt = json_file.read_text(encoding="utf8")
        except:
            err = "Could not read file %s"%json_file
            print(err)
            return Fragment.createErrorFragment(err)

        try:
            info = json.loads(json_txt)
        except:
            err = "Could not parse file %s"%json_file
            print(err)
            return Fragment.createErrorFragment(err)

        for attr in ['name', 'direction', 'gpoints']:
            if attr not in info:
                err = "file %s missing parameter %s"%(json_file, attr)
                print(err)
                return Fragment.createErrorFragment(err)

        if 'color' in info:
            color = QColor(info['color'])
        else:
            color = Utils.getNextColor()
        name = info['name']
        direction = info['direction']
        gpoints = info['gpoints']
        frag = Fragment(name, direction)
        frag.setColor(color)
        frag.valid = True
        if len(gpoints) > 0:
            frag.gpoints = np.array(gpoints, dtype=np.int32)
        return frag

    def setColor(self, qcolor):
        self.color = qcolor
        rgba = qcolor.getRgbF()
        self.cvcolor = [int(65535*c) for c in rgba] 

class FragmentView:

    def __init__(self, project_view, fragment):
        self.project_view = project_view
        self.fragment = fragment
        # cur_volume_view holds the volume associated
        # with current zsurf and ssurf
        self.cur_volume_view = None
        self.visible = True
        self.tri = None
        self.line = None
        self.lineAxis = -1
        self.lineAxisPosition = 0
        self.zsurf = None
        self.ssurf = None
        self.nearbyNode = -1
        # gpoints converted to ijk coordinates relative
        # to current volume, using trijk based on 
        # fragment's direction
        self.fpoints = np.zeros((0,4), dtype=np.float32)
        # same as above, but trijk based on cur_volume_view's 
        # direction
        self.vpoints = np.zeros((0,4), dtype=np.float32)

    def setVolumeView(self, vol_view):
        self.cur_volume_view = vol_view
        if vol_view is not None:
            self.setLocalPoints()
            self.createZsurf()

    # direction is not used here, but this notifies fragment view
    # to recompute things
    def setVolumeViewDirection(self, direction):
        self.setLocalPoints()
        self.createZsurf()

    def setLocalPoints(self):
        # print("set local points", self.cur_volume_view.volume.name)
        self.fpoints = self.cur_volume_view.volume.globalPositionsToTransposedIjks(self.fragment.gpoints, self.fragment.direction)
        npts = self.fpoints.shape[0]
        if npts > 0:
            indices = np.reshape(np.arange(npts), (npts,1))
            self.fpoints = np.concatenate((self.fpoints, indices), axis=1)

        if self.cur_volume_view is None:
            self.vpoints = None
        else:
            self.vpoints = self.cur_volume_view.volume.globalPositionsToTransposedIjks(self.fragment.gpoints, self.cur_volume_view.direction)
            if npts > 0:
                indices = np.reshape(np.arange(npts), (npts,1))
                # print(self.vpoints.shape, indices.shape)
                self.vpoints = np.concatenate((self.vpoints, indices), axis=1)
                # print(self.vpoints[0])


    def createZsurf(self):
        self.triangulate()
        nk,nj,ni = self.cur_volume_view.trdata.shape
        if self.fragment.direction != self.cur_volume_view.direction:
            ni,nj,nk = nk,nj,ni
        ns = (ni,nj,nk)
        self.zsurf = np.zeros((nj,ni), dtype=np.float32)
        self.zsurf.fill(np.nan)
        if self.tri is not None:
            # interp = LinearNDInterpolator(self.tri, self.lpoints[:,2])
            interp = CloughTocher2DInterpolator(self.tri, self.fpoints[:,2])
            pts = np.indices((ni, nj)).transpose()
            self.zsurf = interp(pts)
        if self.line is not None and self.lineAxis > -1:
            # print("createZsurf from line")
            spline = CubicSpline(
                    self.line[:,0], self.line[:,1], extrapolate=False)
            xs = np.arange(ns[1-self.lineAxis])
            ys = spline(xs)
            # print(self.zsurf.shape)
            # print(xs.shape)
            # print(self.lineAxis, self.lineAxisPosition)
            if self.lineAxis == 0:
                self.zsurf[:,self.lineAxisPosition] = ys
            else:
                self.zsurf[self.lineAxisPosition,:] = ys
            # print(ys)

        if self.fragment.direction != self.cur_volume_view.direction:
            self.ssurf = None
            return
        
        ssi = np.indices((ni, nj))
        # print("ssi shape",ssi.shape, ssi.dtype)
        xs = ssi[0].flatten()
        ys = ssi[1].flatten()
        # print ("xs shape", xs.shape, xs.dtype)
        xys = np.array((ys,xs), dtype = xs.dtype)
        # print ("xys shape", xys.shape, xys.dtype)
        zs = self.zsurf[tuple(xys)]
        # print ("zs shape", zs.shape, zs.dtype)
        xyzs = np.array((zs,ys,xs))
        # print ("xyzs shape", xyzs.shape, xyzs.dtype)
        xyzsn = xyzs[:,~np.isnan(zs)]
        # print ("xyzsn shape", xyzsn.shape, xyzsn.dtype)
        # Occurs when fragment has only 1 point
        if xyzsn.shape[1] == 0:
            # print("xyzsn wrong size")
            self.ssurf = None
            return
        ixyzs = np.rint(xyzsn).astype(np.int32)
        # print ("ixyzs shape", ixyzs.shape, ixyzs.dtype)
        ftrdata = self.cur_volume_view.volume.trdatas[self.fragment.direction]
        ## print("ixyzs max", ixyzs.max(axis=1))
        ## print("trdata", trdata.shape)
        # print("ixyzs rot max", ixyzs[(2,0,1),:].max(axis=1))
        # print("rixyzs max", rixyzs.max(axis=1))
        self.ssurf = np.zeros((nj,ni), dtype=np.uint16)
        # print ("ssurf shape", self.ssurf.shape, self.ssurf.dtype)
        # print ("trdata shape", self.cur_volume_view.trdata.shape, self.cur_volume_view.trdata.dtype)
        ## print("ssurf",self.ssurf.shape)

        # recall that index order is k,j,i
        ixyzs = ixyzs[:,ixyzs[0,:]<ftrdata.shape[0]]
        ixyzs = ixyzs[:,ixyzs[0,:]>=0]
        self.ssurf[(ixyzs[1,:],ixyzs[2,:])] = ftrdata[(ixyzs[0,:], ixyzs[1,:], ixyzs[2,:])]

    # returns zsurf points, as array of [ipos, jpos] values
    # for the slice with the given axis and axis position
    # (axis and position relative to volume-view axes)
    def getZsurfPoints(self, vaxis, vaxisPosition):
        if self.zsurf is None:
            return
        if self.fragment.direction == self.cur_volume_view.direction:
            nk,nj,ni = self.cur_volume_view.trdata.shape
            if vaxis == 0:
                ivec = np.arange(nj)
                jvec = self.zsurf[:,vaxisPosition]
                pts = np.array((ivec,jvec)).transpose()
                # print(pts.shape)
                pts = pts[~np.isnan(pts[:,1])]
                # print(pts.shape)
                return pts
            elif vaxis == 1:
                ivec = np.arange(ni)
                jvec = self.zsurf[vaxisPosition, :]
                # print(self.cur_volume_view.trdata.shape, pts.shape, self.zsurf.shape)
                # print(pts.shape)
                pts = np.array((ivec,jvec)).transpose()
                pts = pts[~np.isnan(pts[:,1])]
                return pts
            else:
                pts = np.indices((nj, ni))
                # print(pts.shape, self.zsurf.shape)
                pts = pts[:, np.rint(self.zsurf)==vaxisPosition].transpose()
                # print(pts.shape)
                pts = pts[:,(1,0)]
                # print(pts)
                return pts
        else:
            vnk,vnj,vni = self.cur_volume_view.trdata.shape
            fni,fnj,fnk = vnk,vnj,vni
            if vaxis == 0: # faxis = 2
                pts = np.indices((fnj, fni))
                # print(fni,fnj,fnk)
                # print(self.cur_volume_view.trdata.shape, pts.shape, self.zsurf.shape)
                pts = pts[:, np.rint(self.zsurf)==vaxisPosition].transpose()
                # print(pts.shape)
                # print(pts)
                return pts
            if vaxis == 1: # faxis = 1
                ivec = np.arange(fni)
                jvec = self.zsurf[vaxisPosition, :]
                pts = np.array((jvec,ivec)).transpose()
                pts = pts[~np.isnan(pts[:,0])]
                return pts
            else: # faxis = 0
                ivec = np.arange(fnj)
                jvec = self.zsurf[:,vaxisPosition]
                pts = np.array((jvec,ivec)).transpose()
                # print(pts.shape)
                pts = pts[~np.isnan(pts[:,0])]
                # print(pts.shape)
                return pts

    def triangulate(self):
        nppoints = self.fpoints[:,0:2]
        if self.fpoints.shape[0] == 0:
            self.tri = None
            self.line = None
            self.lineAxis = -1
            return
        try:
            self.tri = Delaunay(nppoints)
            self.line = None
            self.lineAxis = -1
        except QhullError as err:
            # print("qhull error")
            self.tri = None
            if self.fpoints.shape[0] > 1:
                # self.fpoints is addressed by:
                # pts[pt #, (i,j,depth)]
                self.lineAxis = -1
                iuniques = len(np.unique(self.fpoints[:,0]))
                juniques = len(np.unique(self.fpoints[:,1]))
                if iuniques == 1 and juniques > 1:
                    # points share a common i value
                    self.lineAxis = 0
                elif juniques == 1 and iuniques > 1:
                    # points share a common j value
                    self.lineAxis = 1
                if self.lineAxis >= 0:
                    self.lineAxisPosition = int(self.fpoints[0,self.lineAxis])
                    self.line = self.fpoints[:, (1-self.lineAxis, 2)]
                    self.line = self.line[self.line[:,0].argsort()]
                    # print(self.lineAxis, self.lineAxisPosition)
                    # print(self.fpoints)
                    # print(self.line)


    def getPointsOnSlice(self, axis, i):
        matches = self.vpoints[(self.vpoints[:, axis] == i)]
        return matches

    def vijkToFijk(self, vijk):
        if self.cur_volume_view.direction == self.fragment.direction:
            fijk = vijk
        else:
            fijk = (vijk[2], vijk[1], vijk[0])
        return fijk

    def fijkToVijk(self, fijk):
        if self.cur_volume_view.direction == self.fragment.direction:
            vijk = fijk
        else:
            vijk = (fijk[2], fijk[1], fijk[0])
        return vijk

    def addPoint(self, tijk):
        # ijk using volume-view's direction
        fijk = self.vijkToFijk(tijk)

        # use fijk rounded to nearest integer (note that this
        # may have consequences if current volume is coarser
        # than global volume
        ijk = np.rint(np.array(fijk))
        # if same ij (in fragment's direction) as existing
        # point, replace point with existing point
        ij = ijk[0:2]
        matches = np.where((np.rint(self.fpoints[:, 0:2]) == ij).all(axis=1))[0]
        # print(matches)
        if matches.shape[0] > 0:
            # print("duplicate at", tijk)
            # print("deleting duplicates at row", matches)
            # print("    ", self.vpoints[matches[0]])
            # delete old point if exists at same ij as new point
            self.fragment.gpoints = np.delete(self.fragment.gpoints, matches, 0)
        # create new point
        gijk = self.cur_volume_view.transposedIjkToGlobalPosition(tijk)
        self.fragment.gpoints = np.append(self.fragment.gpoints, np.reshape(gijk, (1,3)), axis=0)
        # print(self.lpoints)
        self.setLocalPoints()
        self.createZsurf()

    # return True if succeeds, False if fails
    def movePoint(self, index, new_vijk):
        old_fijk = self.fpoints[index]
        new_fijk = self.vijkToFijk(new_vijk)
        new_matches = np.where((np.rint(self.fpoints[:, 0:2]) == np.rint(new_fijk[0:2])).all(axis=1))[0]
        if (round(old_fijk[0]) != round(new_fijk[0]) or round(old_fijk[1]) != round(new_fijk[1])) and new_matches.shape[0] > 0:
            print("movePoint point already exists at this ij", new_vijk)
            return False
        new_gijk = self.cur_volume_view.transposedIjkToGlobalPosition(new_vijk)
        # print(self.fragment.gpoints)
        # print(match, new_gijk)
        self.fragment.gpoints[index, :] = new_gijk
        # print(self.fragment.gpoints)
        self.setLocalPoints()
        self.createZsurf()
        return True

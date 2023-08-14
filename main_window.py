from pathlib import Path
import shutil
import copy
import os
import json
import time

from PyQt5.QtWidgets import (
        QAction, QApplication, QAbstractItemView,
        QCheckBox, QComboBox,
        QDialog, QDialogButtonBox,
        QFileDialog, QFrame,
        QGridLayout, QGroupBox,
        QHBoxLayout, 
        QLabel, QLineEdit,
        QMainWindow, QMenuBar, QMessageBox,
        QPlainTextEdit, QPushButton,
        QSizePolicy,
        QSpacerItem, QSpinBox, QDoubleSpinBox,
        QStatusBar, QStyle, QStyledItemDelegate,
        QTableView, QTabWidget, QTextEdit, QToolBar,
        QVBoxLayout, 
        QWidget, 
        )
from PyQt5.QtCore import (
        QAbstractTableModel, QCoreApplication, QObject,
        QSize, QTimer, Qt, qVersion, QSettings,
        )
from PyQt5.QtGui import QPainter, QPalette, QColor, QCursor, QIcon, QPixmap, QImage

from PyQt5.QtSvg import QSvgRenderer

from PyQt5.QtXml import QDomDocument

from tiff_loader import TiffLoader
from data_window import DataWindow, SurfaceWindow
from project import Project, ProjectView
from fragment import Fragment, FragmentsModel, FragmentView
from volume import (
        Volume, VolumesModel, 
        DirectionSelectorDelegate,
        ColorSelectorDelegate)
from utils import Utils

class ColorBlock(QLabel):

    def __init__(self, color, text=""):
        super(ColorBlock, self).__init__()
        self.setAutoFillBackground(True)
        self.setText(text)
        self.setAlignment(Qt.AlignCenter)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)

class RoundnessSetter(QGroupBox):
    def __init__(self, main_window, parent=None):
        super(RoundnessSetter, self).__init__("Skinny border triangles", parent)
        self.main_window = main_window
        self.min_roundness = 0.
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        # gb = QGroupBox("Skinny border triangles", self)
        vlayout = QVBoxLayout()
        # gb.setLayout(vlayout)
        self.setLayout(vlayout)
        self.cb = QCheckBox("Hide skinny triangles")
        self.cb.clicked.connect(self.onClicked)
        vlayout.addWidget(self.cb)
        hlayout = QHBoxLayout()
        label = QLabel("Min roundness:")
        hlayout.addWidget(label)
        self.edit = QLineEdit()
        fm = self.edit.fontMetrics()
        w = 7*fm.width('0')
        self.edit.setFixedWidth(w)
        self.edit.editingFinished.connect(self.onEditingFinished)
        self.edit.textEdited.connect(self.onTextEdited)
        hlayout.addWidget(self.edit)
        hlayout.addStretch()
        vlayout.addLayout(hlayout)
        vlayout.addStretch()

    def onClicked(self, s):
        # s is bool
        # self.main_window.setVolBoxesVisible(s==Qt.Checked)
        # print("hide skinny trgls", s)
        self.setHideSkinnyTriangles(s)

    def setHideSkinnyTriangles(self, state):
        self.cb.setChecked(state)
        self.main_window.setHideSkinnyTriangles(state)

    def getHideSkinnyTriangles(self):
        return self.cb.checked()

    def setMinRoundness(self, value):
        self.min_roundness = value
        txt = "%.2f"%value
        self.edit.setText(txt)
        self.edit.setStyleSheet("")
        self.main_window.setMinRoundness(value)

    def getMinRoundness(self):
        return self.min_roundness

    def parseText(self, txt):
        valid = True
        f = 0
        try:
            f = float(txt)
        except:
            valid = False
        if f < 0 or f > 1:
            valid = False
        # f = min(f, 1.)
        # f = max(f, 0.)
        return valid, f

    def onTextEdited(self, txt):
        valid, f = self.parseText(txt)
        # print("ote", valid)
        if valid:
            self.edit.setStyleSheet("")
        else:
            self.edit.setStyleSheet("QLineEdit { color: red }")

    def onEditingFinished(self):
        txt = self.edit.text()
        valid, value = self.parseText(txt)
        if not valid:
            self.setMinRoundness(self.min_roundness)
            return
        self.setMinRoundness(value)
        # if value != self.min_roundness:
        #     # print("oef", valid, value)
        #     self.min_roundness = value
        #     self.main_window.setMinRoundness(value)


class InfillDialog(QDialog):
    def __init__(self, main_window, parent=None):
        instructions = "In order to fit the curved fragment surface,\ninfill points are added to the exported mesh.\nThe distance between points is given in voxels.\n16 is a good default.\n0 means do not infill."
        super(InfillDialog, self).__init__(parent)
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        hlayout = QHBoxLayout()
        vlayout.addLayout(hlayout)
        hlayout.addWidget(QLabel("Infill spacing"))
        self.edit = QLineEdit("16")
        fm = self.edit.fontMetrics()
        w = 7*fm.width('0')
        self.edit.setFixedWidth(w)
        self.edit.editingFinished.connect(self.onEditingFinished)
        self.edit.textEdited.connect(self.onTextEdited)
        hlayout.addWidget(self.edit)
        hlayout.addStretch()
        self.wlabel = QLabel("")
        vlayout.addWidget(self.wlabel)
        self.ilabel = QLabel(instructions)
        vlayout.addWidget(self.ilabel)
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accepted)
        bbox.rejected.connect(self.rejected)
        vlayout.addWidget(bbox)
        self.is_accepted = False
        # print("InfillDialog created")

    def getValue(self):
        txt = self.edit.text()
        valid, i = self.parseText(txt)
        if not valid or not self.is_accepted:
            i = -1
        # print("getValue", i)
        return i

    def accepted(self):
        self.is_accepted = True
        value = self.getValue()
        # print("accepted", value)
        if value < 0:
            self.edit.setText("")
            self.wlabel.setText('Please enter a valid value (an integer >= 0)\nor press "Cancel"')
            self.wlabel.setStyleSheet("QLabel { color: red; font-weight: bold }")
        else:
            self.close()

    def rejected(self):
        self.is_accepted = False
        # print("rejected")
        self.close()

    def onEditingFinished(self):
        txt = self.edit.text()
        valid, size = self.parseText(txt)
        # print("oef", valid, size)
        # if valid:
        #     self.main_window.setVoxelSizeUm(size)

    def parseText(self, txt):
        valid = True
        i = -1
        try:
            i = int(txt)
        except:
            valid = False
        if i < 0:
            valid = False
        return valid, i

    def onTextEdited(self, txt):
        valid, i = self.parseText(txt)
        # print("ote", valid)
        if valid:
            self.edit.setStyleSheet("")
        else:
            self.edit.setStyleSheet("QLineEdit { color: red }")

class ZInterpolationSetter(QWidget):
    def __init__(self, main_window, parent=None):
        super(ZInterpolationSetter, self).__init__(parent)
        self.main_window = main_window
        hlayout = QHBoxLayout()
        self.setLayout(hlayout)
        label = QLabel("Z interpolation:")
        hlayout.addWidget(label)
        cb = QComboBox()
        cb.addItem("Linear")
        cb.addItem("Nearest")
        cb.setCurrentIndex(0)
        cb.activated.connect(self.onActivated)
        hlayout.addWidget(cb)
        hlayout.addStretch()

    def onActivated(self, index):
        self.main_window.setZInterpolation(index)

class CreateFragmentButton(QPushButton):
    def __init__(self, main_window, parent=None):
        super(CreateFragmentButton, self).__init__("Start New Fragment", parent)
        self.main_window = main_window
        self.setToolTip("Once the new fragment is created use\nshift plus left mouse button to create new nodes")
        self.clicked.connect(self.onButtonClicked)

    def onButtonClicked(self, s):
        self.main_window.createFragment()


class CopyActiveFragmentButton(QPushButton):
    def __init__(self, main_window, parent=None):
        # super(CopyActiveFragmentButton, self).__init__("Copy Active Fragment", parent)
        super(CopyActiveFragmentButton, self).__init__("Copy", parent)
        self.main_window = main_window
        self.setToolTip("Create a new fragment that is a copy\nof the currently active fragment")
        self.clicked.connect(self.onButtonClicked)

    def onButtonClicked(self, s):
        self.main_window.copyActiveFragment()

class MoveActiveFragmentButton(QPushButton):
    def __init__(self, main_window, text, step, parent=None):
        super(MoveActiveFragmentButton, self).__init__(text, parent)
        self.main_window = main_window
        self.step = step
        # up and down are opposite signs to what you might expect
        if step > 0:
            self.setToolTip("Move active fragment %d pixel(s) down"%step)
        else:
            self.setToolTip("Move active fragment %d pixel(s) up"%(-step))
        self.clicked.connect(self.onButtonClicked)

    def onButtonClicked(self, s):
        self.main_window.moveActiveFragment(self.step)

class LiveZsurfUpdateButton(QPushButton):
    def __init__(self, main_window, parent=None):
        super(LiveZsurfUpdateButton, self).__init__("", parent)
        self.main_window = main_window
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.setStyleSheet("QPushButton {padding: 5}")
        # self.setCheckable(True)
        self.checked = False
        self.setText("LU")
        self.clicked.connect(self.onButtonClicked)
        self.setChecked(self.main_window.live_zsurf_update)

    def onButtonClicked(self, s):
        self.setChecked(not self.checked)
        '''
        self.checked = not self.checked
        self.main_window.setLiveZsurfUpdate(self.checked)
        if self.checked:
            self.setStyleSheet("QPushButton {padding: 5}")
        else:
            self.setStyleSheet("QPushButton { background-color: red ; padding: 5 }")
        self.doSetToolTip()
        '''

    def doSetToolTip(self):
        main_str = "Live Update mode:\nSets whether data slices will be updated in real time\nas nodes are modified"
        add_strs = [
                "\n(Currently not in live-update mode)",
                "\n(Currently in live-update mode)", 
                ]
        self.setToolTip(main_str+add_strs[int(self.checked)])

    def setChecked(self, flag):
        # super(LiveZsurfUpdateButton, self).setChecked(flag)
        self.checked = flag
        self.main_window.setLiveZsurfUpdate(self.checked)
        if self.checked:
            self.setStyleSheet("QPushButton {padding: 5}")
        else:
            self.setStyleSheet("QPushButton { background-color: red ; padding: 5 }")
        self.doSetToolTip()

class CursorModeButton(QPushButton):
    def __init__(self, main_window, parent=None):
        super(CursorModeButton, self).__init__("", parent)
        self.main_window = main_window
        self.setCheckable(True)
        '''
        self.cross = QIcon(QCursor(Qt.CrossCursor).pixmap())
        self.arrow = QIcon(QCursor(Qt.ArrowCursor).pixmap())
        self.cross = self.style().standardIcon(QStyle.SP_MessageBoxCritical)
        self.arrow = self.style().standardIcon(QStyle.SP_MessageBoxWarning)
        '''
        '''
        self.cross = QIcon(QPixmap(":/crosshair.svg"))
        svgr = QSvgRenderer(":/crosshair.svg")
        image = QImage(64,64,QImage.Format_ARGB32)
        image.fill(0x00000000)
        svgr.render(QPainter(image))
        pixmap = QPixmap.fromImage(image)
        self.cross = QIcon(pixmap)
        '''
        # path = QApplication.applicationDirPath()
        # path = QApplication.applicationDirPath()
        args = QCoreApplication.arguments()
        # path = os.path.dirname(os.path.realpath(sys.argv[0]))
        path = os.path.dirname(os.path.realpath(args[0]))
        # print("path is", path, args[0])
        # crosshair.svg is from https://iconduck.com/icons/14824/crosshair
        self.cross = QIcon(path+"/crosshair.svg")
        self.setIcon(self.cross)
        self.setChecked(self.main_window.add_node_mode)
        # self.doSetToolTip()
        self.clicked.connect(self.onButtonClicked)
        # print("cursor mode button")

    def onButtonClicked(self, s):
        # self.add_node_mode = not self.add_node_mode
        self.main_window.add_node_mode = self.isChecked()
        self.doSetToolTip()
        # print("add node mode:", self.main_window.add_node_mode)
        # if self.add_node_mode:
        #     self.setIcon(self.arrow)
        # else:
        #     self.setIcon(self.cross)

    def doSetToolTip(self):
        main_str = "Sets whether cursor is in add-node mode or panning mode"
        add_strs = [
                "\n(Currently in panning mode)", 
                "\n(Currently in add-node mode)" ]
        self.setToolTip(main_str+add_strs[self.main_window.add_node_mode])

    def setChecked(self, flag):
        super(CursorModeButton, self).setChecked(flag)
        self.doSetToolTip()


class VolBoxesVisibleCheckBox(QCheckBox):
    def __init__(self, main_window, parent=None):
        super(VolBoxesVisibleCheckBox, self).__init__("Volume Boxes Visible", parent)
        self.main_window = main_window
        # at the time this widget is created, state is unknown,
        # so no use to check it
        # self.setChecked(state)
        self.stateChanged.connect(self.onStateChanged)

    def onStateChanged(self, s):
        self.main_window.setVolBoxesVisible(s==Qt.Checked)

class TrackingCursorsVisibleCheckBox(QCheckBox):
    def __init__(self, main_window, parent=None):
        super(TrackingCursorsVisibleCheckBox, self).__init__("Show tracking cursors", parent)
        self.main_window = main_window
        # at the time this widget is created, state is unknown,
        # so no use to check it
        # self.setChecked(state)
        self.stateChanged.connect(self.onStateChanged)
        self.name = "tracking_cursors"
        self.class_name = "show"
        self.setChecked(main_window.draw_settings[self.name][self.class_name])
        main_window.draw_settings_widgets[self.name][self.class_name] = self

    def onStateChanged(self, s):
        # self.main_window.setVolBoxesVisible(s==Qt.Checked)
        self.main_window.setTrackingCursorsVisible(s==Qt.Checked)

    def updateValue(self, value):
        self.setChecked(value)

class VoxelSizeEditor(QWidget):
    def __init__(self, main_window, parent=None):
        super(VoxelSizeEditor, self).__init__(parent)
        self.main_window = main_window
        # print("widget margins", self.contentsMargins())
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        # print("layout margins", layout.contentsMargins())
        # print("layout spacing", layout.spacing())
        # layout.setSpacing(0)
        self.setLayout(layout)
        self.edit = QLineEdit()
        fm = self.edit.fontMetrics()
        w = 7*fm.width('0')
        self.edit.setFixedWidth(w)
        self.edit.editingFinished.connect(self.onEditingFinished)
        self.edit.textEdited.connect(self.onTextEdited)
        self.setToVoxelSize()
        layout.addWidget(self.edit)
        label = QLabel("Voxel size in μm")
        layout.addWidget(label)
        layout.addStretch()

    def setToVoxelSize(self):
        voxel_size_um = self.main_window.getVoxelSizeUm()
        txt = self.floatToText(voxel_size_um)
        self.edit.setText(txt)
        self.onTextEdited(txt)

    def floatToText(self, value):
        return "%.3f"%value

    def onEditingFinished(self):
        txt = self.edit.text()
        valid, size = self.parseText(txt)
        # print("oef", valid, size)
        if valid:
            self.main_window.setVoxelSizeUm(size)

    def parseText(self, txt):
        valid = True
        f = 0
        try:
            f = float(txt)
        except:
            valid = False
        if f <= 0:
            valid = False
        return valid, f

    def onTextEdited(self, txt):
        valid, f = self.parseText(txt)
        # print("ote", valid)
        if valid:
            self.edit.setStyleSheet("")
        else:
            self.edit.setStyleSheet("QLineEdit { color: red }")

class ShiftClicksSpinBox(QSpinBox):
    def __init__(self, main_window, parent=None):
        super(ShiftClicksSpinBox, self).__init__(parent)
        self.main_window = main_window
        self.name = "shift_clicks"
        self.class_name = "count"
        self.setMinimum(0)
        self.setMaximum(2)
        self.setValue(main_window.draw_settings[self.name][self.class_name])
        self.valueChanged.connect(self.onValueChanged, Qt.QueuedConnection)
        main_window.draw_settings_widgets[self.name][self.class_name] = self

    def onValueChanged(self, value):
        self.main_window.setShiftClicksCount(value)
        self.lineEdit().deselect()

    def updateValue(self, value):
        self.setValue(value)

class WidthSpinBox(QSpinBox):
    def __init__(self, main_window, name, parent=None):
        super(WidthSpinBox, self).__init__(parent)
        self.main_window = main_window
        self.name = name
        self.class_name = "width"
        self.setMinimum(0)
        self.setMaximum(10)
        self.setValue(main_window.draw_settings[self.name][self.class_name])
        self.valueChanged.connect(self.onValueChanged, Qt.QueuedConnection)
        main_window.draw_settings_widgets[self.name][self.class_name] = self

    def onValueChanged(self, value):
        self.main_window.setDrawSettingsValue(self.name, self.class_name, value)
        self.lineEdit().deselect()

    def updateValue(self, value):
        self.setValue(value)

class OpacitySpinBox(QDoubleSpinBox):
    def __init__(self, main_window, name, parent=None):
        super(OpacitySpinBox, self).__init__(parent)
        self.main_window = main_window
        self.name = name
        self.class_name = "opacity"
        self.setMinimum(0.0)
        self.setMaximum(1.0)
        self.setDecimals(1)
        self.setSingleStep(0.1)
        self.setValue(main_window.draw_settings[name][self.class_name])
        self.valueChanged.connect(self.onValueChanged, Qt.QueuedConnection)
        main_window.draw_settings_widgets[self.name][self.class_name] = self

    def onValueChanged(self, value):
        rvalue = round(value*10)/10.
        if rvalue != value:
            # print("rvalue",rvalue,"!=","value",value)
            self.setValue(rvalue)
        self.main_window.setDrawSettingsValue(self.name, self.class_name, rvalue)
        self.lineEdit().deselect()

    def updateValue(self, value):
        self.setValue(value)

class ApplyOpacityCheckBox(QCheckBox):
    def __init__(self, main_window, name, enabled, parent=None):
        super(ApplyOpacityCheckBox, self).__init__(parent)
        self.main_window = main_window
        self.name = name
        self.class_name = "apply_opacity"
        self.setChecked(main_window.draw_settings[self.name][self.class_name])
        self.setEnabled(enabled)
        self.stateChanged.connect(self.onStateChanged)
        main_window.draw_settings_widgets[self.name][self.class_name] = self

    def onStateChanged(self, s):
        self.main_window.setDrawSettingsValue(self.name, self.class_name, s==Qt.Checked)

    def updateValue(self, value):
        self.setChecked(value)


    '''
    # class function
    def widthSpinBoxFactory(parent):
        sb = QSpinBox(parent)
        sb.minimum = 0
        sb.maximum = 10
        return sb

    # class function
    def opacitySpinBoxFactory(parent):
        sb = QDoubleSpinBox(parent)
        sb.minimum = 0.
        sb.maximum = 1.0
        sb.decimals = 1
        sb.singleStep = 0.1
        return sb
    '''


class MainWindow(QMainWindow):

    appname = "χάρτης"

    draw_settings_defaults = {
        "node": {
            "width": 5,
            "opacity": 1.0,
            "apply_opacity": True,
        },
        "line": {
            "width": 1,
            "opacity": 1.0,
            "apply_opacity": True,
        },
        "mesh": {
            "width": 1,
            "opacity": 1.0,
            "apply_opacity": True,
        },
        "axes": {
            "width": 2,
            "opacity": 1.0,
            "apply_opacity": True,
        },
        "borders": {
            "width": 5,
            "opacity": 1.0,
            "apply_opacity": True,
        },
        "labels": {
            "width": 1,
            "opacity": 1.0,
            "apply_opacity": False,
        },
        "overlay": {
            "opacity": 1.0,
            "apply_opacity": True,
        },
        "tracking_cursors": {
            "show": False,
        },
        "shift_clicks": {
            "count": 1,
        },
    }

    def __init__(self, appname, app):
        super(MainWindow, self).__init__()

        self.app = app
        self.settings = QSettings(QSettings.IniFormat, QSettings.UserScope, 'khartes.org', 'khartes')
        print("Loaded settings from", self.settings.fileName())
        qv = [int(x) for x in qVersion().split('.')]
        # print("Qt version", qv)
        if qv[0] > 5 or qv[0] < 5 or qv[1] < 12:
            print("Need to use Qt version 5, subversion 12 or above")
            # 5.12 or above is needed for QImage::Format_RGBX64
            exit()

        self.live_zsurf_update = True

        self.setWindowTitle(MainWindow.appname)
        self.setMinimumSize(QSize(750,600))
        self.settingsApplySizePos()

        self.draw_settings = copy.deepcopy(MainWindow.draw_settings_defaults)
        self.settingsLoadDrawSettings()
        self.draw_settings_widgets = copy.deepcopy(MainWindow.draw_settings_defaults)

        # if False, shift lock only requires a single click
        # self.shift_lock_double_click = True

        grid = QGridLayout()

        self.project_view = None
        self.cursor_tijk = None
        self.cursor_window = None
        args = QCoreApplication.arguments()
        path = os.path.dirname(os.path.realpath(args[0]))
        # https://iconduck.com/icons/163625/openhand
        px = QPixmap(path+"/openhand.svg")
        # print("px size",px.size())
        self.cursor_center = (16, 8)
        self.openhand = QCursor(px, *self.cursor_center)
        px = QPixmap(path+"/openhand transparent.svg")
        # print("px size",px.size())
        self.openhand_transparent = QCursor(px, *self.cursor_center)
        self.openhand_transparents = self.transparentSvgs(path+"/openhand transparent.svg", 11)
        self.openhand_transparent = self.openhand_transparents[0]

        # x slice or y slice in data
        self.depth = DataWindow(self, 2)

        # z slice in data
        self.inline = DataWindow(self, 0)

        # y slice or x slice in data
        self.xline = DataWindow(self, 1)

        # slice of data from interpreted surface
        self.surface = SurfaceWindow(self)

        # GUI panel
        self.tab_panel = QTabWidget()
        self.tab_panel.setMinimumSize(QSize(200,200))

        grid.addWidget(self.xline, 0, 0, 2, 2)
        grid.addWidget(self.inline, 2, 0, 2, 2)
        grid.addWidget(self.depth, 4, 0, 2, 2)
        grid.addWidget(self.surface, 0, 2, 4, 3)
        grid.addWidget(self.tab_panel, 4, 2, 2, 3)

        # self.edit = QPlainTextEdit(self)
        self.edit = QTextEdit(self)
        self.edit.setReadOnly(True)
        # self.edit.setMaximumSize(QSize(200,300))
        grid.addWidget(self.edit, 0, 2, 4, 3)

        self.addVolumesPanel()
        self.addFragmentsPanel()
        self.addSettingsPanel()
        self.addDevToolsPanel()

        widget = QWidget()
        widget.setLayout(grid)
        self.setCentralWidget(widget)

        self.load_hardwired_project_action = QAction("Load hardwired project", self)
        self.load_hardwired_project_action.triggered.connect(self.onLoadHardwiredProjectButtonClick)
        self.new_project_action = QAction("New project...", self)
        self.new_project_action.triggered.connect(self.onNewProjectButtonClick)

        self.open_project_action = QAction("Open project...", self)
        self.open_project_action.triggered.connect(self.onOpenProjectButtonClick)

        self.save_project_action = QAction("Save project", self)
        self.save_project_action.triggered.connect(self.onSaveProjectButtonClick)
        self.save_project_action.setEnabled(False)

        self.save_project_as_action = QAction("Save project as...", self)
        self.save_project_as_action.triggered.connect(self.onSaveProjectAsButtonClick)
        self.save_project_as_action.setEnabled(False)

        self.import_nrrd_action = QAction("Import NRRD files...", self)
        self.import_nrrd_action.triggered.connect(self.onImportNRRDButtonClick)
        self.import_nrrd_action.setEnabled(False)

        self.import_tiffs_action = QAction("Create volume from TIFF files...", self)
        self.import_tiffs_action.triggered.connect(self.onImportTiffsButtonClick)
        self.import_tiffs_action.setEnabled(False)

        self.export_mesh_action = QAction("Export fragment as mesh...", self)
        self.export_mesh_action.triggered.connect(self.onExportAsMeshButtonClick)
        self.export_mesh_action.setEnabled(False)

        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.onExitButtonClick)

        # Qt trickery to put menu bar and tool bar on same line
        self.menu_toolbar = self.addToolBar("Menu")
        self.menu_toolbar.setFloatable(False)
        self.menu_toolbar.setMovable(False)
        # self.menu = self.menuBar()
        self.menu = QMenuBar()
        self.menu.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.menu_toolbar.addWidget(self.menu)
        self.file_menu = self.menu.addMenu("&File")
        self.file_menu.addAction(self.open_project_action)
        self.file_menu.addAction(self.new_project_action)
        self.file_menu.addAction(self.save_project_action)
        self.file_menu.addAction(self.save_project_as_action)
        self.file_menu.addAction(self.import_nrrd_action)
        self.file_menu.addAction(self.import_tiffs_action)
        self.file_menu.addAction(self.export_mesh_action)
        # self.file_menu.addAction(self.load_hardwired_project_action)
        self.file_menu.addAction(self.exit_action)

        # put space between end of menu bar and start of tool bar
        sep = QAction(" ", self)
        self.menu.addAction(sep)
        sep.setDisabled(True)

        self.toolbar = self.addToolBar("Tools")

        self.add_node_mode = False

        self.add_node_mode_button = CursorModeButton(self)
        self.toolbar.addWidget(self.add_node_mode_button)
        self.last_shift_time = 0

        self.live_zsurf_update_button = LiveZsurfUpdateButton(self)
        self.toolbar.addWidget(self.live_zsurf_update_button)

        self.toggle_direction_action = QAction("Toggle direction", self)
        self.toggle_direction_action.triggered.connect(self.onToggleDirectionButtonClick)
        self.next_volume_action = QAction("Next volume", self)
        self.next_volume_action.triggered.connect(self.onNextVolumeButtonClick)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        # is this needed?
        self.volumes_model = VolumesModel(None, self)
        self.tiff_loader = TiffLoader(self)
        # self.setDrawSettingsToDefaults()
        # command line arguments
        args = QCoreApplication.arguments()
        # print("command line arguments", args)
        if len(args) > 1 and args[1].endswith('.khprj'):
            self.loadProject(args[1])
            QTimer.singleShot(100, self.drawSlices)

    def setCursorPosition(self, data_window, tijk):
        # show_tracking_cursors = self.draw_settings["tracking_cursors"]["show"]
        # TODO remove after testing
        # show_tracking_cursors = True
        # if not show_tracking_cursors:
        if not self.getTrackingCursorsVisible():
            self.cursor_tijk = None
            self.cursor_window = None
            return
        self.cursor_tijk = tijk
        self.cursor_window = data_window
        self.drawSlices()

    # loosely based on https://stackoverflow.com/questions/15123544/change-the-color-of-an-svg-in-qt
    def transparentSvgs(self, fname, cnt):
        try:
            fd = open(fname, "r")
        except:
            print("could not open",fname)
            return []
        svg_txt = fd.read()
        doc = QDomDocument()
        doc.setContent(svg_txt)
        paths = doc.elementsByTagName("path")
        # print(len(paths),"paths")
        gradients = [
                {
                    "fill-opacity": (.2, 0.),
                    # "stroke-opacity": (0., 1.)
                    "stroke-opacity": (0., .4)
                    },
                {
                    "fill-opacity": (1., 0.),
                    # "stroke-opacity": (0., 1.)
                    "stroke-opacity": (0., .4)
                    },
                {
                    # "fill-opacity": (1., 0.2),
                    "fill-opacity": (1., 0.0),
                    # "stroke-opacity": (0., 0.6)
                    "stroke-opacity": (0., 0.4)
                    }
                ]
        if len(gradients) != len(paths):
            print("gradient-path mismatch")
            return []

        delta = 1./(cnt-1)

        cursors = []
        for i in range(0,cnt+1):
            a = i*delta
            # for j,path in enumerate(paths):
            for j in range(paths.length()):
                path = paths.at(j).toElement()

                gradient = gradients[j]
                for attr, x01 in gradient.items():
                    x0 = x01[0]
                    x1 = x01[1]
                    v = (1.-a)*x0 + x1
                    strv = "%.3f"%v
                    # print(v,strv)
                    path.setAttribute(attr, strv)
            # if i == 5:
            #     print(doc.toString(4))
            rend = QSvgRenderer(doc.toByteArray())
            pix = QPixmap(rend.defaultSize())
            pix.fill(Qt.transparent)
            painter = QPainter(pix)
            rend.render(painter)
            painter.end()
            cursors.append(QCursor(pix, *self.cursor_center))

        return cursors

    def addFragmentsPanel(self):
        panel = QWidget()
        vlayout = QVBoxLayout()
        panel.setLayout(vlayout)
        hlayout = QHBoxLayout()
        label = QLabel("Hover mouse over column headings for more information")
        label.setAlignment(Qt.AlignCenter)
        hlayout.addWidget(label)
        create_frag = CreateFragmentButton(self)
        create_frag.setStyleSheet("QPushButton { background-color : beige; padding: 5; }")
        hlayout.addWidget(create_frag)
        label = QLabel("Active fragment:")
        # label.setStyleSheet("QLabel { background-color : beige; padding-left: 5}")
        label.setStyleSheet("QLabel { padding-left: 5}")
        hlayout.addWidget(label)
        # active_frame = QGroupBox("Actions on active fragment")
        # af_layout = QHBoxLayout()
        # active_frame.setLayout(af_layout)
        self.copy_frag = CopyActiveFragmentButton(self)
        self.copy_frag.setStyleSheet("QPushButton { background-color : beige; padding: 5; }")
        self.copy_frag.setEnabled(False)
        hlayout.addWidget(self.copy_frag)
        self.move_frag_up = MoveActiveFragmentButton(self, "Z ↑", -1)
        self.move_frag_up.setStyleSheet("QPushButton { background-color : beige; padding: 5; }")
        self.move_frag_up.setEnabled(False)
        hlayout.addWidget(self.move_frag_up)
        self.move_frag_down = MoveActiveFragmentButton(self, "Z ↓", 1)
        self.move_frag_down.setStyleSheet("QPushButton { background-color : beige; padding: 5; }")
        self.move_frag_down.setEnabled(False)
        hlayout.addWidget(self.move_frag_down)
        # af_layout.addWidget(self.copy_frag)
        # hlayout.addWidget(active_frame)
        hlayout.addStretch()
        vlayout.addLayout(hlayout)
        self.fragments_table = QTableView()
        hh = self.fragments_table.horizontalHeader()
        # print("mss", hh.minimumSectionSize())
        # hh.setMinimumSectionSize(20)
        hh.setMinimumSectionSize(30)
        # hh.setMinimumSectionSize(40)
        fragments_csd = ColorSelectorDelegate(self.fragments_table)
        # self.direction_selector_delegate = dsd
        # need to attach these to "self" so they don't
        # get deleted on going out of scope
        self.fragments_csd = fragments_csd
        self.fragments_table.setItemDelegateForColumn(3, fragments_csd)
        # print("edit triggers", int(self.volumes_table.editTriggers()))
        # self.volumes_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        # print("mss", hh.minimumSectionSize())

        self.fragments_table.setModel(FragmentsModel(None, self))
        self.fragments_table.resizeColumnsToContents()
        vlayout.addWidget(self.fragments_table)
        self.tab_panel.addTab(panel, "Fragments")

    def addDevToolsPanel(self):
        panel = QWidget()
        vlayout = QVBoxLayout()
        panel.setLayout(vlayout)
        ivlayout = QVBoxLayout()
        label = QLabel("Warning: Developer tools; use at your own risk!")
        label.setStyleSheet("QLabel { font-weight: bold; color: white ; background-color: red}")
        label.setAlignment(Qt.AlignCenter)
        # TODO set style (white letters on dark red background)
        ivlayout.addWidget(label)
        label = QLabel("Settings will not be retained after you exit")
        ivlayout.addWidget(label)
        vlayout.addLayout(ivlayout)
        # hlayout = QHBoxLayout()
        # label = "Z interpolation:"
        interp = ZInterpolationSetter(self)
        vlayout.addWidget(interp)
        roundness = RoundnessSetter(self)
        roundness.setHideSkinnyTriangles(False)
        roundness.setMinRoundness(.5)
        vlayout.addWidget(roundness)
        vlayout.addStretch()
        self.tab_panel.addTab(panel, "Dev Tools")

    def setLiveZsurfUpdate(self, lzu):
        if lzu == self.live_zsurf_update:
            return
        self.live_zsurf_update = lzu
        pv = self.project_view
        if pv is not None:
            for fv in pv.fragments.values():
                fv.setLiveZsurfUpdate(lzu)
        self.drawSlices()

    def setZInterpolation(self, index):
        linear = True
        if index != 0:
            linear = False
        if FragmentView.use_linear_interpolation == linear:
            return
        FragmentView.use_linear_interpolation = linear
        if self.project_view is not None:
            self.project_view.updateFragmentViews()
            self.drawSlices()

    def setHideSkinnyTriangles(self, state):
        # print("shst", state)
        if state == FragmentView.hide_skinny_triangles:
            return
        FragmentView.hide_skinny_triangles = state
        if self.project_view is not None:
            self.project_view.updateFragmentViews()
            self.drawSlices()

    def setMinRoundness(self, value):
        # print("smr", value)
        if value == Fragment.min_roundness:
            return
        Fragment.min_roundness = value
        if self.project_view is not None:
            self.project_view.updateFragmentViews()
            self.drawSlices()

    def addVolumesPanel(self):
        panel = QWidget()
        vlayout = QVBoxLayout()
        panel.setLayout(vlayout)
        hlayout = QHBoxLayout()
        label = QLabel("Hover mouse over column headings for more information")
        label.setAlignment(Qt.AlignCenter)
        hlayout.addWidget(label)
        vbv = VolBoxesVisibleCheckBox(self)
        self.settings_vol_boxes_visible2 = vbv
        # vbv.setAutoFillBackground(True)
        # palette = vbv.palette()
        # palette.setColor(QPalette.Window, QColor("green"))
        vbv.setStyleSheet("QCheckBox { background-color : beige; padding: 5; }")
        # vbv.setPalette(palette)
        hlayout.addWidget(vbv)
        hlayout.addStretch()
        vlayout.addLayout(hlayout)
        self.volumes_table = QTableView()
        hh = self.volumes_table.horizontalHeader()
        # print("mss", hh.minimumSectionSize())
        # hh.setMinimumSectionSize(20)
        hh.setMinimumSectionSize(30)
        # hh.setMinimumSectionSize(40)
        volumes_dsd = DirectionSelectorDelegate(self.volumes_table)
        volumes_csd = ColorSelectorDelegate(self.volumes_table)
        # self.direction_selector_delegate = dsd
        # need to attach these to "self" so they don't
        # get deleted on going out of scope
        self.volumes_csd = volumes_csd
        self.volumes_dsd = volumes_dsd
        self.volumes_table.setItemDelegateForColumn(2, volumes_csd)
        self.volumes_table.setItemDelegateForColumn(4, volumes_dsd)
        # print("edit triggers", int(self.volumes_table.editTriggers()))
        # self.volumes_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        # print("mss", hh.minimumSectionSize())

        self.volumes_table.setModel(VolumesModel(None, self))
        self.volumes_table.resizeColumnsToContents()
        # grid.addWidget(self.volumes_table, 1,0,1,1)
        vlayout.addWidget(self.volumes_table)
        self.tab_panel.addTab(panel, "Volumes")

    def setDrawSettingsToDefaults(self):
        ndict = {}
        for key,value in self.draw_settings_defaults.items():
            if key in ["tracking_cursors", "shift_clicks"]:
                continue
            self.draw_settings[key] = copy.deepcopy(value)
        # self.draw_settings = copy.deepcopy(MainWindow.draw_settings_defaults)
        self.updateDrawSettingsWidgets()
        self.settingsSaveDrawSettings()
        self.drawSlices()

    def setDrawSettingsValue(self, obj_type, val_type, value):
        self.draw_settings[obj_type][val_type] = value
        self.settingsSaveDrawSettings()
        self.drawSlices()

    def updateDrawSettingsWidgets(self):
        for class_name, class_dict in self.draw_settings.items():
            for name, value in class_dict.items():
                dc = self.draw_settings_widgets.get(class_name, None)
                if dc is None: 
                    continue
                widget = dc.get(name, None)
                if not isinstance(widget, QWidget):
                    continue
                widget.updateValue(value)


    def addSettingsPanel(self):
        panel = QWidget()
        hlayout = QHBoxLayout()
        panel.setLayout(hlayout)
        draw_settings_vbox = QVBoxLayout()
        hlayout.addLayout(draw_settings_vbox)
        draw_settings_frame = QGroupBox("Display settings")
        # draw_settings_frame.setFlat(True)
        draw_settings_vbox.addWidget(draw_settings_frame)
        draw_settings_vbox.addStretch()
        draw_settings_layout = QGridLayout()
        draw_settings_layout.setContentsMargins(2,6,2,2)
        draw_settings_layout.setVerticalSpacing(1)
        draw_settings_frame.setLayout(draw_settings_layout)
        dsl = draw_settings_layout
        tr = 0
        dsl.addWidget(QLabel("Opacity"), tr, 1)
        dsl.addWidget(OpacitySpinBox(self, "overlay"), tr, 2)
        tr += 1
        # hline = QFrame()
        # hline.setFrameShape(QFrame.HLine)
        # hline.setFrameShadow(QFrame.Sunken)
        # dsl.addWidget(hline, tr, 0, 1, 3)
        # dsl.addWidget(QLabel(" "), tr, 0)
        dsl.addItem(QSpacerItem(1,5), tr, 0)
        tr += 1
        cba = Qt.AlignRight
        dsl.addWidget(QLabel("Size"), tr, 1)
        dsl.addWidget(QLabel("Apply\nOpacity"), tr, 2, Qt.AlignRight)
        # dsl.addWidget(QLabel("Apply"), 1, tr)
        tr += 1
        dsl.addWidget(QLabel("Node"), tr, 0)
        dsl.addWidget(WidthSpinBox(self, "node"), tr, 1)
        # dsl.addWidget(OpacitySpinBox(self, "node"), tr, 2)
        # dsl.addWidget(OpacitySpinBox(self, "overlay"), tr, 2, 6, 1)
        dsl.addWidget(ApplyOpacityCheckBox(self, "node", True), tr, 2, cba)
        tr += 1
        dsl.addWidget(QLabel("Line"), tr, 0)
        dsl.addWidget(WidthSpinBox(self, "line"), tr, 1)
        # dsl.addWidget(OpacitySpinBox(self, "line"), tr, 2)
        dsl.addWidget(ApplyOpacityCheckBox(self, "line", True), tr, 2, cba)
        tr += 1
        dsl.addWidget(QLabel("Mesh"), tr, 0)
        dsl.addWidget(WidthSpinBox(self, "mesh"), tr, 1)
        # dsl.addWidget(OpacitySpinBox(self, "mesh"), tr, 2)
        dsl.addWidget(ApplyOpacityCheckBox(self, "mesh", True), tr, 2, cba)
        tr += 1
        dsl.addWidget(QLabel("Axes"), tr, 0)
        dsl.addWidget(WidthSpinBox(self, "axes"), tr, 1)
        # dsl.addWidget(OpacitySpinBox(self, "axes"), tr, 2)
        dsl.addWidget(ApplyOpacityCheckBox(self, "axes", True), tr, 2, cba)
        tr += 1
        dsl.addWidget(QLabel("Borders"), tr, 0)
        bwsb = WidthSpinBox(self, "borders")
        bwsb.setMinimum(1)
        bwsb.setMaximum(11)
        bwsb.setSingleStep(2)
        dsl.addWidget(bwsb, tr, 1)
        dsl.addWidget(ApplyOpacityCheckBox(self, "borders", True), tr, 2, cba)
        tr += 1
        dsl.addWidget(QLabel("Labels"), tr, 0)
        lwsb = WidthSpinBox(self, "labels")
        lwsb.setMaximum(1)
        # lwsb.setEnabled(False)
        dsl.addWidget(lwsb, tr, 1)
        dsl.addWidget(ApplyOpacityCheckBox(self, "labels", True), tr, 2, cba)
        tr += 1
        dsl.addItem(QSpacerItem(1,5), tr, 0)
        tr += 1
        dpb = QPushButton("Restore defaults")
        dsl.addWidget(dpb, tr, 0, 1, 3)
        dpb.clicked.connect(lambda s: self.setDrawSettingsToDefaults())

        # sb.valueChanged.connect(lambda d: self.onValueChanged(d, sb, index))
        # self.clicked.connect(self.onButtonClicked)

        slices_vbox = QVBoxLayout()
        hlayout.addLayout(slices_vbox)
        slices_frame = QGroupBox("Other settings")
        slices_vbox.addWidget(slices_frame)
        slices_vbox.addStretch()
        slices_layout = QVBoxLayout()
        slices_frame.setLayout(slices_layout)
        vbv = VolBoxesVisibleCheckBox(self)
        self.settings_vol_boxes_visible = vbv
        slices_layout.addWidget(vbv)
        tcv = TrackingCursorsVisibleCheckBox(self)
        self.settings_tracking_cursors_visible = tcv
        slices_layout.addWidget(tcv)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Shift-lock after"))
        scc = ShiftClicksSpinBox(self)
        self.settings_shift_clicks_count = scc
        hbox.addWidget(scc)
        hbox.addWidget(QLabel("clicks"))
        hbox.addStretch()
        slices_layout.addLayout(hbox)
        vs = VoxelSizeEditor(self)
        self.settings_voxel_size_um = vs
        slices_layout.addWidget(vs)

        hlayout.addStretch()
        # fragment_layout = QVBoxLayout()
        # fragment_layout.addWidget(QLabel("Fragment View"))

        self.tab_panel.addTab(panel, "Settings")

    # gijk is a numpy array
    def recenterCurrentVolume(self, gijk):
        if self.project_view is None:
            return
        vv = self.project_view.cur_volume_view
        if vv is None:
            return
        gijks = gijk.reshape((1,-1))
        # print("rcv g", gijk, gijks)
        tijk = vv.globalPositionsToTransposedIjks(gijks)
        # print("rcv t", tijk)
        vv.setIjkTf(tijk[0].tolist())
        self.drawSlices()

    def volumeView(self):
        return self.project_view.cur_volume_view

    def toggleTrackingCursorsVisible(self):
        vis = self.getTrackingCursorsVisible()
        self.setTrackingCursorsVisible(not vis)

    def getTrackingCursorsVisible(self):
        return self.draw_settings["tracking_cursors"]["show"]

    def setTrackingCursorsVisible(self, value):
        # slices = self.project_view.settings['slices']
        # vbv = 'vol_boxes_visible'
        # old_value = slices[vbv]
        old_value = self.getTrackingCursorsVisible()
        if old_value == value:
            return
        self.draw_settings["tracking_cursors"]["show"] = value
        self.settings_tracking_cursors_visible.setChecked(self.getTrackingCursorsVisible())
        # self.project_view.notifyModified()
        self.settingsSaveDrawSettings()
        self.drawSlices()

    def getShiftClicksCount(self):
        # print ("scc", self.draw_settings["shift_clicks"]["count"])
        return self.draw_settings["shift_clicks"]["count"]

    def setShiftClicksCount(self, value):
        old_value = self.getShiftClicksCount()
        if old_value == value:
            return
        self.draw_settings["shift_clicks"]["count"] = value
        self.settings_shift_clicks_count.setValue(self.getShiftClicksCount())
        # self.project_view.notifyModified()
        self.settingsSaveDrawSettings()
        # self.drawSlices()

    def getVolBoxesVisible(self):
        if self.project_view is None:
            return
        # slices = self.project_view.settings['slices']
        # vbv = 'vol_boxes_visible'
        # return slices[vbv]
        return self.project_view.vol_boxes_visible

    def setVolBoxesVisible(self, value):
        if self.project_view is None:
            return
        # slices = self.project_view.settings['slices']
        # vbv = 'vol_boxes_visible'
        # old_value = slices[vbv]
        old_value = self.project_view.vol_boxes_visible
        if old_value == value:
            return
        self.project_view.vol_boxes_visible = value
        # slices[vbv] = value
        self.settings_vol_boxes_visible.setChecked(self.getVolBoxesVisible())
        self.settings_vol_boxes_visible2.setChecked(self.getVolBoxesVisible())
        self.project_view.notifyModified()
        self.drawSlices()

    def onNewFragmentButtonClick(self, s):
        self.createFragment()

    def uniqueFragmentName(self, start):
        pv = self.project_view
        if pv is None:
            print("Warning, cannot find unique fragment names without a project")
            return None
        names = set()
        for frag in pv.fragments.keys():
            name = frag.name
            names.add(name)
        stem = "frag"
        # mfv = pv.mainActiveVisibleFragmentView(unaligned_ok=True)
        # if mfv is not None:
        #     stem = mfv.fragment.name
        if start is not None:
            stem = start
        if stem not in names:
            return stem
        for i in range(1,1000):
            # name = "%s%d"%(stem,i)
            name = Utils.nextName(stem, i)
            if name not in names:
                return name
        return None

    def enableWidgetsIfActiveFragment(self):
        pv = self.project_view
        active = False
        if pv is not None:
            # active = (len(pv.activeFragmentViews(unaligned_ok=True)) > 0)
            active = (pv.mainActiveFragmentView(unaligned_ok=True) is not None)
        self.export_mesh_action.setEnabled(active)
        self.copy_frag.setEnabled(active)
        self.move_frag_up.setEnabled(active)
        self.move_frag_down.setEnabled(active)

    def moveActiveFragment(self, step):
        pv = self.project_view
        if pv is None:
            print("Warning, cannot create new fragment without project")
            return
        mfv = pv.mainActiveFragmentView(unaligned_ok=True)
        if mfv is None:
            # this should never be reached; button should be
            # inactive in this case
            print("No currently active fragment")
            return
        # mf = mfv.fragment
        mfv.moveInK(step)
        self.drawSlices()

    def copyActiveFragment(self):
        pv = self.project_view
        if pv is None:
            print("Warning, cannot create new fragment without project")
            return
        # vv = self.volumeView()
        # if vv is None:
        #     print("Warning, cannot create new fragment without volume view set")
        #     return
        mfv = pv.mainActiveFragmentView(unaligned_ok=True)
        if mfv is None:
            # this should never be reached; Copy button should be
            # inactive in this case
            print("No currently active fragment")
            return
        mf = mfv.fragment
        stem = mf.name+"-copy"
        name = self.uniqueFragmentName(stem)
        if name is None:
            print("Can't create unique fragment name from", stem)
            return
        # frag = Fragment(name, mf.direction)
        # frag.setColor(mf.qcolor, no_notify=True)
        # frag.gpoints = np.copy(mf.gpoints)
        frag = mf.createCopy(name)
        print("created fragment %s from %s"%(frag.name, mf.name))
        self.fragments_table.model().beginResetModel()
        pv.project.addFragment(frag)
        self.setFragments()
        self.fragments_table.model().endResetModel()
        exclusive = (len(pv.activeFragmentViews(unaligned_ok=True)) == 1)
        self.setFragmentActive(frag, True, exclusive)
        self.enableWidgetsIfActiveFragment()
        # need to make sure new fragment is added to table
        # before calling scrollToRow
        self.app.processEvents()
        index = pv.project.fragments.index(frag)
        self.fragments_table.model().scrollToRow(index)

    def createFragment(self):
        pv = self.project_view
        if pv is None:
            print("Warning, cannot create new fragment without project")
            return
        vv = self.volumeView()
        if vv is None:
            print("Warning, cannot create new fragment without volume view set")
            return
        '''
        names = set()
        for frag in pv.fragments.keys():
            name = frag.name
            names.add(name)
        stem = "frag"
        mfv = pv.mainActiveVisibleFragmentView(unaligned_ok=True)
        if mfv is not None:
            stem = mfv.fragment.name
        for i in range(1,1000):
            # name = "%s%d"%(stem,i)
            name = Utils.nextName(stem, i)
            if name not in names:
                break
        '''
        stem = "frag"
        mfv = pv.mainActiveVisibleFragmentView(unaligned_ok=True)
        if mfv is not None:
            stem = mfv.fragment.name
        name = self.uniqueFragmentName(stem)
        if name is None:
            print("Can't create unique fragment name from stem", stem)
            return
        # print("color",color)
        frag = Fragment(name, vv.direction)
        frag.setColor(Utils.getNextColor(), no_notify=True)
        frag.valid = True
        print("created fragment %s"%frag.name)
        self.fragments_table.model().beginResetModel()
        # print("start cafv")
        # if len(pv.activeFragmentViews(unaligned_ok=True)) == 1:
        #     pv.clearActiveFragmentViews()
        # print("end cafv")
        pv.project.addFragment(frag)
        self.setFragments()
        self.fragments_table.model().endResetModel()
        # fv = pv.fragments[frag]
        # fv.active = True
        # self.export_mesh_action.setEnabled(len(pv.activeFragmentViews(unaligned_ok=True)) > 0)
        exclusive = (len(pv.activeFragmentViews(unaligned_ok=True)) == 1)
        self.setFragmentActive(frag, True, exclusive)
        self.enableWidgetsIfActiveFragment()
        # need to make sure new fragment is added to table
        # before calling scrollToRow
        self.app.processEvents()
        index = pv.project.fragments.index(frag)
        self.fragments_table.model().scrollToRow(index)

    def renameFragment(self, frag, name):
        if frag.name == name:
            return
        self.fragments_table.model().beginResetModel()
        frag.name = name
        frag.notifyModified()
        proj = self.project_view.project
        proj.alphabetizeFragments()
        self.fragments_table.model().endResetModel()
        self.app.processEvents()
        index = proj.fragments.index(frag)
        self.fragments_table.model().scrollToRow(index)
        self.drawSlices()

    def movePoint(self, fragment_view, index, new_tijk):
        self.fragments_table.model().beginResetModel()
        result = fragment_view.movePoint(index, new_tijk)
        self.fragments_table.model().endResetModel()
        return result

    def addPointToCurrentFragment(self, tijk):
        cur_frag_view = self.project_view.mainActiveVisibleFragmentView()
        if cur_frag_view is None:
            print("no current fragment view set")
            return
        self.fragments_table.model().beginResetModel()
        cur_frag_view.addPoint(tijk)
        self.fragments_table.model().endResetModel()

    def deleteNearbyNode(self):
        pv = self.project_view
        if pv.nearby_node_fv is None or pv.nearby_node_index < 0:
            print("deleteNearbyNode: no current nearby node")
            return
        self.fragments_table.model().beginResetModel()
        pv.nearby_node_fv.deletePointByIndex(pv.nearby_node_index)
        pv.nearby_node_fv = None
        pv.nearby_node_index = -1
        self.fragments_table.model().endResetModel()


    def onNextCurrentFragmentButtonClick(self, s):
        cfv = self.project_view.cur_fragment_view
        cvv = self.project_view.cur_volume_view
        if cvv is None:
            print("Warning, cannot select cur fragment without volume view set")
            return
        vdir = cvv.direction
        valid_fvs = []
        for fv in self.project_view.fragments.values():
            if fv.fragment.direction == vdir:
                valid_fvs.append(fv)
        lv = len(valid_fvs)
        if lv == 0:
            print("Warning, no fragments are eligible to be cur fragment in this direction")
            return
        if cfv is None:
            next_fv = valid_fvs[0]
        else:
            for i,fv in enumerate(valid_fvs):
                next_fv = valid_fvs[(i+1)%lv]
                if fv == cfv:
                    break

        next_f = next_fv.fragment
        # print("setting %s as cur fragment"%next_f.name)
        # I'm a bit inconsistent about whether a function
        # like self.setCurrentFragment should assume that
        # the project_view has already been set, or whether
        # the function should call the project_view's setter
        # self.project_view.setCurrentFragment(next_f)
        self.setCurrentFragment(next_f)
        self.drawSlices()

    def onSaveProjectButtonClick(self, s):
        print("save project clicked")
        if self.project_view is None or self.project_view.project is None:
            print("No project currently loaded")
            return
        print("calling project save")
        # self.project_view.project.save()
        self.project_view.save()
        self.projectModifiedCallback(self.project_view.project)

    # Qt trickery to get around a problem with QFileDialog
    # when the user double-clicks on
    # a directory.  If the directory name ends with ".khprj",
    # we want that directory name to be immediately sent back to
    # the dialog's caller, rather than having the QFileDialog 
    # descend into the selected directory
    def onDirectoryEntered(self, sdir, dialog):
        # print("directory entered", sdir)
        if sdir.endswith(".khprj"):
            # print("match!")
            # dialog.done(sdir)
            dialog.khartes_directory = sdir
            dialog.done(1)

    # override
    def closeEvent(self, e):
        # print("close event")
        e.ignore()
        if not self.warnIfNotSaved("exit khartes"):
            # print("Canceled by user after warning")
            return
        # if self.tiff_loader is not None:
        #     self.tiff_loader.close()
        e.accept()

    # returns True if ok to continue, False if not ok
    def warnIfNotSaved(self, astr):
        if self.project_view is None:
            return True
        project = self.project_view.project
        uptodate = project.isSaveUpToDate()
        if uptodate:
            return True
        print("Project has been modified since last save")
        answer = QMessageBox.warning(self, 
        "khartes", 
        'The current project has been modified since it was last saved.\nDo you want to %s anyway?\nSelect "Ok" to %s; select "Cancel" to cancel the operation, giving you a chance to save the current project.'%(astr,astr), 
        QMessageBox.Ok|QMessageBox.Cancel, 
        QMessageBox.Cancel)
        if answer != QMessageBox.Ok:
            print("Operation to %s cancelled by user"%astr)
            return False
        print("Operation to %s accepted by user"%astr)
        return True

    def onNewProjectButtonClick(self, s):
        print("new project clicked")
        if not self.warnIfNotSaved("create a new project"):
            # print("Canceled by user after warning")
            return
        dialog = QFileDialog(self)
        sdir = self.settingsGetDirectory()
        if sdir is not None:
            print("setting directory to", sdir)
            dialog.setDirectory(sdir)
        # dialog.setOptions(QFileDialog.ShowDirsOnly)
        dialog.setOptions(QFileDialog.ShowDirsOnly|QFileDialog.DontUseNativeDialog)
        # dialog.setOptions(QFileDialog.DontUseNativeDialog)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter("Khartes project directories (*.khprj)")
        dialog.setLabelText(QFileDialog.Accept, "Create new project")
        dialog.directoryEntered.connect(lambda d: self.onDirectoryEntered(d, dialog))
        if not dialog.exec():
            print("No khprj directory selected")
            return

        file_names = dialog.selectedFiles()
        khartes_directory = getattr(dialog, "khartes_directory", None)
        # dialog.close()
        # self.app.processEvents()

        if khartes_directory is not None:
            file_names = [khartes_directory]
        print(file_names)
        if len(file_names) < 1:
            print("The khprj directory list is empty")
            return

        idir = file_names[0]

        pdir = Path(idir)
        name = pdir.name
        # Project.open does this check as well, but need
        # to do it here prior to checking whether the file
        # name already exists
        if not name.endswith(".khprj"):
            name += ".khprj"
        pdir = pdir.with_name(name)
        print("new project", pdir)
        if pdir.exists():
            answer = QMessageBox.warning(self, "khartes", "The project directory %s already exists.\nDo you want to overwrite it?"%str(pdir), QMessageBox.Ok|QMessageBox.Cancel, QMessageBox.Ok)
            if answer != QMessageBox.Ok:
                print("New project cancelled by user")
                return

        self.unsetProjectView()
        new_prj = Project.create(pdir)
        if not new_prj.valid:
            err = new_prj.error
            print("Failed to create new project: %s"%err)
            return

        path = pdir.absolute()
        parent = path.parent
        self.settingsSaveDirectory(str(parent))

        self.setWindowTitle("%s - %s"%(MainWindow.appname, pdir.name))
        self.setProject(new_prj)

    def projectModifiedCallback(self, project):
        uptodate = project.isSaveUpToDate()
        # print("uptodate", uptodate)
        self.save_project_action.setEnabled(not uptodate)

    def onSaveProjectAsButtonClick(self, s):
        print("save project as clicked")
        if self.project_view is None or self.project_view.project is None:
            print("No project currently loaded")
            return
        dialog = QFileDialog(self)
        sdir = self.settingsGetDirectory()
        if sdir is not None:
            print("setting directory to", sdir)
            dialog.setDirectory(sdir)
        # dialog.setOptions(QFileDialog.ShowDirsOnly|QFileDialog.DontUseNativeDialog)
        dialog.setOptions(QFileDialog.ShowDirsOnly)
        dialog.setFileMode(QFileDialog.AnyFile)
        # dialog.setFileMode(QFileDialog.Directory|QFileDialog.DontUseNativeDialog)
        dialog.setNameFilter("Khartes project directories (*.khprj)")
        dialog.setLabelText(QFileDialog.Accept, "Save as .khprj project")
        # dialog.filesSelected.connect(self.onFilesSelected)
        # dialog.directoryEntered.connect(self.onDirectoryEntered)
        # see comment at def of onDirectoryEntered
        dialog.directoryEntered.connect(lambda d: self.onDirectoryEntered(d, dialog))
        if not dialog.exec():
            print("No khprj directory selected")
            return

        file_names = dialog.selectedFiles()
        khartes_directory = getattr(dialog, "khartes_directory", None)
        if khartes_directory is not None:
            file_names = [khartes_directory]
        print(file_names)
        if len(file_names) < 1:
            print("The khprj directory list is empty")
            return

        idir = file_names[0]
        print("save as", idir)
        pdir = Path(idir)
        if pdir.exists():
            answer = QMessageBox.warning(self, "Save project as...", "The project directory %s already exists.\nDo you want to overwrite it?"%idir, QMessageBox.Ok|QMessageBox.Cancel, QMessageBox.Ok)
            if answer != QMessageBox.Ok:
                print("Save as cancelled by user")
                return

        old_prj = self.project_view.project
        new_prj = Project.create(idir)
        if not new_prj.valid:
            err = new_prj.error
            print("Failed to create project to save to: %s"%err)
            msg = QMessageBox()
            msg.setWindowTitle("Save project as...")
            msg.setIcon(QMessageBox.Critical)
            msg.setText("'Save as' failed: %s"%err)
            msg.exec()
            return

        self.setWindowTitle("%s - %s"%(MainWindow.appname, Path(idir).name))
        old_vpath = old_prj.volumes_path
        new_vpath = new_prj.volumes_path
        loading = self.showLoading("Copying volumes...")
        for nrrd in old_vpath.glob("*.nrrd"):
            print ("nrrd:", nrrd.name)
            shutil.copy2(nrrd, new_vpath / nrrd.name)

        old_prj.path = new_prj.path
        old_prj.volumes_path = new_prj.volumes_path
        old_prj.fragments_path = new_prj.fragments_path

        self.project_view.save()
        self.projectModifiedCallback(old_prj)

        path = pdir.absolute()
        parent = path.parent
        self.settingsSaveDirectory(str(parent))

    def settingsSaveDrawSettings(self):
        self.settings.beginGroup("MainWindow")
        jstr = json.dumps(self.draw_settings)
        self.settings.setValue("drawSettings", jstr)
        self.settings.endGroup()

    def settingsLoadDrawSettings(self):
        self.settings.beginGroup("MainWindow")
        jstr = self.settings.value("drawSettings", None)
        if jstr is not None:
            try:
                settings = json.loads(jstr)
                # merge into current settings, in case
                # some are missing from the json file
                # self.draw_settings = settings
                # print("json settings", settings)
                # print("draw settings", self.draw_settings)
                self.draw_settings = Utils.updateDict(self.draw_settings, settings)
                # print("updated draw settings", self.draw_settings)
                # No need; settingsLoadDrawSettings is
                # called before widgets are created
                # self.updateDrawSettingsWidgets()
            except Exception as e:
                print("Failed to parse drawSettings json string:", str(e))
        self.settings.endGroup()

    def settingsApplySizePos(self):
        self.settings.beginGroup("MainWindow")
        size = self.settings.value("size", None)
        if size is not None:
            # self.resize(size.toSize())
            self.resize(size)
        pos = self.settings.value("pos", None)
        if pos is not None:
            # self.resize(pos.toPoint())
            self.move(pos)
        self.settings.endGroup()

    def settingsSaveSizePos(self):
        self.settings.beginGroup("MainWindow")
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())
        self.settings.endGroup()

    def settingsGetDirectory(self, prefix=""):
        self.settings.beginGroup("MainWindow")
        sdir = self.settings.value(prefix+"directory", None)
        self.settings.endGroup()
        return sdir

    def settingsSaveDirectory(self, directory, prefix=""):
        self.settings.beginGroup("MainWindow")
        print("settings: %sdirectory %s"%(prefix, str(directory)))
        self.settings.setValue(prefix+"directory", str(directory))
        self.settings.endGroup()

    def onImportNRRDButtonClick(self, s):
        print("import nrrd clicked")
        if self.project_view is None or self.project_view.project is None:
            print("No project currently loaded")
            return
        sdir = self.settingsGetDirectory("nrrd_")
        if sdir is None:
            sdir = self.settingsGetDirectory()
        if sdir is None:
            sdir = ""

        files = QFileDialog.getOpenFileNames(self, "Select one or more files to import", sdir, "Volumes (*.nrrd)")
        if len(files) == 0:
            print("No files selected")
            return
        files = files[0]
        if len(files) == 0:
            print("No files selected")
            return
        print("nrrd files", files)
        project = self.project_view.project
        vpath = project.volumes_path
        loading = self.showLoading("Copying volumes...")
        for rnrrd in files:
            nrrd = Path(rnrrd)
            print ("nrrd:", nrrd.name)
            shutil.copy2(nrrd, vpath / nrrd.name)
            vol = Volume.loadNRRD(nrrd)
            if vol is not None and vol.valid:
                self.volumes_table.model().beginResetModel()
                project.addVolume(vol)
                self.volumes_table.model().endResetModel()
                # setVolume has its own calls to begin/end ResetModel
                self.setVolume(vol)
                self.volumes_table.resizeColumnsToContents()
            
        path = Path(files[0])
        path = path.absolute()
        parent = path.parent
        self.settingsSaveDirectory(str(parent), "nrrd_")

    def onExportAsMeshButtonClick(self, s):
        print("export mesh clicked")
        if self.project_view is None or self.project_view.project is None:
            print("No project currently loaded")
            return
        frags = []
        fvs = []
        for frag, fv in self.project_view.fragments.items():
            if fv.active:
                frags.append(frag)
                fvs.append(fv)
        if len(frags) == 0:
            print("No active fragment")
            return
        sdir = self.settingsGetDirectory("mesh_")
        if sdir is None:
            sdir = self.settingsGetDirectory()
        if sdir is None:
            sdir = ""

        filename_tuple = QFileDialog.getSaveFileName(self, "Save Fragment as Mesh", sdir, "Mesh *.obj")
        print("user selected", filename_tuple)
        # [0] is filename, [1] is the selector used
        filename = filename_tuple[0]
        if filename is None or filename == "":
            print("No file selected")
            return

        pname = Path(filename)

        name = pname.name

        dialog = InfillDialog(self)
        dialog.exec()
        infill = dialog.getValue()
        if infill < 0:
            print("Export mesh cancelled by user")
            return
        err = Fragment.saveListAsObjMesh(fvs, pname, infill)

        if err != "":
            msg = QMessageBox()
            msg.setWindowTitle("Save fragment as mesh")
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error: %s"%err)
            msg.exec()

        self.settingsSaveDirectory(str(pname.parent), "mesh_")


    def onImportTiffsButtonClick(self, s):
        self.tiff_loader.show()
        self.tiff_loader.raise_()

    # TODO: Need to alert user if load fails
    def onOpenProjectButtonClick(self, s):
        print("open project clicked")
        ''''''
        if not self.warnIfNotSaved("load a new project"):
            # print("Canceled by user after warning")
            return
        dialog = QFileDialog(self)
        sdir = self.settingsGetDirectory()
        if sdir is not None:
            print("setting directory to", sdir)
            dialog.setDirectory(sdir)

        # dialog.setOptions(QFileDialog.ShowDirsOnly|QFileDialog.DontUseNativeDialog)
        dialog.setOptions(QFileDialog.ShowDirsOnly)
        dialog.setFileMode(QFileDialog.Directory)
        # dialog.setFileMode(QFileDialog.Directory|QFileDialog.DontUseNativeDialog)
        dialog.setNameFilter("Khartes project directories (*.khprj)")
        dialog.setLabelText(QFileDialog.Accept, "Open selected .khprj folder")
        # see comment at def of onDirectoryEntered
        dialog.directoryEntered.connect(lambda d: self.onDirectoryEntered(d, dialog))
        if not dialog.exec():
            print("No khprj directory selected")
            return

        file_names = dialog.selectedFiles()
        khartes_directory = getattr(dialog, "khartes_directory", None)
        if khartes_directory is not None:
            file_names = [khartes_directory]
        print(file_names)
        if len(file_names) < 1:
            print("The khprj directory list is empty")
            return

        idir = file_names[0]
        self.loadProject(idir)

    def loadProject(self, fname):
        loading = self.showLoading()
        self.unsetProjectView()
        pv = ProjectView.open(fname)
        if not pv.valid:
            print("Project file %s not opened: %s"%(fname, pv.error))
            return
        # print("setting project view")
        self.setProjectView(pv)
        self.setWindowTitle("%s - %s"%(MainWindow.appname, Path(fname).name))
        cur_volume = pv.cur_volume
        if cur_volume is None:
            print("no cur volume set")
            spv = pv.volumes
            if len(pv.volumes) > 0:
                cur_volume = list(spv.keys())[0]
        # print("setting volume")
        self.setVolume(cur_volume, no_notify=True)
        # print("volume set")
        # intentionally called a second time to use
        # cur_volume information to set fragment view volume
        self.setProjectView(pv)
        # print("project view set")
        path = Path(fname)
        path = path.absolute()
        parent = path.parent
        self.settingsSaveDirectory(str(parent))

    def onLoadHardwiredProjectButtonClick(self, s):
        print("load hardwired project clicked")
        loading = self.showLoading()
        # pv = ProjectView.open('H:\\Vesuvius\\try1.khprj')
        pv = ProjectView.open('H:\\Vesuvius\\try2.khprj')
        if not pv.valid:
            print("Project file not opened: %s"%pv.error)
            return
        # self.project_view = pv
        self.setProjectView(pv)
        self.setWindowTitle("%s - %s"%(MainWindow.appname, Path(idir).name))
        # volume = project.volumes[0]
        # self.setProject(project)
        cur_volume = pv.cur_volume
        if cur_volume is None:
            print("no cur volume set")
            spv = pv.volumes
            if len(pv.volumes) > 0:
                cur_volume = list(spv.keys())[0]
        self.setVolume(cur_volume)

    def onExitButtonClick(self, s):
        # print("exit button clicked")
        if not self.warnIfNotSaved("exit khartes"):
            # print("Canceled by user after warning")
            return
        self.settingsSaveSizePos()
        self.app.quit()

    def onNextVolumeButtonClick(self, s):
        print("next volume clicked")
        if self.project_view is None:
            return
        spv = self.project_view.volumes
        if spv is None or len(spv) == 0:
            return
        sv = self.volumeView()
        if sv is None:
            self.setVolume(list(spv.keys())[0])
        else:
            for i in range(len(spv.values())):
                if list(spv.values())[i] == sv:
                    break
            i = (i+1)%len(spv)
            self.setVolume(list(spv.keys())[i])
        if self.volumeView().zoom == 0.:
            self.volumeView().setDefaultParameters(self)
        if self.volumeView().minZoom == 0.:
            self.volumeView().setDefaultMinZoom(self)
        self.drawSlices()


    def onToggleDirectionButtonClick(self, s):
        print("toggle direction clicked")
        if self.project_view is None:
            return
        if self.volumeView() is None:
            return

        direction = 1-self.volumeView().direction
        self.setDirectionOfCurrentVolume(direction)

    def setDirection(self, volume, direction):
        if self.project_view is None:
            print("Warning: setting direction but no project set")
            return
        self.volumes_table.model().beginResetModel()
        self.project_view.setDirection(volume, direction)
        self.volumes_table.model().endResetModel()
        if volume == self.project_view.cur_volume:
            self.drawSlices()
        # don't need to check if volume is cur_volume because 
        # project_view.setVolume checks this



    def setDirectionOfCurrentVolume(self, direction):
        # self.volumeView().setDirection(direction)
        self.project_view.setDirectionOfCurrentVolume(direction)
        print("setting volume direction to",self.volumeView().direction)
        # for fv in self.project_view.fragments.values():
        #     fv.setVolumeViewDirection(direction)
        self.drawSlices()

    def setStatusText(self, txt):
        self.status_bar.showMessage(txt)

    class LoadingWidget(QWidget):
        def __init__(self, parent, text=None):
            super(MainWindow.LoadingWidget, self).__init__(parent, Qt.Window|Qt.CustomizeWindowHint)
            layout = QVBoxLayout()
            if text is None:
                text = "Loading data..."
            self.label = QLabel(text)
            # self.label.setStyleSheet("QLabel { background-color : red; color : blue; }")
            layout.addWidget(self.label)
            self.setLayout(layout)
            font = self.label.font()
            font.setPointSize(16)
            self.label.setFont(font)
            print("Loading widget created", self)
            self.show()

        def closeEvent(self, e):
            print("Loading widget close event", self)
            e.accept()

        def mousePressEvent(self, e):
            print("Loading widget mouse press event", self)
            self.close()

    def loadingDestroyed(self, widget):
        print("Loading destroyed, loading widget to close", widget)
        widget.close()

    # Sort of complicated.  
    # We want the "Loading..." label to be
    # closed when the load operation is done.  One way to do this
    # is to tie it to a variable that will go out of scope, and
    # be deleted, when the loading operation finishes.
    # The "Loading..." label is drawn by an instance of the 
    # LoadingWidget class.  This instance is a child of the MainWindow
    # instance, and thus is not automatically deleted when it
    # goes out of scope, because from the point of view of the
    # garbage collector, it is still in use by the MainWindow instance.
    # So an instance of the Loading class
    # is created, which will be deleted when it goes out of scope;
    # this Loading class instance, when it is deleted, closes
    # the corresponding LoadingWidget instance.
    class Loading(QObject):
        def __init__(self, parent, text=None):
            super(MainWindow.Loading, self).__init__()
            widget = MainWindow.LoadingWidget(parent, text)
            widget.setAttribute(Qt.WA_DeleteOnClose)
            self.destroyed.connect(lambda o: parent.loadingDestroyed(widget))
            print("Loading created", self, widget)

    def showLoading(self, text=None):
            loading = MainWindow.Loading(self, text)
            self.app.processEvents()
            return loading

    def setVolume(self, volume, no_notify=False):
        pv = self.project_view
        if volume is not None and volume.data is None:
            loading = self.showLoading()

        self.volumes_table.model().beginResetModel()
        pv.setCurrentVolume(volume, no_notify)
        self.volumes_table.model().endResetModel()
        vv = None
        if volume is not None:
            vv = pv.cur_volume_view
            if vv.zoom == 0.:
                print("setting volume default parameters", volume.name)
                vv.setDefaultParameters(self, no_notify)
            if vv.minZoom == 0.:
                vv.setDefaultMinZoom(self)
        # print("pv set updata frag views")
        pv.updateFragmentViews()
        # print("set vol views")
        self.depth.setVolumeView(vv);
        self.xline.setVolumeView(vv);
        self.inline.setVolumeView(vv);
        self.surface.setVolumeView(vv);
        # print("draw slices")
        self.drawSlices()

    def setVolumeViewColor(self, volume_view, color):
        self.volumes_table.model().beginResetModel()
        volume_view.setColor(color)
        self.volumes_table.model().endResetModel()
        self.drawSlices()

    def setFragments(self):
        fragment_views = list(self.project_view.fragments.values())
        for fv in fragment_views:
            fv.setVolumeView(self.volumeView())
            fv.setLiveZsurfUpdate(self.live_zsurf_update)
        self.drawSlices()

    def toggleFragmentVisibility(self):
        afvs = self.project_view.activeFragmentViews(unaligned_ok=True)
        if len(afvs) == 0:
            return
        any_visible = False
        for afv in afvs:
            if afv.visible:
                any_visible = True
                break
        self.fragments_table.model().beginResetModel()
        for afv in afvs:
            afv.visible = not any_visible
            afv.notifyModified()
        self.fragments_table.model().endResetModel()
        self.drawSlices()


    def setFragmentVisibility(self, fragment, visible):
        fragment_view = self.project_view.fragments[fragment]
        if fragment_view.visible == visible:
            return
        self.fragments_table.model().beginResetModel()
        fragment_view.visible = visible
        fragment_view.notifyModified()
        self.fragments_table.model().endResetModel()
        self.drawSlices()

    def setFragmentActive(self, fragment, active, exclusive=False):
        # print("sfa", fragment.name, active, exclusive)
        fragment_view = self.project_view.fragments[fragment]
        if fragment_view.active == active:
            return
        self.fragments_table.model().beginResetModel()
        if exclusive:
            self.project_view.clearActiveFragmentViews()
        fragment_view.active = active
        fragment_view.notifyModified()
        self.fragments_table.model().endResetModel()
        # self.export_mesh_action.setEnabled(self.project_view.mainActiveVisibleFragmentView() is not None)
        # self.export_mesh_action.setEnabled(len(self.project_view.activeFragmentViews(unaligned_ok=True)) > 0)
        self.enableWidgetsIfActiveFragment()
        self.drawSlices()

    def setFragmentColor(self, fragment, color):
        self.fragments_table.model().beginResetModel()
        fragment.setColor(color)
        self.fragments_table.model().endResetModel()
        self.drawSlices()

    def unsetProjectView(self):
        if self.project_view == None:
            return
        self.setVolume(None, no_notify=True)
        self.live_zsurf_update = True
        self.live_zsurf_update_button.setChecked(self.live_zsurf_update)
        self.setFragments()
        self.project_view = None
        self.volumes_model = VolumesModel(None, self)
        self.fragments_model = FragmentsModel(None, self)
        self.save_project_action.setEnabled(False)
        self.save_project_as_action.setEnabled(False)
        self.import_nrrd_action.setEnabled(False)
        self.import_tiffs_action.setEnabled(False)
        self.enableWidgetsIfActiveFragment()
        self.drawSlices()
        self.app.processEvents()


    def setProjectView(self, project_view):
        project_view.project.modified_callback = self.projectModifiedCallback
        self.project_view = project_view
        self.volumes_model = VolumesModel(project_view, self)
        self.volumes_table.setModel(self.volumes_model)
        self.volumes_table.resizeColumnsToContents()
        self.fragments_model = FragmentsModel(project_view, self)
        self.fragments_table.setModel(self.fragments_model)
        self.fragments_table.resizeColumnsToContents()
        self.settings_vol_boxes_visible.setChecked(self.getVolBoxesVisible())
        self.settings_vol_boxes_visible2.setChecked(self.getVolBoxesVisible())
        self.settings_voxel_size_um.setToVoxelSize()
        self.save_project_action.setEnabled(False)
        self.save_project_as_action.setEnabled(True)
        self.import_nrrd_action.setEnabled(True)
        self.import_tiffs_action.setEnabled(True)
        # self.export_mesh_action.setEnabled(project_view.mainActiveFragmentView() is not None)
        # self.export_mesh_action.setEnabled(len(self.project_view.activeFragmentViews(unaligned_ok=True)) > 0)
        self.enableWidgetsIfActiveFragment()

    def setProject(self, project):
        project_view = ProjectView(project)
        self.setProjectView(project_view)
        self.setVolume(None, no_notify=True)
        self.setFragments()
        # self.setCurrentFragment(None)
        self.drawSlices()

    def resizeEvent(self, e):
        self.settingsSaveSizePos()
        self.drawSlices()

    def moveEvent(self, e):
        self.settingsSaveSizePos()

    def keyPressEvent(self, e):
        # print("key press event in main window")
        if e.key() == Qt.Key_Shift:
            t = time.time()
            # if self.shift_lock_double_click:
            if self.getShiftClicksCount() == 2:
                if t - self.last_shift_time < .5:
                    # double click
                    self.add_node_mode = not self.add_node_mode
                    self.add_node_mode_button.setChecked(self.add_node_mode)
                    self.last_shift_time = 0
                else:
                    self.last_shift_time = t
            else:
                # print("press", t)
                self.last_shift_time = t

        if e.key() == Qt.Key_L:
            self.setLiveZsurfUpdate(not self.live_zsurf_update)
            self.live_zsurf_update_button.setChecked(self.live_zsurf_update)
        elif e.modifiers() == Qt.ControlModifier and e.key() == Qt.Key_S:
            self.onSaveProjectButtonClick(True)
        elif e.key() == Qt.Key_T:
            self.toggleTrackingCursorsVisible()
            w = QApplication.widgetAt(QCursor.pos())
            method = getattr(w, "keyPressEvent", None)
            if w != self and method is not None:
                w.keyPressEvent(e)
            self.drawSlices()
        elif e.key() == Qt.Key_V:
            self.toggleFragmentVisibility()
            w = QApplication.widgetAt(QCursor.pos())
            method = getattr(w, "keyPressEvent", None)
            if w != self and method is not None:
                w.keyPressEvent(e)
        else:
            w = QApplication.widgetAt(QCursor.pos())
            method = getattr(w, "keyPressEvent", None)
            if w != self and method is not None:
                w.keyPressEvent(e)

    def keyReleaseEvent(self, e):
        # if e.key() == Qt.Key_Shift and not self.shift_lock_double_click:
        if e.key() == Qt.Key_Shift and self.getShiftClicksCount() == 1:
            t = time.time()
            # print("release", t)
            if t - self.last_shift_time < .25:
                # quick click-and-release
                self.add_node_mode = not self.add_node_mode
                self.add_node_mode_button.setChecked(self.add_node_mode)
            w = QApplication.widgetAt(QCursor.pos())
            method = getattr(w, "keyReleaseEvent", None)
            if w != self and method is not None:
                w.keyReleaseEvent(e)
            self.last_shift_time = 0
        else:
            w = QApplication.widgetAt(QCursor.pos())
            method = getattr(w, "keyReleaseEvent", None)
            if w != self and method is not None:
                w.keyReleaseEvent(e)

    def drawSlices(self):
        self.depth.drawSlice()
        self.xline.drawSlice()
        self.inline.drawSlice()
        self.surface.drawSlice()

    def getVoxelSizeUm(self):
        if self.project_view is None:
            return Project.default_voxel_size_um
        return self.project_view.project.getVoxelSizeUm()

    def setVoxelSizeUm(self, size):
        if self.project_view is None:
            return
        old_size = self.getVoxelSizeUm()
        if old_size == size:
            return
        self.project_view.project.setVoxelSizeUm(size)
        self.drawSlices()

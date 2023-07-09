from pathlib import Path
import shutil

from PyQt5.QtWidgets import (
        QAction, QApplication, QAbstractItemView,
        QCheckBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout, 
        QLabel,
        QMainWindow, QMessageBox,
        QPlainTextEdit, QPushButton,
        QStatusBar,
        QTableView, QTabWidget, QTextEdit, QToolBar,
        QVBoxLayout, 
        QWidget, 
        )
from PyQt5.QtCore import QSize, Qt, qVersion, QSettings
from PyQt5.QtGui import QPalette, QColor, QCursor

from tiff_loader import TiffLoader
from data_window import DataWindow, SurfaceWindow
from project import Project, ProjectView
from fragment import Fragment, FragmentsModel
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

class CreateFragmentButton(QPushButton):
    def __init__(self, main_window, parent=None):
        super(CreateFragmentButton, self).__init__("Start New Fragment", parent)
        self.main_window = main_window
        self.setToolTip("Once the new fragment is created use\nshift plus left mouse button to create new nodes")
        self.clicked.connect(self.onButtonClicked)

    def onButtonClicked(self, s):
        self.main_window.createFragment()


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


class MainWindow(QMainWindow):

    appname = "χάρτης"

    def __init__(self, appname, app):
        super(MainWindow, self).__init__()

        self.app = app
        self.settings = QSettings(QSettings.IniFormat, QSettings.UserScope, 'khartes.org', 'khartes')
        print(self.settings.fileName())
        qv = [int(x) for x in qVersion().split('.')]
        # print("Qt version", qv)
        if qv[0] > 5 or qv[0] < 5 or qv[1] < 12:
            print("Need to use Qt version 5, subversion 12 or above")
            # 5.12 or above is needed for QImage::Format_RGBX64
            exit()

        self.setWindowTitle(MainWindow.appname)
        self.setMinimumSize(QSize(750,600))
        self.settingsApplySizePos()

        grid = QGridLayout()

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

        self.import_tiffs_action = QAction("Import TIFF files...", self)
        self.import_tiffs_action.triggered.connect(self.onImportTiffsButtonClick)
        self.import_tiffs_action.setEnabled(False)

        self.export_mesh_action = QAction("Export fragment as mesh...", self)
        self.export_mesh_action.triggered.connect(self.onExportAsMeshButtonClick)
        self.export_mesh_action.setEnabled(False)

        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.onExitButtonClick)

        self.menu = self.menuBar()
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

        self.toggle_direction_action = QAction("Toggle direction", self)
        self.toggle_direction_action.triggered.connect(self.onToggleDirectionButtonClick)
        self.next_volume_action = QAction("Next volume", self)
        self.next_volume_action.triggered.connect(self.onNextVolumeButtonClick)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.project_view = None
        # is this needed?
        self.volumes_model = VolumesModel(None, self)
        self.tiff_loader = TiffLoader(self)

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

    def addSettingsPanel(self):
        panel = QWidget()
        hlayout = QHBoxLayout()
        panel.setLayout(hlayout)
        slices_layout = QVBoxLayout()
        hlayout.addLayout(slices_layout)
        slices_layout.addWidget(QLabel("Slices"))
        vbv = VolBoxesVisibleCheckBox(self)
        self.settings_vol_boxes_visible = vbv
        slices_layout.addWidget(vbv)
        slices_layout.addStretch()
        fragment_layout = QVBoxLayout()
        fragment_layout.addWidget(QLabel("Fragment View"))

        self.tab_panel.addTab(panel, "Settings")

    def volumeView(self):
        return self.project_view.cur_volume_view

    def getVolBoxesVisible(self):
        if self.project_view is None:
            return
        slices = self.project_view.settings['slices']
        vbv = 'vol_boxes_visible'
        return slices[vbv]

    def setVolBoxesVisible(self, value):
        if self.project_view is None:
            return
        slices = self.project_view.settings['slices']
        vbv = 'vol_boxes_visible'
        old_value = slices[vbv]
        if old_value == value:
            return
        slices[vbv] = value
        self.settings_vol_boxes_visible.setChecked(self.getVolBoxesVisible())
        self.settings_vol_boxes_visible2.setChecked(self.getVolBoxesVisible())
        self.drawSlices()

    def onNewFragmentButtonClick(self, s):
        self.createFragment()

    def createFragment(self):
        vv = self.volumeView()
        if vv is None:
            print("Warning, cannot create new fragment without volume view set")
            return
        names = set()
        for frag in self.project_view.fragments.keys():
            name = frag.name
            names.add(name)
        stem = "frag"
        mfv = self.project_view.mainActiveVisibleFragmentView(unaligned_ok=True)
        if mfv is not None:
            stem = mfv.fragment.name
        for i in range(1,1000):
            # name = "%s%d"%(stem,i)
            name = Utils.nextName(stem, i)
            if name not in names:
                break
        # print("color",color)
        frag = Fragment(name, vv.direction)
        frag.setColor(Utils.getNextColor())
        print("created fragment %s"%frag.name)
        self.fragments_table.model().beginResetModel()
        self.project_view.project.addFragment(frag)
        self.fragments_table.model().endResetModel()
        self.setFragments()
        fv = self.project_view.fragments[frag]
        fv.active = True
        self.export_mesh_action.setEnabled(len(self.project_view.activeFragmentViews(unaligned_ok=True)) > 0)
        # need to make sure new fragment is added to table
        # before calling scrollToRow
        self.app.processEvents()
        index = self.project_view.project.fragments.index(frag)
        self.fragments_table.model().scrollToRow(index)

    def renameFragment(self, frag, name):
        if frag.name == name:
            return
        self.fragments_table.model().beginResetModel()
        frag.name = name
        proj = self.project_view.project
        proj.alphabetizeFragments()
        self.fragments_table.model().endResetModel()
        self.app.processEvents()
        index = proj.fragments.index(frag)
        self.fragments_table.model().scrollToRow(index)
        self.drawSlices()

    def addPointToCurrentFragment(self, tijk):
        # cur_frag_view = self.project_view.cur_fragment_view
        cur_frag_view = self.project_view.mainActiveVisibleFragmentView()
        if cur_frag_view is None:
            print("no current fragment view set")
            return
        self.fragments_table.model().beginResetModel()
        cur_frag_view.addPoint(tijk)
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

    def onNewProjectButtonClick(self, s):
        print("new project clicked")
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

        new_prj = Project.create(pdir)
        if not new_prj.valid:
            err = new_prj.error
            print("Failed to create new project: %s", err)
            return

        self.setWindowTitle("%s - %s"%(MainWindow.appname, pdir.name))
        self.setProject(new_prj)

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
        dialog.setLabelText(QFileDialog.Accept, "Save in .khprj folder")
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

    # TODO Should have dialog where user can select infill parameter
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

        # TODO: allow the user to set the infill
        # TODO: save all active fragments
        # err = Fragment.saveListAsObjMesh(frags, pname, 16)
        err = Fragment.saveListAsObjMesh(fvs, pname, 16)

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

    def onOpenProjectButtonClick(self, s):
        print("open project clicked")
        ''''''
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

        loading = self.showLoading()
        pv = ProjectView.open(idir)
        if not pv.valid:
            print("Project file %s not opened: %s"%(idir, pv.error))
            return
        self.setProjectView(pv)
        self.setWindowTitle("%s - %s"%(MainWindow.appname, Path(idir).name))
        cur_volume = pv.cur_volume
        if cur_volume is None:
            print("no cur volume set")
            spv = pv.volumes
            if len(pv.volumes) > 0:
                cur_volume = list(spv.keys())[0]
        self.setVolume(cur_volume)
        # intentionally called a second time to use
        # cur_volume information to set fragment view volume
        self.setProjectView(pv)
        path = Path(idir)
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

    class Loading(QWidget):
        def __init__(self, text=None):
            super().__init__()
            layout = QVBoxLayout()
            if text is None:
                text = "Loading data..."
            self.label = QLabel(text)
            # self.label.setStyleSheet("QLabel { background-color : red; color : blue; }")
            layout.addWidget(self.label)
            self.setLayout(layout)
            self.setWindowFlags(Qt.CustomizeWindowHint)
            font = self.label.font()
            font.setPointSize(16)
            self.label.setFont(font)

    def showLoading(self, text=None):
            loading = MainWindow.Loading(text)
            loading.show()
            # needed to make label visible
            self.app.processEvents()
            return loading

    def setVolume(self, volume):
        if volume is not None and volume.data is None:
            loading = self.showLoading()

        self.volumes_table.model().beginResetModel()
        self.project_view.setCurrentVolume(volume)
        self.volumes_table.model().endResetModel()
        vv = None
        if volume is not None:
            vv = self.project_view.cur_volume_view
            if vv.zoom == 0.:
                print("setting volume default parameters", volume.name)
                vv.setDefaultParameters(self)
            if vv.minZoom == 0.:
                vv.setDefaultMinZoom(self)
        self.project_view.updateFragmentViews()
        self.depth.setVolumeView(vv);
        self.xline.setVolumeView(vv);
        self.inline.setVolumeView(vv);
        self.surface.setVolumeView(vv);
        self.drawSlices()

    def setVolumeViewColor(self, volume_view, color):
        self.volumes_table.model().beginResetModel()
        volume_view.setColor(color)
        self.volumes_table.model().endResetModel()
        self.drawSlices()

    # TODO: this should be called whenever the user creates
    # a new fragment
    def setFragments(self):
        fragment_views = list(self.project_view.fragments.values())
        for fv in fragment_views:
            fv.setVolumeView(self.volumeView())
        self.drawSlices()

    def setFragmentVisibility(self, fragment, visible):
        fragment_view = self.project_view.fragments[fragment]
        if fragment_view.visible == visible:
            return
        self.fragments_table.model().beginResetModel()
        fragment_view.visible = visible
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
        self.fragments_table.model().endResetModel()
        # self.export_mesh_action.setEnabled(self.project_view.mainActiveVisibleFragmentView() is not None)
        self.export_mesh_action.setEnabled(len(self.project_view.activeFragmentViews(unaligned_ok=True)) > 0)
        self.drawSlices()

    def setFragmentColor(self, fragment, color):
        self.fragments_table.model().beginResetModel()
        fragment.setColor(color)
        self.fragments_table.model().endResetModel()
        self.drawSlices()

    def setProjectView(self, project_view):
        self.project_view = project_view
        self.volumes_model = VolumesModel(project_view, self)
        self.volumes_table.setModel(self.volumes_model)
        self.volumes_table.resizeColumnsToContents()
        self.fragments_model = FragmentsModel(project_view, self)
        self.fragments_table.setModel(self.fragments_model)
        self.fragments_table.resizeColumnsToContents()
        self.settings_vol_boxes_visible.setChecked(self.getVolBoxesVisible())
        self.settings_vol_boxes_visible2.setChecked(self.getVolBoxesVisible())
        self.save_project_action.setEnabled(True)
        self.save_project_as_action.setEnabled(True)
        self.import_nrrd_action.setEnabled(True)
        self.import_tiffs_action.setEnabled(True)
        # self.export_mesh_action.setEnabled(project_view.mainActiveFragmentView() is not None)
        self.export_mesh_action.setEnabled(len(self.project_view.activeFragmentViews(unaligned_ok=True)) > 0)

    def setProject(self, project):
        project_view = ProjectView(project)
        self.setProjectView(project_view)
        self.setVolume(None)
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
        if e.modifiers() == Qt.ControlModifier and e.key() == Qt.Key_S:
            self.onSaveProjectButtonClick(True)
        else:
            w = QApplication.widgetAt(QCursor.pos())
            method = getattr(w, "keyPressEvent", None)
            if w != self and method is not None:
                w.keyPressEvent(e)

    def drawSlices(self):
        self.depth.drawSlice()
        self.xline.drawSlice()
        self.inline.drawSlice()
        self.surface.drawSlice()

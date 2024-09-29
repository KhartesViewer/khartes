import re
import time
from pathlib import Path
# import traceback
import numpy as np
from volume import Volume
from volume_zarr import CachedZarrVolume

from PyQt5.QtWidgets import (
        QAction,
        QApplication, QAbstractItemView,
        QCheckBox, QColorDialog, QComboBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout, 
        QLabel, QLineEdit,
        QMainWindow, QMessageBox,
        QPlainTextEdit, QPushButton,
        QStatusBar,
        QTableView, QTabWidget, QTextEdit, QToolBar,
        QVBoxLayout, 
        QWidget, 
        )
from PyQt5.QtCore import QSize, Qt, qVersion, QSettings
from PyQt5.QtGui import (
        # QAction, 
        QPalette, QColor, QCursor, QIntValidator)

class ColorEdit(QPushButton):

    def __init__(self, tiff_loader):
        super(ColorEdit, self).__init__()
        self.tiff_loader = tiff_loader
        self.clicked.connect(self.onClicked)
        self.setColor("magenta")

    def setColor(self, color_name):
        self.setStyleSheet("ColorEdit {background-color: %s}"%color_name)
        #print("set color to", color_name)


    def getColor(self):
        return self.palette().color(QPalette.Window)

    def onClicked(self, d):
        old_color = self.getColor()
        new_color = QColorDialog.getColor(old_color, self)
        if new_color.isValid() and new_color != old_color:
            self.setColor(new_color.name())
            self.tiff_loader.onChange()

class ZarrStreamLoader(QMainWindow):

    default_urls = [
       "https://dl.ash2txt.org/other/dev/scrolls/1/volumes/54keV_7.91um.zarr",
       "https://dl.ash2txt.org/other/dev/scrolls/2/volumes/54keV_7.91um.zarr",
       "https://dl.ash2txt.org/other/dev/scrolls/2/volumes/88keV_7.91um.zarr",
       "https://dl.ash2txt.org/other/dev/scrolls/3/volumes/53keV_3.24um.zarr",
       "https://dl.ash2txt.org/other/dev/scrolls/3/volumes/53keV_7.91um.zarr",
       "https://dl.ash2txt.org/other/dev/scrolls/3/volumes/70keV_3.24um.zarr",
       "https://dl.ash2txt.org/other/dev/scrolls/4/volumes/53keV_7.91um.zarr",
       "https://dl.ash2txt.org/other/dev/scrolls/4/volumes/88keV_3.24um.zarr",
            ]

    def __init__(self, main_window):
        super(ZarrStreamLoader, self).__init__(main_window)

        self.main_window = main_window
        # a bit confusing: vc_render is the flag, vcrender is the widget
        self.vc_render = False
        self.setStyleSheet("font-size: 12pt;")
        self.setWindowTitle("Connect to remote OME/Zarr data store")
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("URL (you can select or edit):"))
        # urlbutton = QPushButton("Select .zarr directory or one containing .tif files")
        '''
        dirbutton.clicked.connect(self.onDirButtonClicked)
        '''
        urlcb = QComboBox(self)
        urlcb.setEditable(True)
        # for url in self.default_urls:
        #     urlcb.addItem(url)
        urlcb.addItems(self.default_urls)
        self.urlcb = urlcb

        hbox.addWidget(urlcb)
        hbox.addStretch()
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Volume name:"))
        self.nameedit = QLineEdit()
        self.nameedit.textChanged.connect(self.onNameEdited)
        hbox.addWidget(self.nameedit)
        
        self.vcrender = QCheckBox("Data is from vc_layers")
        self.vcrender.setChecked(self.vc_render)
        self.vcrender.clicked.connect(self.onVcrenderClicked)
        hbox.addStretch()
        hbox.addWidget(self.vcrender)
        vbox.addLayout(hbox)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Volume color:"))
        self.color_editor = ColorEdit(self)
        hbox.addWidget(self.color_editor)
        hbox.addStretch()
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        self.gobutton = QPushButton("Connect")
        self.gobutton.clicked.connect(self.onGoButtonClicked)
        self.gobutton.setEnabled(False)
        hbox.addWidget(self.gobutton)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.onCancelClicked)
        hbox.addWidget(cancel)
        vbox.addLayout(hbox)

        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)
        self.name_valid = False
        self.name = None
        self.directory_valid = False
        self.directory = None
        self.directory_has_tiffs = False

    """
    # Qt trickery to get around a problem with QFileDialog
    # when the user double-clicks on
    # a directory.  If the directory name ends with ".zarr",
    # or contains TIFF files
    # we want that directory name to be immediately sent back to
    # the dialog's caller, rather than having the QFileDialog 
    # descend into the selected directory
    def onDirectoryEntered(self, sdir, dialog):
        print("zarr/tiff directory entered", sdir)
        if self.main_window.is_macos:
            print("ignoring because running MacOS")
            return
        ignore_directory = getattr(dialog, "khartes_ignore_directory", None)
        # print("ignore_directory", ignore_directory)
        if ignore_directory is not None and Path(sdir) == Path(ignore_directory):
            # print("ignoring",sdir)
            return
        # First, is this a .zarr directory?j
        # TODO: this can be a zarr/OME directory even if it
        # doesn't end in .zarr, if it contains certain dot files.
        if sdir.endswith(".zarr"):
            dialog.zarr_directory = sdir
            # The sleep is needed to prevent crashes 
            # with PySide6
            # time.sleep(.5)
            # dialog.done(1)
            dialog.accept()

        # if not, then see if directory contains TIFF files:
        # if sdir has no sub-directories
        # and sdir has .tif files,
        # call dialog.done(1)
        # else return None
        pdir = Path(sdir)
        # note that the following globs are enclosed in list()
        # calls because the globs are generators.
        # "*/" returns list of all directories in pdir
        # but it only works on python 3.11 and above
        # dirs = list(pdir.glob("*/"))
        # this determines if there are any non-empty sub directories:
        dirs = list(pdir.glob("*/*"))
        tifs = list(pdir.glob("*.tif"))
        if len(dirs) == 0 and len(tifs) > 0:
            # print("match", sdir)
            dialog.zarr_directory = sdir
            # The sleep is needed to prevent crashes 
            # with PySide6
            # time.sleep(.5)
            # dialog.done(1)
            dialog.accept()
        return None

    def onDirButtonClicked(self, s):
        print("On dir button clicked")
        title = "Attach Zarr/OME/TIFFs"
        dialog = QFileDialog(self)
        sdir = self.main_window.settingsGetDirectory("zarr_")
        # print("sdir", sdir)
        if sdir is not None and Path(sdir).is_dir():
            dialog.setDirectory(sdir)
            dialog.khartes_ignore_directory = sdir
            # print("will ignore",sdir)
        dialog.setOptions(QFileDialog.ShowDirsOnly)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setLabelText(QFileDialog.Accept, "Select .zarr folder or folder containing TIFF files")
        dialog.directoryEntered.connect(lambda d: self.onDirectoryEntered(d, dialog))
        de = dialog.exec()
        zarr_directory = getattr(dialog, "zarr_directory", None)
        if not de and zarr_directory is None:
            print("No directory selected")
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setIcon(QMessageBox.Information)
            msg.setText("No directory selected")
            msg.exec()
            self.directory_valid = False
            self.onChange()
            return

        # if user pushes "Select .zarr folder" button, 
        # dialog.selectedFiles() is set
        file_names = dialog.selectedFiles()
        print("file_names", file_names)
        # otherwise, if user double-clicked on a zarr or tiff directory,
        # dialog.zarr_directory is set
        if zarr_directory is not None:
            file_names = [zarr_directory]
        if len(file_names) < 1:
            # Probably should never reach here
            print("The tiff directory list is empty")
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setIcon(QMessageBox.Information)
            msg.setText("No directory selected")
            msg.exec()
            self.directory_valid = False
            self.onChange()
            return

        sdir = file_names[0]
        pdir = Path(sdir)
        # make sure directory exists
        if not pdir.exists():
            # probably won't get here, because directory-selection
            # widget checks this as well
            print("Directory %s does not exist"%pdir)
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Directory %s does not exist"%pdir)
            msg.exec()
            return

        # At this point we don't know whether the directory
        # is zarr, OME, or contains TIFFs
        # Convert generator to list
        tifs = list(pdir.glob("*.tif"))
        self.directory_has_tiffs = (len(tifs) > 0)

        self.main_window.settingsSaveDirectory(str(pdir.parent), "zarr_")
        self.dirbutton.setText(str(pdir))
        self.directory = pdir
        self.directory_valid = True

        pv = self.main_window.project_view
        if pv is not None:
            cv = pv.cur_volume
            if cv is not None:
                self.vc_render = cv.from_vc_render
                self.vcrender.setChecked(self.vc_render)
        
        self.onChange()
"""
    
    def onNameEdited(self, txt):
        vol_names = set(v.name for v in self.main_window.project_view.volumes.keys())
        if txt == "" or txt in vol_names:
            self.name_valid = False
            self.nameedit.setStyleSheet("QLineEdit { color: red }")
        else:
            self.name_valid = True
            self.name = txt
            self.nameedit.setStyleSheet("")
        # print("oNE", txt, self.name_valid)
        self.onChange()

    def color(self):
        return self.color_editor.getColor()

    def onVcrenderClicked(self, s):
        self.vc_render = self.vcrender.isChecked()
        # self.main_window.drawSlices()
    
    def onGoButtonClicked(self, s):
        # title = "Attach Zarr/OME/TIFFs"
        title = "Connect to remote OME/Zarr data store"
        self.directory = self.urlcb.currentText()

        print(self.directory, self.name, self.color().name())
        vcrender = self.vc_render
        old_volume = self.main_window.project_view.cur_volume
        # unloads old volume
        self.main_window.setVolume(None)

        pdir = self.directory
        volume_name = self.name
        project = self.main_window.project_view.project
        main_window = self.main_window
        main_window.app.processEvents()
        loading = main_window.showLoading()
        """
        if self.directory_has_tiffs:
            tifs = list(pdir.glob("*.tif"))
            tiff0 = tifffile.imread(tifs[0])
            shape = tiff0.shape
            # createFromTiffs accepts 2D and 3D TIFF files.  However,
            # zarr data stores created from 2D TIFFs are in practice
            # almost unusable.
            # So if the user selects a directory with 2D TIFFs,
            # they are redirected to the create-volume-from-tiffs
            # option.
            # if not vcrender and (len(shape) == 2 or (1 in shape)):
            if len(shape) == 2 or (1 in shape):
                emsg = '''This directory contains flat (2D) TIFF files.\nFiles of this type should be read using the "Create volume from TIFF files..." option.  The "Attach Zarr/OME/TIFF data store..." option expects TIFF files to be 3D (multi-layer), like those found in the volume_grid directory. '''
                new_volume = CachedZarrVolume.createErrorVolume(emsg)
            else:
                new_volume = CachedZarrVolume.createFromTiffs(project, pdir, volume_name, vcrender)
        else:
            new_volume = CachedZarrVolume.createFromZarr(project, pdir, volume_name, vcrender)
        """
        # new_volume = CachedZarrVolume.createFromZarr(project, pdir, volume_name, vcrender)
        options = None
        if self.main_window.stream_use_cache_directory:
            options = {"stream_cache_directory", self.main_window.stream_cache_directory}
        new_volume = CachedZarrVolume.createFromZarr(project, pdir, volume_name, vcrender, options)
        loading = None

        if new_volume is None or not new_volume.valid:
            err_msg = ""
            if new_volume is not None:
                err_msg = new_volume.error
            print("new volume error", err_msg)
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Failed to attach data store : %s"%new_volume.error)
            msg.exec()
            self.main_window.setVolume(old_volume)
        else:
            self.main_window.setVolume(new_volume)
            vv = self.main_window.project_view.cur_volume_view
            # vv.setColor(QColor(self.color()))
            self.main_window.setVolumeViewColor(vv, QColor(self.color()))
            self.hide()
        # unset name of volume
        self.nameedit.setText("")
        # setText causes a call to self.onNameEdited which
        # in turn calls self.onChange
    
    def onCancelClicked(self, s):
        self.hide()

    '''
    def hideEvent(self, e):
        print("Hiding self; spontaneous:", e.spontaneous())
        print("Traceback:")
        traceback.print_stack()
        print("End of traceback")
    '''

    def areValid(self):
        if not self.name_valid:
            return False
        directory = self.urlcb.currentText()
        if directory == "":
            return False
        return True
    
    def onChange(self):
        if self.areValid():
            self.gobutton.setEnabled(True)
            self.gobutton.setDefault(True)
        else:
            self.gobutton.setEnabled(False)
            self.gobutton.setDefault(False)
        self.raise_()

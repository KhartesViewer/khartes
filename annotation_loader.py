import re
from pathlib import Path
import numpy as np
import cv2
from volume import Volume
from volume_zarr import CachedZarrVolume

from PyQt5.QtWidgets import (
        QAction, QApplication, QAbstractItemView,
        QCheckBox, QColorDialog,
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
from PyQt5.QtGui import QPalette, QColor, QCursor, QIntValidator
import os
import zarr

CHUNK_SIZE = 500



class AnnotationData():
    
    def __init__(self, path, volume_shape=None):
        self.path = path
        if os.path.exists(self.path):
            self.volume = self.load_writable_volume()
            if volume_shape is not None:
                assert self.volume.shape == volume_shape
        else:
            assert volume_shape is not None
            self.volume = self.create_writeable_volume(volume_shape)
        self.valid = True

    @property
    def shape(self):
        return self.volume.shape

    @staticmethod
    def load_from_ref(fn):
        """Loads annotation data from a reference file in the project directory.
        """
        with open(fn, "r") as infile:
            annopath = infile.readline().strip()
        return AnnotationData(annopath)

    def load_writable_volume(self):
        """This function takes a path to a zarr DirectoryStore folder that contains
        chunked volume information.  This zarr Array is writable and can be used
        for fast persistent storage.
        """
        if not os.path.exists(self.path):
            raise ValueError("Error: f{path} does not exist")
        store = zarr.DirectoryStore(self.path)
        root = zarr.group(store=store, overwrite=False)
        return root.volume

    def create_writeable_volume(self, volume_shape):
        """Generates a new zarr Array object serialized to disk as an empty array of
        zeros.  Requires the size of the array to be given; this may be arbitrarily
        large.  When initialized, this takes up very little space on disk since all
        chunks are empty.  As it is written, it can get much larger.  Be sure you
        have enough disk space!
        """
        if os.path.exists(self.path):
            raise ValueError("Error: f{path} already exists")
        store = zarr.DirectoryStore(self.path)
        root = zarr.group(store=store, overwrite=True)
        volume = root.zeros(
            name="volume",
            shape=volume_shape,
            chunks=tuple([CHUNK_SIZE for d in volume_shape]),
            dtype=np.uint16,
            write_empty_chunks=False,
        )
        return volume
    
    def write_annotations(self):
        pass

class AnnotationLoader(QMainWindow):
    """Class to allow opening an existing annotation (as a .volzarr directory)
    or creation of a new one.
    """

    def __init__(self, main_window):
        super(AnnotationLoader, self).__init__(main_window)

        self.main_window = main_window
        widget = QWidget()

        self.setWindowTitle("Annotation loader")
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        dirlabel = QLabel("Directory:")
        hbox.addWidget(dirlabel)
        dirbutton = QPushButton("Select directory")
        dirbutton.clicked.connect(self.onDirButtonClicked)
        self.dirbutton = dirbutton
        hbox.addWidget(dirbutton)
        hbox.addStretch()

        vbox.addLayout(hbox)
        self.txt = QLabel("")
        vbox.addWidget(self.txt)

        hbox = QHBoxLayout()
        self.newbutton = QPushButton("Create New")
        self.newbutton.clicked.connect(self.onNewButtonClicked)
        self.newbutton.setEnabled(False)
        hbox.addWidget(self.newbutton)

        self.openbutton = QPushButton("Open Existing")
        self.openbutton.clicked.connect(self.onOpenButtonClicked)
        self.openbutton.setEnabled(False)
        hbox.addWidget(self.openbutton)

        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.onCancelClicked)
        hbox.addWidget(cancel)
        vbox.addLayout(hbox)

        widget.setLayout(vbox)
        self.setCentralWidget(widget)

    def onDirButtonClicked(self, s):
        dialog = QFileDialog(self)
        dialog.setOptions(QFileDialog.ShowDirsOnly)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setLabelText(QFileDialog.Accept, "Open annotation directory")
        dialog.exec()

        filenames = dialog.selectedFiles()
        print(filenames)
        assert len(filenames) == 1
        dirname = filenames[0]
        self.txt.setText(dirname)
        if dirname.endswith(".volzarr"):
            self.openbutton.setEnabled(True)
        else:
            self.newbutton.setEnabled(True)

    def onNewButtonClicked(self, s):
        dirname = self.txt.text()
        # We need to query the volume to get the size of the annotations to generate
        vv = self.main_window.volumeView()
        volume_shape = vv.volume.shape
        path = os.path.join(dirname, f"{vv.volume.name}_annotations.volzarr")
        self.main_window.annotation = AnnotationData(path, volume_shape)
        self.main_window.annotation_label.setText(path)

        ref_filepath = os.path.join(
            self.main_window.project_view.project.annotations_path,
            f"{vv.volume.name}_annotations.volzarr",
        )
        with open(ref_filepath, "w") as outfile:
            outfile.write(f"{path}")

        self.hide()

    def onOpenButtonClicked(self, s):
        dirname = self.txt.text()
        vv = self.main_window.volumeView()
        volume_shape = vv.volume.shape
        self.main_window.annotation = AnnotationData(dirname, volume_shape)
        self.main_window.annotation_label.setText(dirname)

        ref_filepath = os.path.join(
            self.main_window.project_view.project.annotations_path,
            f"{vv.volume.name}_annotations.volzarr",
        )
        if not os.path.exists(ref_filepath):
            with open(ref_filepath, "w") as outfile:
                outfile.write(f"{dirname}")

        self.hide()

    def onCancelClicked(self, s):
        self.hide()
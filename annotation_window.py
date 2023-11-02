from pathlib import Path
import shutil
import copy
import os
import json
import time
import numpy as np

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
from trgl_fragment import TrglFragment, TrglFragmentView
from base_fragment import BaseFragment, BaseFragmentView
from volume import (
        Volume, VolumesModel, 
        DirectionSelectorDelegate,
        ColorSelectorDelegate)
from ppm import Ppm
from utils import Utils

class AnnotationWindow(QWidget):

    def __init__(self, main_window, slices=9):
        super(AnnotationWindow, self).__init__()
        self.show()
        self.setWindowTitle("Volume Annotations")
        self.main_window = main_window
        self.volume_view = None
        self.has_had_volume_view = False
        self.slices = slices

        grid = QGridLayout()
        self.depth = [
            DataWindow(self.main_window, 2)
            for i in range(slices)
        ]
        self.inline = [
            DataWindow(self.main_window, 0)
            for i in range(slices)
        ]
        self.xline = [
            DataWindow(self.main_window, 1)
            for i in range(slices)
        ]

        for i in range(slices):
            grid.addWidget(self.xline[i], 0, i)
            grid.addWidget(self.inline[i], 1, i)
            grid.addWidget(self.depth[i], 2, i)

        self.setLayout(grid)

    def setVolumeView(self, volume_view):
        print("VV set")
        self.volume_view = volume_view
        if volume_view is None:
            return
        for i in range(self.slices):
            for datawindow in [self.depth[i], self.inline[i], self.xline[i]]:
                datawindow.setVolumeView(volume_view)

    def drawSlices(self):
        print("drawing")
        for i in range(self.slices):
            for datawindow in [self.depth[i], self.inline[i], self.xline[i]]:
                offsets = [0, 0, 0]
                axis = datawindow.axis
                offsets[axis] += (i - (self.slices // 2)) * 10
                datawindow.drawSlice(offsets, crosshairs=False, fragments=False)

    def closeEvent(self, event):
        """We need to reset the main window's link to this when 
        the user closes this window.
        """
        print("Closing window")
        self.main_window.annotation_window = None
        event.accept()
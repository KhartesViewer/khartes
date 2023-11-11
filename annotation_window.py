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
        QTableView, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QToolBar,
        QVBoxLayout, 
        QWidget,
        QSlider,
        QHeaderView,
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
from utils import COLORLIST

import maxflow
from skimage.morphology import flood
from skimage.segmentation import expand_labels, watershed
from skimage.measure import label as apply_labels

MAXFLOW_STRUCTURE = np.array(
    [[[0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]],

    [[0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]],

    [[0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]]]
)

def compute_linkages(new_labels, saved_labels):
    """Counts the number of adjacent positions between every two labels.
    """
    linkages = {}
    if saved_labels is None:
        return linkages
    expanded = expand_labels(new_labels)
    for l in [label for label in np.unique(expanded) if label > 0]:
        adjacent = (expanded == l) & (saved_labels > 0)
        links = {}
        for a in np.unique(saved_labels[adjacent]):
            mask = adjacent & (saved_labels == a)
            links[a] = mask.sum()
        linkages[l] = links
    return linkages

def find_sheets(section, saved_labels=None, threshold=32000, minsize=50, minval=2500):
    """Takes a 3D numpy array of uint16s and tries to separate papyrus sheets from 
    background.  First applies a simple threshold, then filters the resulting
    regions for both pixel count and average signal.

    Returns a new numpy array of the same size of integers indicating which 
    label is associated with each pixel.  
    """
    if saved_labels is None:
        mask = (section > threshold)
    else:
        mask = (section > threshold) & (saved_labels == 0)
    section_labels = apply_labels(mask).astype(np.uint16)
    label_ids = np.unique(section_labels)
    for l in label_ids:
        if l == 0:
            # These were below the threshold
            continue
        mask = section_labels == l
        count = mask.sum()
        value = np.mean(section[mask]) - threshold
        if count < minsize or value < minval:
            section_labels[mask] = 0
    # Resort the labels to increase incrementally from 1
    label_ids = sorted(np.unique(section_labels))
    for i, l in enumerate(label_ids):
        if l == 0:
            continue
        mask = section_labels == l
        section_labels[mask] = i
    return section_labels


def split_label_maxflow(
        section_signal,
        section_labels,
        new_labels,
        split_label,
        new_label,
        source_label,
        sink_label,
    ):
    """Tries to split one of the new labels into two separate labels, one connected to the source
    label and one connected to the sink label.  Uses a maximum flow/minimum cut algorithm to 
    try to find the splitting surface that goes through the fewest pixels with the minimum signal.
    """
    graph = maxflow.GraphFloat()
    nodes = graph.add_grid_nodes(section_labels.shape)
    weights = np.zeros_like(section_signal)
    keep_mask = (section_labels == source_label) | (section_labels == sink_label) | (new_labels == split_label)
    weights[keep_mask] = section_signal[keep_mask]
    weights = np.power((weights - 32000) / 64000, 2)
    graph.add_grid_edges(nodes, weights=weights, structure=MAXFLOW_STRUCTURE)
    # Add extremely high capacities to the source & sink nodes
    sourcecaps = np.zeros_like(weights)
    sinkcaps = np.zeros_like(weights)
    sourcecaps[section_labels == source_label] = 1e6
    sinkcaps[section_labels == sink_label] = 1e6

    graph.add_grid_tedges(nodes, sourcecaps, sinkcaps)
    graph.maxflow()
    sgm = graph.get_grid_segments(nodes) & (new_labels > 0)
    new_labels[sgm] = new_label
    return new_labels


class AnnotationWindow(QWidget):

    def __init__(self, main_window, slices=11):
        super(AnnotationWindow, self).__init__()
        self.show()
        self.setWindowTitle("Volume Annotations")
        self.main_window = main_window
        self.volume_view = None
        # This is a copy of the central region of the box, for doing
        # automatic volume segmentation on.
        self.signal_section = None
        self.new_labels = None
        self.saved_labels = None
        self.slices = slices
        # Radius of the central region of the box
        self.radius = [30, 30, 30]
        self.update_annotations = False

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

        # Set up the control panel below the images
        panel = QWidget()
        hlayout = QHBoxLayout()
        vlayout = QVBoxLayout()
        label = QLabel("New volume segments")
        label.setAlignment(Qt.AlignLeft)
        vlayout.addWidget(label)
        auto_annotate = QCheckBox("Auto-update annotations")
        auto_annotate.setChecked(self.update_annotations)
        auto_annotate.stateChanged.connect(self.checkAutoUpdate)
        vlayout.addWidget(auto_annotate)

        # Sliders for controlling the radius of annotation along each axis
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Z Radius:"))
        self.zslider = QSlider(Qt.Horizontal)
        self.zslider.setMinimum(5)
        self.zslider.setMaximum(75)
        self.zslider.setValue(self.radius[0])
        self.zslider.valueChanged.connect(self.updateSliders)
        hl.addWidget(self.zslider)
        self.zrad = QLabel(str(self.radius[0]))
        hl.addWidget(self.zrad)
        vlayout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Y Radius:"))
        self.yslider = QSlider(Qt.Horizontal)
        self.yslider.setMinimum(5)
        self.yslider.setMaximum(75)
        self.yslider.setValue(self.radius[1])
        self.yslider.valueChanged.connect(self.updateSliders)
        hl.addWidget(self.yslider)
        self.yrad = QLabel(str(self.radius[1]))
        hl.addWidget(self.yrad)
        vlayout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("X Radius:"))
        self.xslider = QSlider(Qt.Horizontal)
        self.xslider.setMinimum(5)
        self.xslider.setMaximum(75)
        self.xslider.setValue(self.radius[2])
        self.xslider.valueChanged.connect(self.updateSliders)
        hl.addWidget(self.xslider)
        self.xrad = QLabel(str(self.radius[2]))
        hl.addWidget(self.xrad)
        vlayout.addLayout(hl)

        for i in range(5):
            # Trying to pad things out a bit
            vlayout.addWidget(QLabel(""))

        hlayout.addLayout(vlayout, stretch=1)

        # Table of annotations
        vlayout = QVBoxLayout()
        self.saved_table = QTableWidget(panel)
        labels = [
            "Saved Label ID",
            "Color",
            "Controls",
        ]
        self.saved_table.setColumnCount(len(labels))
        self.saved_table.setHorizontalHeaderLabels(labels)
        header = self.saved_table.horizontalHeader()
        for i in range(len(labels)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        vlayout.addWidget(self.saved_table)
        hlayout.addLayout(vlayout, stretch=3)

        # Table of novel annotations
        vlayout = QVBoxLayout()
        self.new_table = QTableWidget(panel)
        labels = [
            "New Label ID",
            "Color",
            "# Pixels",
            "Mean Signal",
            "Linked Labels",
            "ID to Save As",
            "Split IDs",
            "Save",
            "Split",
        ]
        self.new_table.setColumnCount(len(labels))
        self.new_table.setHorizontalHeaderLabels(labels)
        header = self.new_table.horizontalHeader()
        for i in range(len(labels)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        vlayout.addWidget(self.new_table)
        hlayout.addLayout(vlayout, stretch=3)

        panel.setLayout(hlayout)

        grid.addWidget(panel, 3, 0, 3, self.slices + 1, Qt.AlignmentFlag.AlignTop)

        self.setLayout(grid)

    def checkAutoUpdate(self, checkbox):
        self.update_annotations = checkbox == Qt.Checked
        if self.update_annotations:
            self.drawSlices()


    def updateSliders(self, slider):
        self.radius[0] = self.zslider.value()
        self.radius[1] = self.yslider.value()
        self.radius[2] = self.xslider.value()
        self.zrad.setText(str(self.zslider.value()))
        self.yrad.setText(str(self.yslider.value()))
        self.xrad.setText(str(self.xslider.value()))
        self.drawSlices()


    def setVolumeView(self, volume_view):
        self.volume_view = volume_view
        if volume_view is None:
            return
        for i in range(self.slices):
            for datawindow in [self.depth[i], self.inline[i], self.xline[i]]:
                datawindow.setVolumeView(volume_view)

    def update_saved_label_table(self):
        if self.saved_labels is None:
            return
        idxs = [idx for idx in np.unique(self.saved_labels) if idx > 0]
        self.saved_table.clearContents()
        self.saved_table.setRowCount(len(idxs))
        for row_i, idx in enumerate(idxs):
            #mask = self.saved_labels == idx
            self.saved_table.setItem(row_i, 0, QTableWidgetItem(str(idx)))
            # A cell needs empty text to be able to set the background color
            self.saved_table.setItem(row_i, 1, QTableWidgetItem(""))
            self.saved_table.item(row_i, 1).setBackground(COLORLIST[idx % len(COLORLIST)])
            # Clear button
            button = QPushButton()
            button.setText("Clear")
            button.clicked.connect(self.clear_button_clicked)
            self.saved_table.setCellWidget(row_i, 2, button)

    def update_new_label_table(self):
        if self.new_labels is None:
            return
        linkages = compute_linkages(self.new_labels, self.saved_labels)
        idxs = [idx for idx in np.unique(self.new_labels) if idx != 0]
        self.new_table.clearContents()
        self.new_table.setRowCount(len(idxs))
        for row_i, idx in enumerate(idxs):
            mask = self.new_labels == idx
            self.new_table.setItem(row_i, 0, QTableWidgetItem(str(idx)))
            # A cell needs empty text to be able to set the background color
            self.new_table.setItem(row_i, 1, QTableWidgetItem(""))
            self.new_table.item(row_i, 1).setBackground(COLORLIST[idx % len(COLORLIST)])
            self.new_table.setItem(row_i, 2, QTableWidgetItem(f"{mask.sum():,}"))
            self.new_table.setItem(row_i, 3, 
                                   QTableWidgetItem(f"{int(np.mean(self.signal_section[mask])):,}")
                                                    )
            # Linkages
            links = linkages[idx] if idx in linkages else {}
            self.new_table.setItem(row_i, 4, QTableWidgetItem(str(links)))

            # The label IDs to save to
            line = QLineEdit()
            if len(links) == 1:
                line.setText(str(list(links.keys())[0]))
            self.new_table.setCellWidget(row_i, 5, line)

            # Split IDs
            line = QLineEdit()
            if len(links) == 2:
                ids = list(links.keys())
                line.setText(f"{ids[0]},{ids[1]}")
            self.new_table.setCellWidget(row_i, 6, line)

            # Save button
            button = QPushButton()
            button.setText("Save")
            button.clicked.connect(self.save_button_clicked)
            self.new_table.setCellWidget(row_i, 7, button)

            # Split button
            button = QPushButton()
            button.setText("Split")
            button.clicked.connect(self.split_button_clicked)
            self.new_table.setCellWidget(row_i, 8, button)

    def save_button_clicked(self):
        button = self.sender()
        col_idx = 7
        row_idx = None
        # TODO: this is a terrible hack
        for i in range(self.new_table.rowCount()):
            if self.new_table.cellWidget(i, col_idx) == button:
                row_idx = i
        if row_idx is None:
            print("Button sender not found")
            return
        if self.main_window.annotation is None:
            print("Cannot save data without annotation file loaded")
            return
        label_id = int(self.new_table.item(row_idx, 0).text())
        new_label_id = self.new_table.cellWidget(row_idx, 5).text()
        if not new_label_id:
            print("No saved label provided")
            return
        try:
            new_label_id = int(new_label_id)
        except:
            print("Non-integer saved label provided")
            return
        mask = self.new_labels == label_id
        self.saved_labels[mask] = new_label_id
        vv = self.volume_view
        vol = self.volume_view.volume
        it, jt, kt = vol.transposedIjkToIjk(vv.ijktf, vv.direction)
        islice = slice(
            max(0, it - self.radius[2]), 
            min(vol.data.shape[2], it + self.radius[2] + 1),
            None,
        )
        jslice = slice(
            max(0, jt - self.radius[1]), 
            min(vol.data.shape[1], jt + self.radius[1] + 1),
            None,
        )
        kslice = slice(
            max(0, kt - self.radius[0]), 
            min(vol.data.shape[0], kt + self.radius[0] + 1),
        )
        self.main_window.annotation.write_annotations(self.saved_labels, islice, jslice, kslice)
        self.drawSlices()

    def split_button_clicked(self):
        button = self.sender()
        col_idx = 8
        row_idx = None
        for i in range(self.new_table.rowCount()):
            if self.new_table.cellWidget(i, col_idx) == button:
                row_idx = i
        if row_idx is None:
            print("Button sender not found")
            return
        label_id = int(self.new_table.item(row_idx, 0).text())
        new_label_id = np.max(self.new_labels) + 1
        split_str = self.new_table.cellWidget(row_idx, 6).text()
        if not split_str:
            print("No split labels provided")
            return
        try:
            split1, split2 = tuple([int(s) for s in split_str.split(",")])
        except:
            print("Invalid split labels provided")
            return
        split_label_maxflow(
            self.signal_section,
            self.saved_labels,
            self.new_labels,
            label_id,
            new_label_id,
            split1,
            split2,
        )
        self.update_new_label_table()
        self.drawSlices(update_labels=False)

    def clear_button_clicked(self):
        button = self.sender()
        col_idx = 2
        row_idx = None
        for i in range(self.saved_table.rowCount()):
            if self.saved_table.cellWidget(i, col_idx) == button:
                row_idx = i
        if row_idx is None:
            print("Button sender not found")
            return
        label_id = int(self.saved_table.item(row_idx, 0).text())
        mask = self.saved_labels == label_id
        # Set back to 0 to clear the annotation
        self.saved_labels[mask] = 0
        vv = self.volume_view
        vol = self.volume_view.volume
        it, jt, kt = vol.transposedIjkToIjk(vv.ijktf, vv.direction)
        islice = slice(
            max(0, it - self.radius[2]), 
            min(vol.data.shape[2], it + self.radius[2] + 1),
            None,
        )
        jslice = slice(
            max(0, jt - self.radius[1]), 
            min(vol.data.shape[1], jt + self.radius[1] + 1),
            None,
        )
        kslice = slice(
            max(0, kt - self.radius[0]), 
            min(vol.data.shape[0], kt + self.radius[0] + 1),
        )
        self.main_window.annotation.write_annotations(self.saved_labels, islice, jslice, kslice)
        self.drawSlices()

    def drawSlices(self, update_labels=True):
        vv = self.volume_view
        vol = self.volume_view.volume
        it, jt, kt = vol.transposedIjkToIjk(vv.ijktf, vv.direction)
        islice = slice(
            max(0, it - self.radius[2]), 
            min(vol.data.shape[2], it + self.radius[2] + 1),
            None,
        )
        jslice = slice(
            max(0, jt - self.radius[1]), 
            min(vol.data.shape[1], jt + self.radius[1] + 1),
            None,
        )
        kslice = slice(
            max(0, kt - self.radius[0]), 
            min(vol.data.shape[0], kt + self.radius[0] + 1),
            None,
        )
        if self.update_annotations and update_labels:
            # Pull in saved annotations if available
            if self.main_window.annotation is not None:
                self.saved_labels = self.main_window.annotation.volume[kslice, jslice, islice]
                self.update_saved_label_table()
            else:
                self.saved_labels = None

            # Compute local annotations
            self.signal_section = vol.data[kslice, jslice, islice]
            self.new_labels = find_sheets(self.signal_section, self.saved_labels)
            self.update_new_label_table()

        # Actually draw the datawindows with appropriate shape offsets
        for i in range(self.slices):
            for datawindow, r in zip([self.xline[i], self.depth[i], self.inline[i]], self.radius):
                offsets = [0, 0, 0]
                axis = datawindow.axis
                offsets[axis] += (i - (self.slices // 2)) * ((r * 2) // (self.slices - 1))
                if self.update_annotations:
                    # Add an overlay of the annotations to each view.  First get the center
                    # point for this slice and pull out the labels for this region.
                    ijktf = list(vv.ijktf)
                    ijktf[axis] += offsets[axis]
                    # Coords in absolute space
                    it, jt, kt = vol.transposedIjkToIjk(vv.ijktf, vv.direction)
                    # Coords in local label volume space
                    iidx = list(range(vol.data.shape[2])[islice]).index(it)
                    jidx = list(range(vol.data.shape[1])[jslice]).index(jt)
                    kidx = list(range(vol.data.shape[0])[kslice]).index(kt)
                    slc = vv.getSlice(axis, ijktf)
                    if self.main_window.annotation is not None:
                        overlay = np.copy(self.main_window.annotation.getSlice(vv, axis, ijktf))
                        assert overlay.shape == slc.shape
                    else:
                        overlay = np.zeros_like(slc, dtype=int)
                    # TODO: pasting the overlay data into the center is a hack that 
                    # assumes we're not near the edge of a volume
                    if axis == 1:
                        tmp = (
                            self.new_labels[kidx + offsets[axis], :, :] +
                            self.saved_labels[kidx + offsets[axis], :, :]
                        )
                        x = (overlay.shape[0] - tmp.shape[0]) // 2
                        y = (overlay.shape[1] - tmp.shape[1]) // 2
                        overlay[x:x + tmp.shape[0], y:y + tmp.shape[1]] = tmp
                    elif axis == 2:
                        tmp = (
                            self.new_labels[:, jidx + offsets[axis], :] +
                            self.saved_labels[:, jidx + offsets[axis], :]
                        )
                        x = (overlay.shape[0] - tmp.shape[0]) // 2
                        y = (overlay.shape[1] - tmp.shape[1]) // 2
                        overlay[x:x + tmp.shape[0], y:y + tmp.shape[1]] = tmp
                    elif axis == 0:
                        tmp = (
                            self.new_labels[:, :, iidx + offsets[axis]].T +
                            self.saved_labels[:, :, iidx + offsets[axis]].T
                        )
                        x = (overlay.shape[0] - tmp.shape[0]) // 2
                        y = (overlay.shape[1] - tmp.shape[1]) // 2
                        overlay[x:x + tmp.shape[0], y:y + tmp.shape[1]] = tmp
                else:
                    overlay = None

                datawindow.drawSlice(offsets, crosshairs=False, fragments=False, overlay=overlay)

    def closeEvent(self, event):
        """We need to reset the main window's link to this when 
        the user closes this window.
        """
        print("Closing window")
        self.main_window.annotation_window = None
        event.accept()
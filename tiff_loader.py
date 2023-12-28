import re
from pathlib import Path
import numpy as np
import cv2
from volume import Volume

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

# from data_window import DataWindow, SurfaceWindow
# from project import Project, ProjectView
# from volume import (
#         Fragment, Volume, FragmentsModel, VolumesModel, 
#         DirectionSelectorDelegate,
#         ColorSelectorDelegate)
from utils import Utils

class RangeEdit(QLineEdit):

    def __init__(self, loader, row, col):
        super(RangeEdit, self).__init__()
        self.loader = loader
        self.row = row
        self.col = col
        self.minmax = (0, 0)
        self.lesser = None
        self.greater = None
        self.editingFinished.connect(self.onEditingFinished)
        self.textEdited.connect(self.onTextEdited)
        self.value = -1
        self.valid = False
        self.range_valid = False
        # self.setInputMask('99999')

    def setValue(self, v):
        v = int(round(v))
        txt = "%d"%v
        self.setText(txt)
        self.value = v

    def onEditingFinished(self):
        # print("oef %d,%d"%(self.row,self.col))
        pass

    def onTextEdited(self, txt):
        # print("ote %d,%d %s"%(self.row,self.col,txt))
        v = -1
        valid = True
        range_valid = True
        try:
            v = int(txt)
        except:
            valid = False
        mn, mx = self.minmax
        if v < mn or v > mx:
            valid = False
            range_valid = False
        # if txt=="":
        #     print("blank",v,mn,mx,valid)
        lt = self.lesser
        ltvalid = True
        if range_valid and lt is not None and lt.range_valid and lt.value >= v:
            ltvalid = False
            valid = False
        # if lt is not None:
        #     print("l",v,range_valid,lt.range_valid,lt.value)
        # else:
        #     print("ln",v,range_valid)
        gt = self.greater
        gtvalid = True
        if range_valid and gt is not None and gt.range_valid and gt.value <= v:
            gtvalid = False
            valid = False
        # if gt is not None:
        #     print("g",v,range_valid,gt.range_valid,gt.value)
        # else:
        #     print("gn",v,range_valid)
        if range_valid:
            self.value = v
        else:
            self.value = -1
        if valid:
            # self.setStyleSheet("RangeEdit { color: black }")
            self.setStyleSheet("")
            # self.value = v
        else:
            self.setStyleSheet("RangeEdit { color: red }")
            # self.value = -1

        if range_valid and lt is not None and lt.range_valid:
            # print("ll",ltvalid, lt.valid)
            # if ltvalid and lt.valid:
            if ltvalid:
                # lt.setStyleSheet("RangeEdit { color: black }")
                lt.setStyleSheet("")
            else:
                lt.setStyleSheet("RangeEdit { color: red }")

        if range_valid and gt is not None and gt.range_valid:
            # print("gg",gtvalid, gt.valid)
            # if gtvalid and gt.valid:
            if gtvalid:
                # gt.setStyleSheet("RangeEdit { color: black }")
                gt.setStyleSheet("")
            else:
                gt.setStyleSheet("RangeEdit { color: red }")
        self.valid = valid
        self.range_valid = range_valid
        # print("valid, gtvalid, ltvalid", valid, gtvalid, ltvalid)
        # if lt is not None:
        #     print("lt.valid", lt.valid)
        # if gt is not None:
        #     print("gt.valid", gt.valid)
        self.loader.onChange()
        '''
        print("self",self.row,self.col,self.valid,self.value)
        if lt is not None:
            print("lt",lt.row,lt.col,lt.valid,lt.value)
        if gt is not None:
            print("gt",gt.row,gt.col,gt.valid,gt.value)
        '''

class TiffLoader(QMainWindow):

    def __init__(self, main_window):
        super(TiffLoader, self).__init__(main_window)

        self.main_window = main_window
        # a bit confusing: vc_render is the flag, vcrender is the widget
        self.vc_render = False
        # self.load_as_zarr =  False
        # self.font().setPointSize(20)
        self.setStyleSheet("font-size: 12pt;")
        self.setWindowTitle("TIFF file loader")
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        dirlabel = QLabel("Directory:")
        # dirlabel.setAlignment(Qt.AlignRight)
        hbox.addWidget(dirlabel)
        dirbutton = QPushButton("Select directory containing .tif files")
        dirbutton.clicked.connect(self.onDirButtonClicked)
        self.dirbutton = dirbutton
        hbox.addWidget(dirbutton)
        hbox.addStretch()

        vbox.addLayout(hbox)

        # TODO: need to mention that ranges are inclusive
        grid = QGridLayout()
        minw = 40
        minlabel = QLabel("Min")
        minlabel.setMinimumWidth(minw)
        grid.addWidget(minlabel, 0,1,1,1)
        maxlabel = QLabel("Max")
        maxlabel.setMinimumWidth(minw)
        grid.addWidget(maxlabel, 0,2,1,1)
        steplabel = QLabel("Step")
        steplabel.setMinimumWidth(minw)
        grid.addWidget(steplabel, 0,3,1,1)
        xlabel = QLabel("X")
        xlabel.setAlignment(Qt.AlignRight)
        grid.addWidget(xlabel, 1,0,1,1)
        ylabel = QLabel("Y")
        ylabel.setAlignment(Qt.AlignRight)
        grid.addWidget(ylabel, 2,0,1,1)
        zlabel = QLabel("Z (TIFF file)")
        zlabel.setAlignment(Qt.AlignRight)
        grid.addWidget(zlabel, 3,0,1,1)
        self.range_widgets = []
        for i in range(3): # row
            lrange = []
            for j in range(3): # col
                w = RangeEdit(self, i, j)
                lrange.append(w)
                grid.addWidget(w, i+1,j+1,1,1)
            self.range_widgets.append(lrange)
        vbox.addLayout(grid)
        # todo: enable only when fields are all filled in
        hbox = QHBoxLayout()
        namelabel = QLabel("Volume name:")
        hbox.addWidget(namelabel)
        self.nameedit = QLineEdit()
        # self.nameedit.textEdited.connect(self.onNameEdited)
        self.nameedit.textChanged.connect(self.onNameEdited)
        hbox.addWidget(self.nameedit)
        
        self.vcrender = QCheckBox("TIFFs are from vc_layers")
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
        self.recenter = QPushButton("Re-center view")
        self.recenter.clicked.connect(self.onRecenterClicked)
        self.recenter.setEnabled(False)
        hbox.addWidget(self.recenter)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        self.gobutton = QPushButton("Create")
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

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.directory_valid = False
        self.name_valid = False
        self.name = None
        self.ranges = None
        self.directory = None
        self.filepathdict = None
        self.reading = False
        self.cancel = False

    # override
    def hideEvent(self, e):
        self.onChange()

    # override
    def showEvent(self, e):
        self.onChange()

    #TODO: when shown, make sure a project currently exists.

    # Qt trickery to get around a problem with QFileDialog
    # when the user double-clicks on
    # a directory.  If the directory name ends with ".khprj",
    # we want that directory name to be immediately sent back to
    # the dialog's caller, rather than having the QFileDialog 
    # descend into the selected directory
    def onDirectoryEntered(self, sdir, dialog):
        # print("directory entered", sdir)
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
            dialog.khartes_directory = sdir
            dialog.done(1)
        
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

    def onDirButtonClicked(self, s):
        title = "TIFF loader"
        dialog = QFileDialog(self)
        sdir = self.main_window.settingsGetDirectory("tiff_")
        if sdir is not None and Path(sdir).is_dir():
            # print("setting initial directory to", sdir)
            dialog.setDirectory(sdir)

        dialog.setOptions(QFileDialog.ShowDirsOnly)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setLabelText(QFileDialog.Accept, "Open folder containing TIFF files")
        dialog.directoryEntered.connect(lambda d: self.onDirectoryEntered(d, dialog))
        if not dialog.exec():
            print("No tiff directory selected")
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setIcon(QMessageBox.Information)
            msg.setText("No directory selected")
            msg.exec()
            self.directory_valid = False
            self.onChange()
            return

        file_names = dialog.selectedFiles()
        khartes_directory = getattr(dialog, "khartes_directory", None)
        if khartes_directory is not None:
            file_names = [khartes_directory]
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
            self.directory_valid = False
            self.onChange()
            return
        # print("dirs", sdir, pdir)
        tifs = pdir.glob("*.tif")
        # print(len(tifs))
        inttif = {}
        rec = re.compile(r'[0-9]+')
        for tif in tifs:
            tname = tif.name
            match = rec.search(tname)
            if match is None:
                continue
            ds = match[0]
            itif = int(ds)
            inttif[itif] = tif

        # check if any matching tiff files found
        if len(inttif) == 0:
            print("No TIFF files found in %s"%sdir)
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("No .tif files found in %s"%pdir)
            msg.exec()
            self.directory_valid = False
            self.onChange()
            return

        # print(len(inttif))
        skeys = sorted(inttif.keys())
        inttif = {key: inttif[key] for key in skeys}
        # print(list(inttif.keys())[0])
        # print(list(inttif.keys())[-1])
        # print(list(inttif.values())[0])
        # print(list(inttif.values())[-1])


        self.dirbutton.setText(str(pdir))
        self.pdir = pdir

        minz = skeys[0]
        minp = inttif[minz]
        maxz = skeys[-1]
        iarr = cv2.imread(str(minp), cv2.IMREAD_UNCHANGED)
        if iarr is None:
            print("Could not read tif file %s"%str(minp))
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("TIFF file %s is unreadable"%minp)
            msg.exec()
            self.directory_valid = False
            self.onChange()
            return

        # save now, or save after tif files are read?
        # OK to save now, once directory has been verified
        # as existing and having matching tiffs
        self.main_window.settingsSaveDirectory(str(pdir.parent), "tiff_")

        # look for missing items in skeys
        for i in range(minz, maxz+1):
            if i not in inttif:
                print("TIFF file for image %d is missing"%i)
                msg = QMessageBox()
                msg.setWindowTitle(title)
                msg.setIcon(QMessageBox.Warning)
                msg.setText("TIFF file for image %d is missing"%i)
                msg.setDetailedText("Other files may be missing as well; I haven't checked.\nIf you choose to proceed, make sure your Z min-to-max range does not include any missing files")
                msg.exec()
                break


        maxy = iarr.shape[0]-1
        maxx = iarr.shape[1]-1

        mins = [0,0,minz]
        maxs = [maxx, maxy, maxz]

        for i in range(3):
            mn = mins[i]
            mx = maxs[i]
            self.range_widgets[i][0].setValue(mn)
            self.range_widgets[i][0].minmax = (mn,mx-1)
            self.range_widgets[i][0].greater = self.range_widgets[i][1]
            self.range_widgets[i][0].valid = True
            self.range_widgets[i][0].range_valid = True
            self.range_widgets[i][1].setValue(mx)
            self.range_widgets[i][1].minmax = (mn+1,mx)
            self.range_widgets[i][1].lesser = self.range_widgets[i][0]
            self.range_widgets[i][1].valid = True
            self.range_widgets[i][1].range_valid = True
            self.range_widgets[i][2].setValue(1)
            self.range_widgets[i][2].minmax = (1,mx-mn)
            self.range_widgets[i][2].valid = True
            self.range_widgets[i][2].range_valid = True

        self.directory_valid = True
        self.filepathdict = inttif
        self.directory = pdir
        pv = self.main_window.project_view
        if pv is not None:
            cv = pv.cur_volume
            if cv is not None:
                self.vc_render = cv.from_vc_render
                self.vcrender.setChecked(self.vc_render)
        self.onChange()

    def createRanges(self):
        rw = self.range_widgets
        self.ranges = []
        for i in range(3):
            xmin = rw[i][0].value
            xmax = rw[i][1].value
            step = rw[i][2].value
            rng = [xmin, xmax, step]
            self.ranges.append(rng)

    def setCornerValues(self, ijk, iaxis, jaxis, corner):
        ci = corner%2
        cj = corner//2
        # print(ijk, iaxis, jaxis)

        iar = iaxis
        jar = jaxis
        if self.vc_render:
            axes = (0,2,1)
            iar = axes[iaxis]
            jar = axes[jaxis]
            # print(iaxis, jaxis)

        vi = int(round(ijk[iaxis]))
        vj = int(round(ijk[jaxis]))
        rw = self.range_widgets
        rwi = rw[iar][ci]
        rwj = rw[jar][cj]

        oki = True
        if vi < rwi.minmax[0] or vi > rwi.minmax[1]:
            oki = False
        if rwi.greater is not None and vi >= rwi.greater.value:
            oki = False
        if rwi.lesser is not None and vi <= rwi.lesser.value:
            oki = False

        okj = True
        if vj < rwj.minmax[0] or vj > rwj.minmax[1]:
            okj = False
        if rwj.greater is not None and vj >= rwj.greater.value:
            okj = False
        if rwj.lesser is not None and vj <= rwj.lesser.value:
            okj = False

        if oki:
            rwi.setValue(vi)
        if okj:
            rwj.setValue(vj)
        if oki or okj:
            self.onChange()

    def computeSize(self):
        if not self.areAllRangesValid():
            return 0
        rw = self.range_widgets
        # v = 2 because two bytes per word
        v = 2
        for i in range(3):
            xmin = rw[i][0].value
            xmax = rw[i][1].value
            step = rw[i][2].value
            total = xmax-xmin+1
            n = total//step
            if total%step != 0:
                n += 1
            v *= n
        return v

    def color(self):
        return self.color_editor.getColor()

    def onVcrenderClicked(self, s):
        self.vc_render = self.vcrender.isChecked()
        self.main_window.drawSlices()

    def corners(self):
        if not self.areAllRangesValid():
            return None
        if not self.isVisible():
            return None
        # if self.reading:
        #     return None
        self.createRanges()
        # ranges = [[minx, maxx, dx], [miny, maxy, dy], [minz, maxz, dz]]
        ranges = np.array(self.ranges, dtype=np.int32)
        # rt = [[minx, miny, minz], [maxx, maxy, maxz]]
        rt = ranges.transpose()[0:2]
        # print("rt", rt)
        if self.vc_render:
            rt = rt[:,[0,2,1]]
        return rt


    def onChange(self):
        size = self.computeSize()
        # print("oc size", size)
        if size == 0:
            self.status_bar.showMessage("")
        else:
            gb = size/1000000000
            self.status_bar.showMessage("Volume size: %.1f Gb"%gb)
        if not self.reading:
            self.main_window.drawSlices()
        self.recenter.setEnabled(
                # not self.reading and
                self.corners() is not None
                )
        if self.areValid() and not self.reading:
            self.gobutton.setEnabled(True)
            self.gobutton.setDefault(True)
        else:
            self.gobutton.setEnabled(False)
            self.gobutton.setDefault(False)
        
    def areValid(self):
        # print(self.name_valid, self.directory_valid)
        if not self.name_valid:
            return False
        if not self.directory_valid:
            return False
        return self.areAllRangesValid()

    def areAllRangesValid(self):
        for i in range(3):
            for j in range(3):
                w = self.range_widgets[i][j]
                if not w.valid:
                    return False
        return True

    def onGoButtonClicked(self, s):
        # note that Volume.createFromTiffs checks whether the
        # volume name already exists
        # name = self.nameedit.text()
        # print("volume name is", name)
        # check and warn if size is large
        size = self.computeSize()
        gb = size/1000000000
        title = "TIFF loader"
        if gb > 4:
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Ok|QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Ok)
            msg.setText("The new volume will occupy %.1f Gb of RAM.  Proceed?"%gb)
            msg.setDetailedText("If your computer does not have sufficient memory, this application (or other applications that use the new volume) may crash.\nThis is just a general warning; I have no way of knowing how much memory your computer actually has.")
            answer = msg.exec()
            # print("answer %x"%answer)
            if answer == QMessageBox.Cancel:
                return

        self.cancel = False
        self.reading = True
        self.onChange()

        self.createRanges()
        ranges = self.ranges
        print("ranges", ranges)
        project = self.main_window.project_view.project
        tiff_directory = self.directory
        volume_name = self.name
        filenames = self.filepathdict
        callback = self.readerCallback
        vcrender = self.vc_render

        old_volume = self.main_window.project_view.cur_volume
        # unloads old volume
        self.main_window.setVolume(None)
        new_volume = Volume.createFromTiffs(project, tiff_directory, volume_name, ranges, "", filenames, callback, vcrender)

        self.reading = False
        self.cancel = False
        if new_volume is None or not new_volume.valid:
            print("new volume error", new_volume.error)
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Failed to create new volume from TIFF files: %s"%new_volume.error)
            msg.exec()
            self.main_window.setVolume(old_volume)
        else:
            self.main_window.setVolume(new_volume)
            vv = self.main_window.project_view.cur_volume_view
            vv.setColor(QColor(self.color()))
            # should have been hidden during readerCallback
            self.hide()
        # unset name of volume
        self.nameedit.setText("")
        # setText causes a call to self.onNameEdited which
        # in turn calls self.onChange
        # self.onChange()

    # Callback as specified in Volume.createFromTiffs
    def readerCallback(self, text):
        print("rc", text, end='\r')
        if text.startswith("Loading"):
            self.hide()
        self.status_bar.showMessage(text)
        self.main_window.app.processEvents()
        # True to continue, False to cancel
        if self.cancel:
            return False
        else:
            return True

    def onRecenterClicked(self, s):
        # print("Recenter clicked")
        cs = self.corners()
        avg = .5*(cs[0]+cs[1])
        self.main_window.recenterCurrentVolume(avg)

    def onCancelClicked(self, s):
        if self.reading:
            self.cancel = True
        else:
            self.hide()

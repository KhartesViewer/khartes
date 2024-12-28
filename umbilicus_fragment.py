from fragment import Fragment, FragmentView
import numpy as np
import os
from utils import Utils
import json

from PyQt5.QtWidgets import (
        QDialog, QDialogButtonBox,
        QFileDialog, 
        QGroupBox,
        QMessageBox,
        QVBoxLayout, 
        QRadioButton
        )


class UmbilicusFragment(Fragment):
    def __init__(self, name, direction):
        super(UmbilicusFragment, self).__init__(name, direction)
        self.is_umbilicus = True # Flag to identify umbilicus fragments
        self.type = Fragment.Type.UMBILICUS

    def createView(self, project_view):
        return UmbilicusFragmentView(project_view, self)
        
    def meshExportNeedsInfill(self):
        return False # Umbilicus doesn't need infill since it's just a line
        
    def badTrglsBySkinniness(self, tri, min_roundness):
        # Override to do nothing since we don't use triangulation
        return []
        
    def badTrglsByMaxAngle(self, tri):
        # Override to do nothing since we don't use triangulation 
        return []
        
    def badTrglsByNormal(self, tri, pts):
        # Override to do nothing since we don't use triangulation
        return []
    
    @staticmethod
    def load_umbilicus_from_file(filepath):
        """Load umbilicus points from either .obj or .txt file"""
        if filepath.endswith('.obj'):
            return UmbilicusFragment.load_umbilicus_from_obj(filepath)
        elif filepath.endswith('.txt'):
            return UmbilicusFragment.load_umbilicus_from_txt(filepath)
        else:
            raise ValueError("Unsupported file format. Must be .obj or .txt")

    @staticmethod
    def load_umbilicus_from_obj(filepath):
        """Load points from .obj file, using only vertex positions"""
        points = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):  # vertex line
                    coords = line.split()[1:4]  # get x,y,z coordinates
                    points.append([float(x) for x in coords])
        return np.array(points)

    @staticmethod
    def load_umbilicus_from_txt(filepath):
        """Load points from .txt file with comma-separated x,y,z values"""
        points = []
        with open(filepath, 'r') as f:
            for line in f:
                coords = line.strip().split(',')
                if len(coords) >= 3:  # ensure we have x,y,z
                    points.append([float(x) for x in coords[:3]])
        return np.array(points)

    def toDict(self):
        info = super(UmbilicusFragment, self).toDict()
        info['type'] = self.type.value if self.type else Fragment.Type.UMBILICUS.value
        info['gpoints'] = self.gpoints.tolist()
        return info

class UmbilicusFragmentView(FragmentView):
    def __init__(self, project_view, fragment):
        super(UmbilicusFragmentView, self).__init__(project_view, fragment)
        self.segments = None
        self.manual_points = None  # Store original user-placed points
        self.interpolated_points = None  # Store interpolated points
        
    def interpolatePoints(self):
        """Create linearly interpolated points between manual points"""
        if len(self.manual_points) < 2:
            self.interpolated_points = np.array([])
            return
            
        # Sort manual points by Z
        z_order = np.argsort(self.manual_points[:, 2])
        sorted_points = self.manual_points[z_order]
        
        # Update manual_points to maintain sorted order
        self.manual_points = sorted_points
        
        all_interpolated = []
        
        # Interpolate between each consecutive pair of points
        for i in range(len(sorted_points) - 1):
            p1 = sorted_points[i]
            p2 = sorted_points[i + 1]
            
            # Calculate number of points based on z distance
            z_dist = abs(p2[2] - p1[2])
            num_points = max(2, int(z_dist))  # At least 2 points to include endpoints
            
            # Create evenly spaced points between p1 and p2
            t = np.linspace(0, 1, num_points)[1:-1]  # Exclude endpoints to avoid duplicates
            
            # Linear interpolation for each dimension
            interp_points = np.array([
                p1[0] + (p2[0] - p1[0]) * t,  # x coordinates
                p1[1] + (p2[1] - p1[1]) * t,  # y coordinates
                p1[2] + (p2[2] - p1[2]) * t   # z coordinates
            ]).T
            
            all_interpolated.append(interp_points)
        
        # Combine all interpolated points
        if all_interpolated:
            self.interpolated_points = np.vstack(all_interpolated)
        else:
            self.interpolated_points = np.array([])

    def createZsurf(self, do_update=True):
        # Override to skip triangulation and just sort points by Z
        if len(self.fpoints) <= 1:
            self.zsurf = None
            self.ssurf = None
            self.line = None
            self.segments = None
            return
            
        # Sort points by Z coordinate and break ties using point indices
        # This ensures stable sorting when points have the same Z value
        indices = np.arange(len(self.fpoints))
        sorted_indices = np.lexsort((indices, self.fpoints[:,2]))
        sorted_points = self.fpoints[sorted_indices]
        self.line = sorted_points
        
        # Create line segments between consecutive points
        if len(sorted_points) > 1:
            self.segments = np.array([sorted_points[:-1], sorted_points[1:]]).transpose(1,0,2)
        else:
            self.segments = None
        
        # Skip the rest of createZsurf since we don't need triangulation
        self.zsurf = None 
        self.ssurf = None
        
    def triangulate(self):
        # Override to skip triangulation
        self.tri = None
        if self.fpoints.shape[0] <= 1:
            self.line = None
            self.segments = None
            return
            
        # Sort points by Z and break ties using point indices
        indices = np.arange(len(self.fpoints))
        sorted_indices = np.lexsort((indices, self.fpoints[:,2]))
        sorted_points = self.fpoints[sorted_indices]
        self.line = sorted_points
        return

    def getLinesOnSlice(self, axis, position):
        """Override to return line segments that intersect with the given slice"""
        if self.segments is None or len(self.segments) == 0:
            return None, None
        
        return None, None
            
        # Find segments that cross the slice plane
        crossings = []
        trglist = []
        
        for i, segment in enumerate(self.segments):
            p1, p2 = segment
            
            # Check if segment crosses the slice plane
            if (p1[axis] <= position <= p2[axis]) or (p2[axis] <= position <= p1[axis]):
                # Linear interpolation to find crossing point
                if abs(p2[axis] - p1[axis]) > 1e-10:  # Small threshold for floating point comparison
                    t = (position - p1[axis]) / (p2[axis] - p1[axis])
                    crossing = p1 + t * (p2 - p1)
                    crossings.append([crossing, crossing])  # Duplicate point to create a line segment
                    trglist.append(i)
                else:
                    # Points are at same height, use p1
                    crossings.append([p1, p1])
                    trglist.append(i)
                    
        if len(crossings) == 0:
            return None, None
            
        return np.array(crossings), np.array(trglist)

    def workingTrgls(self):
        """Override to return a boolean array for line segments"""
        if self.segments is None:
            return np.array([], dtype=bool)
        return np.ones(len(self.segments), dtype=bool)

    def trgls(self):
        """Override to return indices for line segments"""
        if self.segments is None:
            return np.array([], dtype=np.int32)
        return np.arange(len(self.segments), dtype=np.int32)

    def hasWorkingNonWorking(self):
        """Override to indicate all segments are working"""
        return (True, False)

    def pushFragmentState(self):
        """Override to do nothing"""
        pass

    def popFragmentState(self):
        """Override to do nothing"""
        pass

    def addPoint(self, tijk, stxy):
        """Override addPoint to handle both manual and interpolated points"""
        print("addPoint Umbilicus", tijk, stxy)
        # Convert to fragment's coordinate system
        fijk = self.vijkToFijk(tijk)
        
        # Round to nearest integer
        ijk = np.rint(np.array(fijk))
        
        # Create new point in global coordinates
        gijk = self.cur_volume_view.transposedIjkToGlobalPosition(tijk)
        
        # Check for and remove any manual points at the same Z value
        if self.manual_points is not None and len(self.manual_points) > 0:
            manual_z_matches = np.where(np.abs(self.manual_points[:, 2] - gijk[2]) < 0.5)[0]
            if len(manual_z_matches) > 0:
                print("z_matches", manual_z_matches)
                self.pushFragmentState()
                self.manual_points = np.delete(self.manual_points, manual_z_matches, 0)
        
        # Add to manual points array
        if self.manual_points is None:
            self.manual_points = np.reshape(gijk, (1,3))
        else:
            self.manual_points = np.append(self.manual_points, np.reshape(gijk, (1,3)), axis=0)
        
        # If we have more than 2 manual points, create interpolated points
        if len(self.manual_points) >= 2:
            self.interpolatePoints()
            if self.interpolated_points is not None and len(self.interpolated_points) > 0:
                # Update fragment points to include both manual and interpolated points
                self.fragment.gpoints = np.vstack((self.manual_points, self.interpolated_points))
        else:
            # Just use manual points if we don't have enough for interpolation
            self.fragment.gpoints = self.manual_points.copy()
        
        self.setLocalPoints(True, False)
        self.fragment.notifyModified()

    def setLocalPoints(self, do_update=True, notify=True):
        """Override to handle manual and interpolated points"""
        super(UmbilicusFragmentView, self).setLocalPoints(do_update, notify)
        
        # Initialize manual points from gpoints if not already set
        if self.manual_points is None and len(self.fragment.gpoints) > 0:
            self.manual_points = self.fragment.gpoints.copy()
            if len(self.manual_points) >= 2:
                self.interpolatePoints()
                if self.interpolated_points is not None and len(self.interpolated_points) > 0:
                    # Update fragment points to include both manual and interpolated points
                    self.fragment.gpoints = np.vstack((self.manual_points, self.interpolated_points))
                    super(UmbilicusFragmentView, self).setLocalPoints(do_update, notify)

    def deletePointByIndex(self, index):
        """Override to handle both manual and interpolated points"""
        if self.manual_points is None or len(self.manual_points) == 0:
            return
        
        # Find if this point is a manual point
        if index < len(self.manual_points):
            # It's a manual point - remove it and reinterpolate
            self.pushFragmentState()
            self.manual_points = np.delete(self.manual_points, index, 0)
            
            # Reinterpolate points if we still have enough manual points
            if len(self.manual_points) >= 2:
                self.interpolatePoints()
                if self.interpolated_points is not None and len(self.interpolated_points) > 0:
                    self.fragment.gpoints = np.vstack((self.manual_points, self.interpolated_points))
                else:
                    self.fragment.gpoints = self.manual_points.copy()
            else:
                # Not enough points for interpolation
                self.interpolated_points = np.array([])
                self.fragment.gpoints = self.manual_points.copy()
            
            self.fragment.notifyModified()
            self.setLocalPoints(True, False)

class UmbilicusExporter:
    """Handles exporting of umbilicus fragments to various file formats"""
    
    def __init__(self, parent_window):
        self.parent = parent_window
        
    def export_fragment(self, fragment, fragment_view):
        """Main export function that handles the export dialog and file saving"""
        # Create format selection dialog
        format_dialog = QDialog(self.parent)
        format_dialog.setWindowTitle("Export Umbilicus Format")
        layout = QVBoxLayout()
        
        # Add radio buttons for coordinate and file format selection
        coord_group = QGroupBox("Coordinate Format")
        coord_layout = QVBoxLayout()
        xyz_radio = QRadioButton("X,Y,Z Format")
        zyx_radio = QRadioButton("Z,Y,X Format")
        xyz_radio.setChecked(True)
        coord_layout.addWidget(xyz_radio)
        coord_layout.addWidget(zyx_radio)
        coord_group.setLayout(coord_layout)
        
        file_group = QGroupBox("File Format")
        file_layout = QVBoxLayout()
        txt_radio = QRadioButton(".txt (comma-separated points)")
        obj_radio = QRadioButton(".obj (vertices and lines)")
        txt_radio.setChecked(True)
        file_layout.addWidget(txt_radio)
        file_layout.addWidget(obj_radio)
        file_group.setLayout(file_layout)
        
        layout.addWidget(coord_group)
        layout.addWidget(file_group)
        
        # Add OK/Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(format_dialog.accept)
        button_box.rejected.connect(format_dialog.reject)
        layout.addWidget(button_box)
        
        format_dialog.setLayout(layout)
        
        # Show dialog and get result
        if format_dialog.exec_() != QDialog.Accepted:
            return
            
        # Get file path from user
        file_filter = "Text Files (*.txt)" if txt_radio.isChecked() else "OBJ Files (*.obj)"
        filename_tuple = QFileDialog.getSaveFileName(self.parent, "Export Umbilicus", "", file_filter)
        if filename_tuple[0] == "":
            return
            
        try:
            # Get manual points (already sorted by Z)
            points = fragment_view.manual_points
            if points is None or len(points) < 2:
                raise ValueError("No manual points found to export")
            
            # Transform coordinates if needed
            if zyx_radio.isChecked():
                points = points[:, [2, 1, 0]]  # Convert X,Y,Z to Z,Y,X
                
            # Export based on selected format
            filepath = filename_tuple[0]
            if txt_radio.isChecked():
                self._export_txt(points, filepath)
            else:
                self._export_obj(points, filepath)
                
            QMessageBox.information(self.parent, "Export Complete", 
                "Umbilicus manual points exported successfully.")
            
        except Exception as e:
            QMessageBox.warning(self.parent, "Export Error", str(e))
            
    def _export_txt(self, points, filepath):
        """Export points as comma-separated values"""
        with open(filepath, 'w') as f:
            for point in points:
                f.write(f"{point[0]},{point[1]},{point[2]}\n")
                
    def _export_obj(self, points, filepath):
        """Export points as OBJ with vertices and lines"""
        with open(filepath, 'w') as f:
            # Write vertices
            for point in points:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
            # Write lines connecting consecutive points
            for i in range(len(points)-1):
                f.write(f"l {i+1} {i+2}\n")

class UmbilicusImporter:
    """Handles importing of umbilicus fragments from various file formats"""
    
    def __init__(self, parent_window):
        self.parent = parent_window
        
    def import_file(self):
        """Main import function that handles the import dialog and file loading"""
        # Get file path from user
        filepath, _ = QFileDialog.getOpenFileName(
            self.parent, 'Import Umbilicus File',
            '', 'Umbilicus Files (*.obj *.txt)')
            
        if not filepath:
            return None
            
        # Create format selection dialog
        format_dialog = QDialog(self.parent)
        format_dialog.setWindowTitle("Select Coordinate Format")
        layout = QVBoxLayout()
        
        # Add radio buttons for format selection
        xyz_radio = QRadioButton("X,Y,Z Format")
        zyx_radio = QRadioButton("Z,Y,X Format")
        xyz_radio.setChecked(True)  # Default to X,Y,Z
        
        layout.addWidget(xyz_radio)
        layout.addWidget(zyx_radio)
        
        # Add OK/Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(format_dialog.accept)
        button_box.rejected.connect(format_dialog.reject)
        layout.addWidget(button_box)
        
        format_dialog.setLayout(layout)
        
        # Show dialog and get result
        if format_dialog.exec_() != QDialog.Accepted:
            return None
            
        try:
            # Load points from file
            points = UmbilicusFragment.load_umbilicus_from_file(filepath)
            
            if len(points) < 2:
                QMessageBox.warning(self.parent, 'Import Error', 
                    'File must contain at least 2 points')
                return None
            
            # Transform coordinates if needed
            if zyx_radio.isChecked():
                # Convert from Z,Y,X to X,Y,Z format
                points = points[:, [2, 1, 0]]  # Reorder columns
                
            # Create new umbilicus fragment
            basename = os.path.splitext(os.path.basename(filepath))[0]
            fragment = UmbilicusFragment(basename, direction=1)
            
            # Set random color
            fragment.setColor(Utils.getNextColor(), no_notify=True)
            fragment.valid = True
            
            # Set points in global coordinates
            fragment.gpoints = points
            
            return fragment
            
        except Exception as e:
            QMessageBox.warning(self.parent, 'Import Error', str(e))
            return None

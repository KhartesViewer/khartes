import pathlib
import shutil
import time
import json
from utils import Utils
from volume import Volume, VolumeView
from fragment import Fragment, FragmentView
from trgl_fragment import TrglFragment, TrglFragmentView
from base_fragment import BaseFragment, BaseFragmentView
from PyQt5.QtGui import QColor 



class ProjectView:

    def __init__(self, project):
        self.project = project
        self.valid = False
        if not project.valid:
            return

        self.volumes = {}
        for volume in project.volumes:
            self.addVolumeView(volume, no_notify=True)

        self.fragments = {}
        for fragment in project.fragments:
            self.addFragmentView(fragment, no_notify=True)

        self.cur_volume = None
        self.cur_volume_view = None
        self.nearby_node_index = -1
        self.nearby_node_fv = None
        project.project_views.append(self)
        '''
        self.settings = {}
        self.settings['fragment'] = {}
        self.settings['slices'] = {}
        self.settings['fragment']['nodes_visible'] = True
        self.settings['fragment']['triangles_visible'] = True
        self.settings['slices']['vol_boxes_visible'] = False
        '''
        self.vol_boxes_visible = False

    def alphabetizeVolumeViews(self):
        vols = list(self.volumes.keys())
        Volume.sortVolumeList(vols)
        new_vols = {}
        for vol in vols:
            new_vols[vol] = self.volumes[vol]
        self.volumes = new_vols

    def addVolumeView(self, volume, no_notify=False):
        if volume not in self.volumes:
            self.volumes[volume] = VolumeView(self, volume)
        if not no_notify:
            self.notifyModified()

    def alphabetizeFragmentViews(self):
        frags = list(self.fragments.keys())
        Fragment.sortFragmentList(frags)
        new_frags = {}
        for frag in frags:
            new_frags[frag] = self.fragments[frag]
        self.fragments = new_frags

    def addFragmentView(self, fragment, no_notify=False):
        if fragment not in self.fragments:
            # self.fragments[fragment] = FragmentView(self, fragment)
            self.fragments[fragment] = fragment.createView(self)
        if not no_notify:
            self.notifyModified()

    # "id" is fragment's "created" attribute
    # returns None if nothing found
    def findFragmentViewById(self, fid):
        for f,fv in self.fragments.elements():
            if f.created == fid:
                return fv;
        return None

    def notifyModified(self, tstamp=""):
        if tstamp == "":
            tstamp = Utils.timestamp()
        self.modified = tstamp
        # print("project view modified", tstamp)
        self.project.notifyModified(tstamp)

    def save(self):
        print("called project_view save")
        self.project.save()

        info = {}
        prj = {}
        if self.cur_volume is not None:
            prj['cur_volume'] = self.cur_volume.name
        prj['vol_boxes_visible'] = self.vol_boxes_visible
        info['project'] = prj

        vvs = {}
        for vol in self.volumes.values():
            vv = {}
            vv['direction'] = vol.direction
            vv['zoom'] = vol.zoom
            vv['ijktf'] = list(vol.ijktf)
            vv['color'] = vol.color.name()
            vvs[vol.volume.name] = vv
        info['volumes'] = vvs

        fvs = {}
        for frag in self.fragments.values():
            fv = {}
            fv['visible'] = frag.visible
            fv['active'] = frag.active
            fvs[frag.fragment.created] = fv
        info['fragments'] = fvs

        # info_txt = json.dumps(info, sort_keys=True, indent=4)
        info_txt = json.dumps(info, indent=4)
        (self.project.path / 'views.json').write_text(info_txt, encoding="utf8")

    def createErrorProjectView(project, err):
        pv = ProjectView(project)
        pv.error = err
        return pv

    def open(fullpath):
        project = Project.open(fullpath)
        if not project.valid:
            err = "Error creating project: %s"%project.error
            print(err)
            pv = ProjectView.createErrorProjectView(project, err)
            return pv
        pv = ProjectView(project)
        pv.valid = True
        info_file = (project.path / 'views.json')
        try:
            info_txt = info_file.read_text(encoding="utf8")
        except:
            err = "Could not read file %s"%info_file
            print(err)
            # Not a fatal error
            return pv

        try:
            info = json.loads(info_txt)
        except:
            err = "Could not parse file %s"%info_file
            print(err)
            return pv

        if 'volumes' in info:
            vinfos = info['volumes']
            for vv in pv.volumes.values():
                name = vv.volume.name
                # print("parsing vv info for %s"%name)
                if name not in vinfos:
                    continue
                vinfo = vinfos[name]
                if 'direction' in vinfo:
                    vv.direction = vinfo['direction']
                if 'zoom' in vinfo:
                    vv.zoom = vinfo['zoom']
                if 'ijktf' in vinfo:
                    vv.ijktf = vinfo['ijktf']
                if 'color' in vinfo:
                    vv.setColor(QColor(vinfo['color']), no_notify=True)
                # else:
                # this else clause is not needed because VolumeView
                # creator sets a random color
                #     vv.setColor(Utils.getNextColor())
                # print("vv info", vv.direction,vv.zoom,vv.ijktf)

        if 'fragments' in info:
            finfos = info['fragments']
            # print("finfos", finfos)
            for fv in pv.fragments.values():
                name = fv.fragment.name
                created = fv.fragment.created
                print(name, created)
                finfo = None
                if name in finfos:
                    finfo = finfos[name]
                    # print("found by name")
                elif created in finfos:
                    finfo = finfos[created]
                    # print("found by created")
                if finfo is None:
                    # print("not found")
                    continue
                if 'visible' in finfo:
                    fv.visible = finfo['visible']
                if 'active' in finfo:
                    fv.active = finfo['active']

        if 'project' in info:
            pinfo = info['project']
            if 'cur_volume' in pinfo:
                cvname = pinfo['cur_volume']
                # print("cv name", cvname)
                for vol in pv.volumes.keys():
                    # print(" vol name", vol.name)
                    if vol.name == cvname:
                        pv.setCurrentVolume(vol, no_notify=True)
                        # print("set cur vol")
                        break
            if 'cur_fragment' in pinfo:
                cfname = pinfo['cur_fragment']
                for frag in pv.fragments.keys():
                    if frag.name == cfname:
                        pv.fragments[frag].active = True
                        break
            if 'vol_boxes_visible' in pinfo:
                pv.vol_boxes_visible = pinfo['vol_boxes_visible']

        return pv


    def updateFragmentViews(self):
        for fv in self.fragments.values():
            # print("svv")
            fv.setVolumeView(self.cur_volume_view)
        # make sure echo fragments are updated
        for fv in self.fragments.values():
            # print("slp")
            fv.setLocalPoints(True)


    def setCurrentVolume(self, volume, no_notify=False):
        if self.cur_volume != volume:
            if self.cur_volume is not None:
                self.cur_volume.unloadData(self)
            if self.cur_volume_view is not None:
                self.cur_volume_view.trdata = None
            if volume is not None:
                volume.loadData(self)
        self.cur_volume = volume
        if volume is None:
            self.cur_volume_view = None
        else:
            self.cur_volume_view = self.volumes[volume]
            self.cur_volume_view.dataLoaded()
        if not no_notify:
            self.notifyModified()

    def setDirection(self, volume, direction):
        if volume == self.cur_volume:
            self.setDirectionOfCurrentVolume(direction)
        else:
            volume_view = self.volumes[volume]
            volume_view.setDirection(direction)

    def setDirectionOfCurrentVolume(self, direction):
        if self.cur_volume_view is None:
            print("Warning, setDirectionOfCurrentVolume: no current volume")
            return
        self.cur_volume_view.setDirection(direction)
        for fv in self.fragments.values():
            fv.setVolumeViewDirection(direction)

    def clearActiveFragmentViews(self):
        for fv in self.fragments.values():
            fv.active = False
            fv.notifyModified()

    def mainActiveVisibleFragmentView(self, unaligned_ok=False):
        last = None
        for fv in self.fragments.values():
            if fv.visible:
                if fv.activeAndAligned():
                    last = fv
                elif fv.active and unaligned_ok:
                    last = fv
        return last

    # Prefer to return last visible active fragment, but
    # if nothing is both active and visible, return last
    # active fragment
    def mainActiveFragmentView(self, unaligned_ok=False):
        last_visible = None
        last = None
        for fv in self.fragments.values():
            if fv.visible:
                if fv.activeAndAligned():
                    last_visible = fv
                    last = fv
                elif fv.active and unaligned_ok:
                    last_visible = fv
                    last = fv
            else:
                if fv.activeAndAligned():
                    last = fv
                elif fv.active and unaligned_ok:
                    last = fv
        if last_visible is not None:
            return last_visible
        return last

    def activeFragmentViews(self, unaligned_ok=False):
        fvs = []
        for fv in self.fragments.values():
            if fv.activeAndAligned():
                fvs.append(fv)
            elif fv.active and unaligned_ok:
                fvs.append(fv)
        return fvs


class Project:

    suffix = ".khprj"

    info_parameters = ["created", "modified", "name", "version", "voxel_size_um"]
    default_voxel_size_um = 7.91


    def __init__(self):
        self.volumes = []
        self.fragments = []
        self.project_views = []
        self.voxel_size_um = Project.default_voxel_size_um
        self.valid = False
        self.error = "no error message set"
        self.modified_callback = None
        self.last_saved = ""

    def createErrorProject(err):
        prj = Project()
        prj.error = err
        return prj

    def create(fullpath, pname=None):
        fp = pathlib.Path(fullpath)
        # print(fullpath, fp)
        # parent = fp.resolve()
        # print(parent)
        parent = fp.resolve().parent
        # print(fullpath, fp, parent)
        if parent is not None and not parent.is_dir():
            err = "Directory %s does not exist or is not a directory"%parent
            print(err)
            return Project.createErrorProject(err)
        name = fp.name
        suffix = fp.suffix
        # print(name, suffix)
        if suffix != Project.suffix:
            name += Project.suffix
            fp = fp.with_name(name)
        # print(fp)
        if not pname:
            pname = fp.stem

        if fp.exists():
            if fp.is_dir():
                try:
                    shutil.rmtree(fp)
                    # print("%s directory removed"%fp)
                except:
                    err = "Could not delete existing directory %s"%fp
                    print(err)
                    return Project.createErrorProject(err)
            else:
                try:
                    fp.unlink()
                    # print("%s unlinked"%fp)
                except:
                    err = "Could not delete existing file %s"%fullpath
                    print(err)
                    return Project.createErrorProject(err)
        try:
            # print("about to sleep")
            # When over-writing an existing project, 
            # this error message is printed during the sleep: QFileSystemWatcher: FindNextChangeNotification failed for "[filename]"  (Access is denied.)
            # If the sleep is not here, the mkdir fails in this case.
            time.sleep(1)
            # print("creating", str(fp))
            fp.mkdir()
            # print("created", str(fp))
        except:
            err = "Could not create new directory %s"%fullpath
            print(err)
            return Project.createErrorProject(err)
        vdir = fp / 'volumes'
        vdir.mkdir()
        fdir = fp / 'fragments'
        fdir.mkdir()
        prj = Project()
        prj.volumes = []
        prj.fragments = []
        prj.valid = True
        prj.path = fp
        prj.name = pname
        prj.created = Utils.timestamp()
        prj.modified = prj.created
        prj.version = 1.0
        prj.volumes_path = vdir
        prj.fragments_path = fdir
        info = {}
        for param in Project.info_parameters:
            info[param] = getattr(prj, param)
        info_txt = json.dumps(info, sort_keys=True, indent=4)
        (fp / 'project.json').write_text(info_txt, encoding="utf8")
        return prj

    def save(self):
        print("called project save")
        frag_path = self.fragments_path
        frag_path_old = self.fragments_path.with_name('fragments.old')
        frag_path_older = self.fragments_path.with_name('fragments.older')
        timestamp = Utils.timestampToVc(Utils.timestamp())
        if frag_path_older.exists() and frag_path_older.is_dir():
            files = list(frag_path_older.glob('*'))
            for file in files:
                try:
                    file.unlink()
                except Exception as e:
                    print(e)
                    print("failed to unlink",file,"in",frag_path_older.name)
            try:
                frag_path_older.rmdir()
            except Exception as e:
                print(e)
                print("failed to rmdir", frag_path_older.name)
        if frag_path_older.exists():
            try:
                newname = "fragments.older."+timestamp
                newpath = frag_path.with_name(newname)
                frag_path_older.rename(newpath)
            except Exception as e:
                print(e)
                print("failed to rename",frag_path_older.name,"to",newname)
        if frag_path_older.exists():
            try:
                newname = "fragments.old."+timestamp
                newpath = frag_path.with_name(newname)
                frag_path_old.rename(newpath)
            except Exception as e:
                print(e)
                print("failed to rename",frag_path_old.name,"to",newname)
        if frag_path_old.exists():
            try:
                newname = "fragments.older"
                newpath = frag_path.with_name(newname)
                frag_path_old.rename(newpath)
            except Exception as e:
                print(e)
                print("failed to rename",frag_path_old.name,"to",newname)
                pass
        if frag_path_old.exists():
            try:
                newname = "fragments.old."+timestamp
                newpath = frag_path.with_name(newname)
                frag_path_old.rename(newpath)
            except Exception as e:
                print(e)
                print("failed to rename",frag_path_old.name,"to",newname)
        if frag_path_old.exists():
            try:
                newname = "fragments."+timestamp
                newpath = frag_path.with_name(newname)
                frag_path.rename(newpath)
            except Exception as e:
                print(e)
                print("failed to rename",frag_path.name,"to",newname)
        else:
            try:
                newname = "fragments.old"
                newpath = frag_path.with_name(newname)
                frag_path.rename(newpath)
            except Exception as e:
                print(e)
                print("failed to rename",frag_path.name,"to",newname)
        if frag_path.exists():
            files = list(frag_path.glob('*'))
            for file in files:
                try:
                    file.unlink()
                except:
                    print("failed to unlink",file,"in",frag_path.name)
            
        else:
            frag_path.mkdir()
        '''
        files = list(self.fragments_path.glob("*.json"))
        # print("glob files", files)
        for file in files:
            older = file.with_name(file.name+".older")
            old = file.with_name(file.name+".old")
            older.unlink(missing_ok=True)
            if old.exists():
                old.rename(older)
            file.rename(old)
        # Fragment.saveList(self.fragments, self.fragments_path, "all")
        '''
        BaseFragment.saveList(self.fragments, self.fragments_path, "all")

        info = {}
        # TODO: set modified-date in info
        for param in Project.info_parameters:
            info[param] = getattr(self, param)
        info_txt = json.dumps(info, sort_keys=True, indent=4)
        (self.path / 'project.json').write_text(info_txt, encoding="utf8")
        self.last_saved = Utils.timestamp()

    notify_counter = 0

    def notifyModified(self, tstamp=""):
        if tstamp == "":
            tstamp = Utils.timestamp()
        self.modified = tstamp
        # print("project modified", tstamp)
        # if Project.notify_counter >= 0:
        #     # intentionally cause a crash, to examine the stack trace
        #     print(asdf)
        if self.modified_callback is not None:
            self.modified_callback(self)
        Project.notify_counter += 1

    def open(fullpath):
        fp = pathlib.Path(fullpath)
        if not fp.is_dir():
            err = "Directory %s does not exist"%fullpath
            print(err)
            return Project.createErrorProject(err)

        vdir = fp / 'volumes'
        if not vdir.is_dir():
            err = "Directory %s does not exist"%vdir
            print(err)
            return Project.createErrorProject(err)

        fdir = fp / 'fragments'
        if not fdir.is_dir():
            err = "Directory %s does not exist"%fdir
            print(err)
            return Project.createErrorProject(err)

        info_file = fp / 'project.json'
        if not info_file.is_file():
            err = "File %s does not exist or an object of that name exists but is not a file"%info_file
            print(err)
            return Project.createErrorProject(err)

        try:
            info_txt = info_file.read_text(encoding="utf8")
        except:
            err = "Could not read file %s"%info_file
            print(err)
            return Project.createErrorProject(err)

        try:
            info = json.loads(info_txt)
        except:
            err = "Could not parse file %s"%info_file
            print(err)
            return Project.createErrorProject(err)

        prj = Project()
        prj.volumes = []
        prj.fragments = []
        prj.valid = True
        prj.path = fp
        prj.volumes_path = vdir
        prj.fragments_path = fdir

        for param in Project.info_parameters:
            if param not in info:
                if param == "voxel_size_um":
                    continue
                err = "project info file is missing parameter '%s'"%param
                print(err)
                return Project.createErrorProject(err)
            setattr(prj, param, info[param])

        prj.last_saved = prj.modified

        for vfile in vdir.glob("*.nrrd"):
            vol = Volume.loadNRRD(vfile)
            if vol is not None and vol.valid:
                prj.addVolume(vol)

        for ffile in fdir.glob("*.json"):
            frags = Fragment.load(ffile)
            if frags is not None:
                for frag in frags:
                    if frag.valid:
                        prj.addFragment(frag)

        for ffile in fdir.glob("*.obj"):
            frags = TrglFragment.load(ffile)
            if frags is not None:
                for frag in frags:
                    if frag.valid:
                        prj.addFragment(frag)

        return prj

    def isSaveUpToDate(self):
        # print("ls",self.last_saved,"m",self.modified)
        return (self.last_saved >= self.modified)

    def alphabetizeVolumes(self):
        Volume.sortVolumeList(self.volumes)
        for pv in self.project_views:
            pv.alphabetizeVolumeViews()

    def addVolume(self, volume):
        volume.setVoxelSizeUm(self.voxel_size_um)
        self.volumes.append(volume)
        for pv in self.project_views:
            pv.addVolumeView(volume)
        self.alphabetizeVolumes()

    def alphabetizeFragments(self):
        Fragment.sortFragmentList(self.fragments)
        for pv in self.project_views:
            pv.alphabetizeFragmentViews()

    def addFragment(self, fragment):
        fragment.project = self
        self.fragments.append(fragment)
        for pv in self.project_views:
            pv.addFragmentView(fragment)
        self.alphabetizeFragments()

    # "id" is fragment's "created" attribute
    # returns None if nothing found
    def findFragmentById(self, fid):
        for frag in self.fragments:
            if frag.created == fid:
                return frag;
        return None

    def getVoxelSizeUm(self):
        return self.voxel_size_um

    def setVoxelSizeUm(self, vs):
        self.voxel_size_um = vs
        for v in self.volumes:
            v.setVoxelSizeUm(vs)
        self.notifyModified()


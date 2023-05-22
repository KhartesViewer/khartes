import pathlib
import shutil
import time
import json
from utils import Utils
from volume import Volume, VolumeView
from volume import Fragment, FragmentView
from PyQt5.QtGui import QColor 



class ProjectView:

    def __init__(self, project):
        self.project = project
        self.valid = False
        if not project.valid:
            return

        self.volumes = {}
        for volume in project.volumes:
            # self.volumes[volume] = VolumeView(self, volume)
            self.addVolumeView(volume)

        self.fragments = {}
        for fragment in project.fragments:
            # self.fragments[fragment] = FragmentView(self, fragment)
            self.addFragmentView(fragment)

        self.cur_volume = None
        self.cur_volume_view = None
        self.cur_fragment = None
        self.cur_fragment_view = None
        project.project_views.append(self)
        self.settings = {}
        self.settings['fragment'] = {}
        self.settings['slices'] = {}
        self.settings['fragment']['nodes_visible'] = True
        self.settings['fragment']['triangles_visible'] = True
        self.settings['slices']['vol_boxes_visible'] = False

    def addVolumeView(self, volume):
        if volume not in self.volumes:
            self.volumes[volume] = VolumeView(self, volume)

    def addFragmentView(self, fragment):
        if fragment not in self.fragments:
            self.fragments[fragment] = FragmentView(self, fragment)


    def save(self):
        print("called project_view save")
        self.project.save()

        info = {}
        prj = {}
        if self.cur_fragment is not None:
            prj['cur_fragment'] = self.cur_fragment.name
        if self.cur_volume is not None:
            prj['cur_volume'] = self.cur_volume.name
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
            fvs[frag.fragment.name] = fv
        info['fragments'] = fvs

        info_txt = json.dumps(info, sort_keys=True, indent=4)
        (self.project.path / 'views.json').write_text(info_txt, encoding="utf8")

    def createErrorProjectView(project, err):
        pv = ProjectView(project)
        pv.error = err
        return prj

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
                    vv.setColor(QColor(vinfo['color']))
                # else:
                # this else clause is not needed because VolumeView
                # creator sets a random color
                #     vv.setColor(Utils.getNextColor())
                # print("vv info", vv.direction,vv.zoom,vv.ijktf)

        if 'fragments' in info:
            finfos = info['fragments']
            for fv in pv.fragments.values():
                name = fv.fragment.name
                if name not in finfos:
                    continue
                finfo = finfos[name]
                if 'visible' in finfo:
                    fv.visible = finfo['visible']

        if 'project' in info:
            pinfo = info['project']
            if 'cur_volume' in pinfo:
                cvname = pinfo['cur_volume']
                # print("cv name", cvname)
                for vol in pv.volumes.keys():
                    # print(" vol name", vol.name)
                    if vol.name == cvname:
                        pv.setCurrentVolume(vol)
                        # print("set cur vol")
                        break
            if 'cur_fragment' in pinfo:
                cfname = pinfo['cur_fragment']
                for frag in pv.fragments.keys():
                    if frag.name == cfname:
                        pv.setCurrentFragment(frag)
                        break

        return pv


    def updateFragmentViews(self):
        for fv in self.fragments.values():
            fv.setVolumeView(self.cur_volume_view)
        # make sure echo fragments are updated
        for fv in self.fragments.values():
            fv.setLocalPoints(True)


    '''
    def update(self):
        # self.volumes = {}
        cur_volumess = set()
        for volume in project.volumes:
            cur_volumes.add(volume)
            if volume not in self.volumes:
                self.volumes[volume] = VolumeView(self, volume)
            # vv = self.volumes[volume]

        nv = {vol, self.volumes[vol] for vol in cur_volumes}
        self.volumes = nv

        self.fragments = {}
        cur_fragments = set()
        for fragment in project.fragments:
            cur_fragments.add(fragment)
            if fragment not in self.fragments:
                self.fragments[fragment] = FragmentView(self, fragment)
            fv = self.fragments[fragment]

        nf = {frag, self.fragments[frag] for frag in cur_fragments}
        self.fragments = nf
    '''


    def setCurrentVolume(self, volume):
        # cur_frag = None
        # if self.cur_volume is not None:
        #     cur_frag = self.cur_fragment
        if self.cur_volume != volume:
            if self.cur_volume is not None:
                self.cur_volume.unloadData(self)
            if volume is not None:
                volume.loadData(self)
        # if self.volume is not None and self.volume != volume:
            # vol = self.volume
            # unloadData checks if any ProjectViews still refer
            # to the volume, so need to remove reference before
            # calling unloadData
            # self.volume = None
            # vol.unloadData()
        # if volume is not None and self.volume != volume:
        #     volume.loadData()
        self.cur_volume = volume
        if volume is None:
            self.cur_volume_view = None
        else:
            self.cur_volume_view = self.volumes[volume]
            self.cur_volume_view.dataLoaded()
        # self.direction = direction
        # make sure volume is in volumes
        # save direction in volume view

        # will unset cur_fragment if direction is not compatible
        self.setCurrentFragment(self.cur_fragment)

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
        self.setCurrentFragment(None)

    def setCurrentFragment(self, fragment):
        self.cur_fragment = fragment
        if fragment is None:
            self.cur_fragment_view = None
        else:
            fv = self.fragments[fragment]
            cvv = self.cur_volume_view
            if cvv is None or cvv.direction != fv.fragment.direction:
                print("cannot set current fragment")
                self.cur_fragment_view = None
            else:
                # print("%s set as cur fragment"%fv.fragment.name)
                self.cur_fragment_view = fv



class Project:

    suffix = ".khprj"

    info_parameters = ["created", "modified", "name", "version"]


    def __init__(self):
        self.volumes = []
        self.fragments = []
        self.project_views = []
        self.valid = False
        self.error = "no error message set"

    '''
    def timestamp():
        t = time.gmtime()
        txt = time.strftime("%Y-%m-%dT%H:%M:%SZ", t)
        return txt
    '''

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
            fp.mkdir()
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
        for frag in self.fragments:
            print("saving frag", frag.name)
            frag.save(self.fragments_path)

        info = {}
        # TODO: set modified-date in info
        for param in Project.info_parameters:
            info[param] = getattr(self, param)
        info_txt = json.dumps(info, sort_keys=True, indent=4)
        (self.path / 'project.json').write_text(info_txt, encoding="utf8")


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
                err = "project info file is missing parameter '%s'"%param
                print(err)
                return Project.createErrorProject(err)
            setattr(prj, param, info[param])

        for vfile in vdir.glob("*.nrrd"):
            vol = Volume.loadNRRD(vfile)
            if vol is not None and vol.valid:
                prj.addVolume(vol)

        for ffile in fdir.glob("*.json"):
            frag = Fragment.load(ffile)
            if frag is not None and frag.valid:
                prj.addFragment(frag)

        return prj


    def addVolume(self, volume):
        self.volumes.append(volume)
        for pv in self.project_views:
            pv.addVolumeView(volume)

    def addFragment(self, fragment):
        self.fragments.append(fragment)
        for pv in self.project_views:
            pv.addFragmentView(fragment)


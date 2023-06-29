
'''
Take thee again another scroll, and write in it all the former words that 
were in the first scroll, which Jehoiakim the king of Judah hath burned.
Jeremiah 36:28
'''

import sys
from PyQt5.QtWidgets import QApplication

from main_window import MainWindow
from volume import Volume
from fragment import Fragment
from project import Project
# from utils import Utils

class Khartes():

    def __init__(self, app):
        window = MainWindow("χάρτης", app)
        self.app = app
        self.window = window
        # window.new_project_action.triggered.connect(self.newProjectButtonClick)
        window.show()


    '''
    def newProjectButtonClick(self, s):
        print("khartes new project button clicked")
        project = testReadProject()
        volume = project.volumes[0]
        print("khartes volume sizes", volume.sizes)
        # volume.setDirection(1)
        # volume.setDefaultParameters(self.window)
        self.window.setProject(project)
        self.window.setVolume(volume)
        # frag = Fragment("frag01", 1)
        # project.addFragment(frag)
        # frags = [frag]
        # self.window.setFragments(frags)
        # always call this after adding/removing a fragment
        # self.window.setFragments()
        # self.window.setCurrentFragment(frag)
    '''

    def loadVolume(self, name):
        # name in function argument is ignored!
        base = 'H:\\Vesuvius\\scroll1\\part6000-7250\\windowed\\'

        file = base+'win1h.nrrd'
        direction = 1

        file = base+'win2h.nrrd'
        direction = 0

        file = base+'win3h.nrrd'
        direction = 1

        volume = Volume.loadNRRD(file)
        if volume is None:
            return
        self.volume = volume
        window = self.window
        volume.setDirection(direction)
        volume.setDefaultParameters(window)
        window.setWindowsVolume(volume)
        # frag = Fragment(volume)
        # frags = [frag]
        # window.setWindowsFragments(frags, frag)
        window.drawSlices()


def testCreateProject():
    project = Project.create('H:\\Vesuvius\\try2w')
    if not project.valid:
        print("Valid project not created: %s"%project.error)
        exit()
    return project
    # project = Project.create('.\\try1.khprj')

'''
def testReadProject():
    # new_prj = Project.open('H:\\Vesuvius\\try1.khprj')
    new_prj = Project.open('H:\\Vesuvius\\try2.khprj')
    if not new_prj.valid:
        print("Valid project not opened: %s"%new_prj.error)
        exit()
    return new_prj
'''

def testCreateVolumes(prj):
    tiff_dir = r'H:\Vesuvius\scroll1\part6000-7250\orig'
    '''
    vnames = ['local1', 'local2', 'coarse']
    zmin = 6000
    zmax = 6100
    rangess = [
            [[3500,4500,1], [2500,3000,1], [zmin,zmax,1]],
            [[3000,4000,1], [2750,3250,1], [zmin,zmax,1]],
            [[3000,5000,2], [2500,3500,2], [zmin,zmax,2]],
            ]
    '''
    '''
    vnames = ['left', 'middle', 'right', 'all']
    zmin = 6000
    zmax = 6500
    rangess = [
            [[2500,3500,1], [3000,5000,1], [zmin,zmax,1]],
            [[3000,5000,1], [4000,5000,1], [zmin,zmax,1]],
            [[4500,5500,1], [3000,5000,1], [zmin,zmax,1]],
            [[1000,7000,8], [3000,5000,8], [zmin,zmax,8]],
            ]
    '''

    vnames = ['left', 'middle', 'right', 'all']
    zmin = 6000
    zmax = 6010
    # zmax = 6500
    # zmax = 7000
    rangess = [
            [[2000,3000,1], [2750,4750,1], [zmin,zmax,1]],
            [[2500,4500,1], [3750,4750,1], [zmin,zmax,1]],
            [[4000,5000,1], [2750,4750,1], [zmin,zmax,1]],
            [[1000,7000,8], [3000,5000,8], [zmin,zmax,8]],
            ]

    for i in range(len(vnames)):
        vname = vnames[i]
        ranges = rangess[i]
        volume = Volume.createFromTiffs(prj, tiff_dir, vname, ranges, "%05d.tif")
        if not volume.valid:
            print("Volume %s not created: %s"%(vname, volume.error))
            exit()

'''
new_prj = testCreateProject()
testCreateVolumes(new_prj)
# prj = testReadProject()

exit()
'''

app = QApplication(sys.argv)

# for i in range(5):
#     color = Utils.getNextColor()
#     print(color)


khartes = Khartes(app)
# khartes.window
# project = testReadProject()
# khartes.loadVolume('D:\\Vesuvius\\scroll1\\part6000-7250\\windowed\\win1h.nrrd')


app.exec()

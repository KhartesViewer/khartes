
'''
Take thee again another scroll, and write in it all the former words that 
were in the first scroll, which Jehoiakim the king of Judah hath burned.
Jeremiah 36:28
'''

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QSurfaceFormat
from main_window import MainWindow

class Khartes():

    def __init__(self, app):
        window = MainWindow("χάρτης", app)
        self.app = app
        self.window = window
        window.show()

fmt = QSurfaceFormat()
# Note that pyQt5 only supports OpenGL versions 2.0, 2.1, and 4.1 Core :
# https://riverbankcomputing.com/pipermail/pyqt/2017-January/038640.html
# To get the latest OpenGL version, perhaps use the python opengl module?
# https://stackoverflow.com/questions/38645674/issues-with-pyqt5s-opengl-module-and-versioning-calls-for-incorrect-qopenglfu
# But for our purposes, 4.1 Core is fine, since that is the last version
# supported by MacOS.
fmt.setVersion(4, 1)
fmt.setProfile(QSurfaceFormat.CoreProfile)
fmt.setOption(QSurfaceFormat.DebugContext)
QSurfaceFormat.setDefaultFormat(fmt)

# According to https://doc.qt.io/qt-5/qopenglwidget.html
# QSurfaceFormat.setDefaultFormat needs to be called before
# constructing QApplication, for MacOS

app = QApplication(sys.argv)

khartes = Khartes(app)
app.exec()

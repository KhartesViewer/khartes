
'''
Take thee again another scroll, and write in it all the former words that 
were in the first scroll, which Jehoiakim the king of Judah hath burned.
Jeremiah 36:28
'''

import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

class Khartes():

    def __init__(self, app):
        window = MainWindow("χάρτης", app)
        self.app = app
        self.window = window
        window.show()

app = QApplication(sys.argv)

khartes = Khartes(app)
app.exec()

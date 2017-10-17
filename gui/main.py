from __future__ import print_function
# TODO: consider using PyQt5
from PyQt4 import QtGui
import sys, design


class ExampleApp(QtGui.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)

def runGUI():    
    # app = QtGui.QApplication(sys.argv)
    # window = QtGui.QWidget()

    d = design.Ui_MainWindow
    window.show()
    sys.exit(app.exec_()) 

def main():
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
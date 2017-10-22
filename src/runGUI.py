from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from design import Ui_MainWindow
import sys, cv2, os
    

class myGUI(QMainWindow):
    def  __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.setup()
        # TODO: think about making those paramters visible or modifiable
        maxsize = (16, 16)
        classNo = 62
        self.model = modelCNN(maxsize, classNo, "trained_model.h5")
        self.show()

    def setup(self):
        self.ui.setupUi(self)
        self.ui.pushButtonLoadFile.clicked.connect(self.openFileDialog)
        self.ui.statusBar.showMessage("by Kamil Kuczaj 2017")
        self.setupCamera()
        self.setupComboBox()

    def openFileDialog(self):
        self.filename = QFileDialog.getOpenFileName()
        self.ui.labelImage.setScaledContents(True)
        self.ui.labelImage.setPixmap(QPixmap(self.filename[0]))

    def setupCamera(self):
        print("TODO: implement camera functionality")

    def setupComboBox(self):
        self.ui.comboBoxAlgorithms.addItem("Convolutional Neural Network")
        self.ui.comboBoxAlgorithms.addItem("k Nearest Neighbors")
        self.ui.comboBoxAlgorithms.currentIndexChanged.connect(
            lambda: print(self.ui.comboBoxAlgorithms.currentIndex()))


def main():
    app = QApplication(sys.argv)
    m = myGUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    sys.path.append(os.path.join(os.getcwd(),"../src"))
    from cnn import modelCNN
    main()
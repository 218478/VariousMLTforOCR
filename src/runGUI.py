# from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
# from PyQt5.QtGui import QPixmap, QImage
from PyQt4 import QtGui
import argparse, sys, multiprocessing

# TODO: check the standard of the order of importing packages and own files
from design import Ui_MainWindow
from cnn import modelCNN
from opencv import TextExtractor

class myGUI(QMainWindow):
    def  __init__(self, pathToNNModels, pathToSeparatedChars):
        super().__init__()
        self.pathToNNModels = pathToNNModels
        self.pathToSeparatedChars = pathToSeparatedChars
        self.ui = Ui_MainWindow()
        self.setup()
        # TODO: think about making those paramters visible or modifiable
        maxsize = (16, 16)
        classNo = 62
        self.show()

    def setup(self):
        self.ui.setupUi(self)
        self.ui.pushButtonLoadFile.clicked.connect(self.openFileDialog)
        self.ui.statusBar.showMessage("by Kamil Kuczaj 2017")
        self.setupCamera()
        self.setupComboBox()

    def openFileDialog(self):
        self.filename = QFileDialog.getOpenFileName(directory='../VariousMLTforOCR/datasets')
        self.ui.labelImage.setScaledContents(True)
        self.ui.labelImage.setPixmap(QPixmap(self.filename[0]))
        self.doOCR()

    def doOCR(self):
        # TODO: change prints to logging and enable it to print on console for debug
        print("Chosen file: " + str(self.filename[0]))
        self.textExtractor = TextExtractor(str(self.filename[0]))
        self.textExtractor.contourExample()
        self.textExtractor.characterExtraction()
        if self.ui.comboBoxAlgorithms.currentIndex == 0:
            self.model = modelCNN(maxsize, classNo, "trained_model.h5")
            print("Convolutional Neural Network")

        if self.ui.comboBoxAlgorithms.currentIndex == 1:
            print("Multilayer Perceptron Neural Network")

        if self.ui.comboBoxAlgorithms.currentIndex == 2:
            print("k Nearest Neighbors")



    def setupCamera(self):
        print("TODO: implement camera functionality")

    def setupComboBox(self):
        self.ui.comboBoxAlgorithms.addItem("Convolutional Neural Network")
        self.ui.comboBoxAlgorithms.addItem("Multilayer Perceptron Neural Network")
        self.ui.comboBoxAlgorithms.addItem("k Nearest Neighbors")
        self.ui.comboBoxAlgorithms.currentIndexChanged.connect(
            lambda: print(self.ui.comboBoxAlgorithms.currentIndex()))


def main(pathToNNModels, pathToSeparatedChars):
    app = QApplication(sys.argv)
    m = myGUI(pathToNNModels, pathToSeparatedChars)
    sys.exit(app.exec_())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToNNModels", help="Directory to stored neural networks models")
    parser.add_argument("pathToSeparatedChars", help="Directory to where separated characters will be stored")
    parser.add_argument("pathToSeparatedWords", help="Directory to where separated words will be stored")
    parser.add_argument("pathToLogFileDir", help="Path to log file")
    args = parser.parse_args()
    main(args.pathToNNModels, args.pathToSeparatedChars)
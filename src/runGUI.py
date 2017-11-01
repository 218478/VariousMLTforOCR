from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QStyleFactory
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor
from PyQt5.QtCore import Qt
import argparse, sys, os

# TODO: check the standard of the order of importing packages and own files
from design import Ui_MainWindow
from cnn import modelCNN
from textExtractor import TextExtractor
from readChars74K import Reader_Chars74K # TODO: hard-code classes

class myGUI(QMainWindow):
    def  __init__(self, pathToNNModels):
        super().__init__()
        self.pathToNNModels = pathToNNModels
        self.ui = Ui_MainWindow()
        self.setup()
        # TODO: think about making those paramters visible or modifiable
        self.maxsize = (16, 16)
        self.classNo = 62
        self.filename = ""
        self.show()
        self.center()

    def center(self):
        """
        Thanks to https://stackoverflow.com/questions/20243637/pyqt4-center-window-on-active-screen
        answered 27.11.13 14:17     accessed on 31.10.2017 23:12
        """
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def setup(self):
        self.ui.setupUi(self)
        self.ui.pushButtonLoadFile.clicked.connect(self.openFileDialog)
        self.ui.horizontalSliderMaxH.valueChanged.connect(self.doOCRwhenSliderUsed)
        self.ui.horizontalSliderMinH.valueChanged.connect(self.doOCRwhenSliderUsed)
        self.ui.horizontalSliderMaxW.valueChanged.connect(self.doOCRwhenSliderUsed)
        self.ui.horizontalSliderMinW.valueChanged.connect(self.doOCRwhenSliderUsed)
        self.ui.pushButtonCamera.clicked.connect(self.setupCamera)
        self.ui.statusBar.showMessage("by Kamil Kuczaj 2017")
        self.setupComboBox()

    def openFileDialog(self):
        self.filename = QFileDialog(parent=self).getOpenFileName(directory='../VariousMLTforOCR/testing/example_images', options=QFileDialog.DontUseNativeDialog)
        self.ui.labelImage.setScaledContents(True)
        self.ui.labelImage.setPixmap(QPixmap(self.filename[0]))
        self.doOCR()

    def doOCRwhenSliderUsed(self):
        if len(self.filename) != 0:
            self.doOCR()

    def doOCR(self):
        # TODO: change prints to logging and enable it to print on console for debug
        print("Chosen file: " + str(self.filename[0]))
        self.textExtractor = TextExtractor(str(self.filename[0]))
        self.textExtractor.contourExample(self.ui.horizontalSliderMaxH.value(), self.ui.horizontalSliderMinH.value(),
                                          self.ui.horizontalSliderMaxW.value(), self.ui.horizontalSliderMinW.value())
        self.textExtractor.characterExtraction()
        print("I read " + str(len(self.textExtractor.characters)) + " words")
        for idx, word in enumerate(self.textExtractor.characters):
            print("For word " + str(idx) + " extracted " + str(len(word)) + " chars")
        print(self.ui.comboBoxAlgorithms.currentIndex())
        if self.ui.comboBoxAlgorithms.currentIndex() == 0:
            self.reader = Reader_Chars74K()
            self.reader.classNo = self.classNo
            self.reader.createReadableLabels()
            self.model = modelCNN(self.maxsize, self.classNo, os.path.join(self.pathToNNModels, "cnn_model.h5"))
            print("temp")
            self.ui.textEdit.clear()
            s = ""
            for word in self.textExtractor.characters:
                for char in word:
                    s += self.reader.readableLabels[self.model.predict(char)]
                s += " "
            self.ui.textEdit.append(s)
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


def main(pathToNNModels):
    app = QApplication(sys.argv)
    m = myGUI(pathToNNModels)
    sys.exit(app.exec_())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToNNModels", help="Directory to stored neural networks models")
    parser.add_argument("pathToLogFileDir", help="Path to log file")
    # TODO: logging
    args = parser.parse_args()
    main(args.pathToNNModels)
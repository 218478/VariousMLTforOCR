from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QStyleFactory
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor
from PyQt5.QtCore import Qt
import argparse, sys, os, cv2

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
        self.filename = QFileDialog(parent=self).getOpenFileName(directory='../VariousMLTforOCR/testing/example_images/', options=QFileDialog.DontUseNativeDialog)
        self.ui.labelImage.setScaledContents(True)
        self.ui.labelImage.setPixmap(QPixmap(self.filename[0]))
        self.doOCR()

    def doOCRwhenSliderUsed(self):
        print("implement this with a lag")
        if len(self.filename) != 0:
            self.doOCR()

    def getTextFromModel(self, model):
        s = ""
        for word in self.tE.charactersFromWord:
            for char in word:
                s += self.reader.readableLabels[model.predict(char)]
            s += " "
        return s

    def extractTextFromSelectedFile(self):
        self.tE = TextExtractor(str(self.filename[0]))
        self.tE.wordExtraction(self.ui.horizontalSliderMaxH.value(), self.ui.horizontalSliderMinH.value(),
                               self.ui.horizontalSliderMaxW.value(), self.ui.horizontalSliderMinW.value())
        self.tE.characterExtraction()
        self.tE.reverseEverything()

    def cvtCvMatToQImg(self, img):
        """
        Expects grayscale image
        https://stackoverflow.com/questions/37284161/i-used-opencv-to-convert-a-picture-to-grayscale-how-to-display-the-picture-on-py
        Answered by tfv on 17.05.2016 20:17          Accessed on 02.11.2017 20:02
        """
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        height, width = img.shape[:2]
        return QImage(img, width, height, QImage.Format_RGB888)

    def doOCR(self):
        self.extractTextFromSelectedFile()
        self.ui.labelImageAfterOCR.setPixmap(QPixmap.fromImage(self.cvtCvMatToQImg(self.tE.image).scaled(self.ui.labelImageAfterOCR.width(), self.ui.labelImageAfterOCR.height())))
        if self.ui.comboBoxAlgorithms.currentIndex() == 0:
            self.reader = Reader_Chars74K()
            self.reader.classNo = self.classNo
            self.reader.createReadableLabels()
            self.ui.textEdit.clear()
            s = self.getTextFromModel(modelCNN(self.maxsize, self.classNo, os.path.join(self.pathToNNModels, "cnn_model.h5")))
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
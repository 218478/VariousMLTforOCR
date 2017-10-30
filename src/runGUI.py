from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import argparse, sys, multiprocessing, os

# TODO: check the standard of the order of importing packages and own files
from design import Ui_MainWindow
from cnn import modelCNN
from opencv import TextExtractor
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
        print(self.ui.comboBoxAlgorithms.currentIndex())
        if self.ui.comboBoxAlgorithms.currentIndex() == 0:
            self.reader = Reader_Chars74K("temp_to_save_np_array.temp",self.classNo)
            self.model = modelCNN(maxsize, classNo, "trained_model.h5")
            print("temp")
            for word in self.textExtractor.characters:
                for char in word:
                    s = self.reader.readableLabels(self.model.predict(char))
                    self.ui.textBrowser = s
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
    #os.environ.pop("QT_STYLE_OVERRIDE")
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
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QStyleFactory
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor
from PyQt5.QtCore import Qt
import argparse, sys, os, cv2, tesserocr
import numpy as np
from tesserocr import PyTessBaseAPI, RIL
from PIL import Image
from time import time

from design import Ui_MainWindow
from cnn import modelCNN
from mlp import modelMLP
from knn import kNN
from textExtractor import TextExtractor
from readChars74K import Reader_Chars74K # TODO: hard-code classes

class myGUI(QMainWindow):
    def  __init__(self, pathToNNModels):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.setupParameters(pathToNNModels=pathToNNModels)
        self.setup()
        self.showMaximized()

    def setupParameters(self, pathToNNModels, imgWidth=16, imgHeight=16, classNo=62):
        """
        TODO: think about making those paramters visible or modifiable
        """
        self.pathToNNModels = pathToNNModels
        self.maxsize = (16, 16)
        self.classNo = 62
        self.filename = ""
        self.tE = TextExtractor()
        self.modelCNN = modelCNN(self.maxsize, self.classNo, os.path.join(self.pathToNNModels, "cnn_model_for_my_dataset.h5"))

    # TODO: add drag'n'drop functionality
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat('image/svg+jpg+png'):
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        print(e.mimeData().text())

    def setupEvents(self):
        self.ui.pushButtonLoadFile.clicked.connect(self.openFileDialog)
        self.ui.pushButton_doOCR.clicked.connect(self.doOCR)
        self.ui.horizontalSliderMaxH.valueChanged.connect(self.displayLCDValue)
        self.ui.horizontalSliderMinH.valueChanged.connect(self.displayLCDValue)
        self.ui.horizontalSliderMaxW.valueChanged.connect(self.displayLCDValue)
        self.ui.horizontalSliderMinW.valueChanged.connect(self.displayLCDValue)

    def setupComboBox(self):
        self.ui.comboBoxAlgorithms.addItem("Convolutional Neural Network")
        self.ui.comboBoxAlgorithms.addItem("Multilayer Perceptron Neural Network")
        self.ui.comboBoxAlgorithms.addItem("k Nearest Neighbors")
        self.ui.comboBoxAlgorithms.addItem("Tesseract API")

    def setup(self):
        self.ui.setupUi(self)
        self.setupEvents()
        self.ui.labelImage.setScaledContents(True)
        self.ui.labelImageAfterOCR.setScaledContents(True)
        self.ui.lcdmaxH.display(str(self.ui.horizontalSliderMaxH.value()))
        self.ui.lcdminH.display(str(self.ui.horizontalSliderMinH.value()))
        self.ui.lcdmaxW.display(str(self.ui.horizontalSliderMaxW.value()))
        self.ui.lcdminW.display(str(self.ui.horizontalSliderMinW.value()))
        self.setupComboBox()

    def setSliderValues(self, width, height):
        self.ui.horizontalSliderMaxH.setMaximum(height)
        self.ui.horizontalSliderMaxH.setMinimum(0)
        self.ui.horizontalSliderMinH.setMaximum(height)
        self.ui.horizontalSliderMinH.setMinimum(0)
        self.ui.horizontalSliderMaxW.setMaximum(width)
        self.ui.horizontalSliderMaxW.setMinimum(0)
        self.ui.horizontalSliderMinW.setMaximum(width)
        self.ui.horizontalSliderMinW.setMinimum(0)

    def openFileDialog(self):
        self.filename = QFileDialog(parent=self).getOpenFileName(options=QFileDialog.DontUseNativeDialog)[0]
        if len(self.filename) != 0:
            width, height = self.ui.labelImage.width(), self.ui.labelImage.height()
            self.ui.labelImage.setPixmap(QPixmap(self.filename).scaled(width, height))
            width, height = cv2.imread(self.filename).shape[:2]
            self.setSliderValues(width, height)

    def displayLCDValue(self):
        self.ui.lcdmaxH.display(str(self.ui.horizontalSliderMaxH.value()))
        self.ui.lcdminH.display(str(self.ui.horizontalSliderMinH.value()))
        self.ui.lcdmaxW.display(str(self.ui.horizontalSliderMaxW.value()))
        self.ui.lcdminW.display(str(self.ui.horizontalSliderMinW.value()))

    def getTextFromModel(self, model):
        s = ""
        for word in self.tE.charactersFromWord:
            for char in word:
                s += self.reader.readableLabels[model.predict(char)]
            s += " "
        return s[:-2] # don't include space at the end

    def extractTextFromSelectedFile(self):
        self.tE.wordExtraction(self.ui.horizontalSliderMaxH.value(), self.ui.horizontalSliderMinH.value(),
                               self.ui.horizontalSliderMaxW.value(), self.ui.horizontalSliderMinW.value())
        self.tE.characterExtraction(displayImages=False)
        self.tE.reverseEverything()

    def cvtCvMatToQImg(self, img):
        """
        Expects grayscale image
        https://stackoverflow.com/questions/37284161/i-used-opencv-to-convert-a-picture-to-grayscale-how-to-display-the-picture-on-py
        Answered by tfv on 17.05.2016 20:17          Accessed on 02.11.2017 20:02
        """
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        height, width = img.shape[:2]
        bytesPerLine = 3 * width
        return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

    def takeCareOfImageRotation(self, image):
        """
        https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
        """
        if len(image.shape) > 2:
            gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        else:
            gray = cv2.bitwise_not(image)
        cv2.imshow("before rotation", gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # grab the (x, y) coordinates of all pixel values that are greater than zero, then use these coordinates to
        # compute a rotated bounding box that contains all coordinates
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # the `cv2.minAreaRect` function returns values in the range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle # otherwise, just take the inverse of the angle to make it positive

        # rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imshow("after rotation", rotated)
        # cv2.waitKey()
        cv2.destroyAllWindows()
        return rotated

    def outputText(self, s):
        self.ui.textEdit.clear()
        self.ui.textEdit.append(s)

    def tesseractBoundingBox(self, image):
        '''
        https://stackoverflow.com/questions/20831612/getting-the-bounding-box-of-the-recognized-words-using-python-tesseract
        '''
        image_boundedBoxes = image.copy()
        with PyTessBaseAPI() as api:
            api.SetImage(Image.fromarray(image))
            boxes = api.GetComponentImages(RIL.TEXTLINE, True)
            for i, (im, box, _, _) in enumerate(boxes):
                api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                cv2.rectangle(image_boundedBoxes, (box['x'], box['y']), (box['x'] + box['w'], box['y']+box['h']),220,2)
                ocrResult = api.GetUTF8Text()
                conf = api.MeanTextConf()
        return image_boundedBoxes

    def doOCR(self):
        if len(self.filename) == 0:
            return
        start = time()
        self.image = cv2.imread(self.filename)
        self.tE.readFromImage(self.takeCareOfImageRotation(self.image))
        self.extractTextFromSelectedFile()
        width, height = self.ui.labelImageAfterOCR.width(), self.ui.labelImageAfterOCR.height()
        if self.ui.comboBoxAlgorithms.currentIndex() == 0: # CNN
            qImg = self.cvtCvMatToQImg(self.tE.image).scaled(width, height)
            self.ui.labelImageAfterOCR.setPixmap(QPixmap.fromImage(qImg))
            self.reader = Reader_Chars74K()
            self.reader.classNo = self.classNo
            self.reader.createReadableLabels()
            self.outputText(self.getTextFromModel(self.modelCNN))

        if self.ui.comboBoxAlgorithms.currentIndex() == 2:
            print("k Nearest Neighbors")

        if self.ui.comboBoxAlgorithms.currentIndex() == 3: # Tesseract
            qImg = self.cvtCvMatToQImg(self.tesseractBoundingBox(self.image)).scaled(width, height)
            self.ui.labelImageAfterOCR.setPixmap(QPixmap.fromImage(qImg))
            img = cv2.cvtColor(self.takeCareOfImageRotation(self.image), cv2.COLOR_BGR2RGB)
            self.outputText(tesserocr.image_to_text(Image.fromarray(img)))
        stop = time()
        self.ui.statusBar.showMessage("The operation took: " + str(stop-start) + " seconds")


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
import argparse
import cv2
import os
import sys
import tesserocr
from tesserocr import PyTessBaseAPI, RIL
from time import time

import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from CNN import CNN
from design import Ui_MainWindow
from NearestNeighbor import NearestNeighbor
from DatasetReader import DatasetReader, print_image_array  # TODO: hard-code classes
from TextExtractor import TextExtractor


def cvmat_to_qimg(img):
    """
    Expects grayscale image
    https://stackoverflow.com/questions/37284161/i-used-opencv-to-convert-a-picture-to-grayscale-how-to-display-the-picture-on-py
    Answered by tfv on 17.05.2016 20:17          Accessed on 02.11.2017 20:02
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    height, width = img.shape[:2]
    bytes_per_line = 3 * width
    return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)


def deskew(image):
    """
    https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    """
    if len(image.shape) > 2:
        gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    else:
        gray = cv2.bitwise_not(image)
    # cv2.imshow("before rotation", gray)
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
        angle = -angle  # otherwise, just take the inverse of the angle to make it positive

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # cv2.imshow("after rotation", rotated)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return rotated


def tesseract_bounding_box(image):
    """
    https://stackoverflow.com/questions/20831612/getting-the-bounding-box-of-the-recognized-words-using-python-tesseract
    """
    image_bounding_boxes = image.copy()
    with PyTessBaseAPI() as api:
        api.SetImage(Image.fromarray(image))
        boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        for i, (im, box, _, _) in enumerate(boxes):
            api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
            cv2.rectangle(image_bounding_boxes, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), 220, 2)
    return image_bounding_boxes


class MyGUI(QMainWindow):
    def __init__(self, path_to_models):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.setup_parameters(path_to_models=path_to_models)
        self.setup()
        self.showMaximized()

    def setup_parameters(self, path_to_models, img_width=64, img_height=64, class_no=62):
        """
        TODO: think about making those parameters visible or modifiable
        """
        self.path_to_models = path_to_models
        self.maxsize = (img_height, img_width)
        self.class_no = class_no
        self.filename = ""  # Hack for the None check later
        self.extractor = TextExtractor(self.maxsize)
        self.cnn = CNN(self.maxsize, self.class_no,
                       os.path.join(self.path_to_models, "cnn_model_for_my_dataset_64x64.h5"))
        self.kNN = NearestNeighbor(self.maxsize, k=11)
        with np.load(os.path.join(self.path_to_models, 'knn_data.npz')) as data:
            self.kNN.train(data['trainSet'], data['trainLabels'])

    def setup_events(self):
        self.ui.pushButtonLoadFile.clicked.connect(self.open_file_dialog)
        self.ui.pushButton_doOCR.clicked.connect(self.do_ocr)
        self.ui.horizontalSliderMaxH.valueChanged.connect(self.display_lcd_value)
        self.ui.horizontalSliderMinH.valueChanged.connect(self.display_lcd_value)
        self.ui.horizontalSliderMaxW.valueChanged.connect(self.display_lcd_value)
        self.ui.horizontalSliderMinW.valueChanged.connect(self.display_lcd_value)

    def setup_combo_box(self):
        self.ui.comboBoxAlgorithms.addItem("Convolutional Neural Network")
        self.ui.comboBoxAlgorithms.addItem("k Nearest Neighbors")
        self.ui.comboBoxAlgorithms.addItem("Tesseract API")

    def setup(self):
        self.ui.setupUi(self)
        self.setup_events()
        self.ui.labelImage.setScaledContents(True)
        self.ui.labelImageAfterOCR.setScaledContents(True)
        self.ui.lcdmaxH.display(str(self.ui.horizontalSliderMaxH.value()))
        self.ui.lcdminH.display(str(self.ui.horizontalSliderMinH.value()))
        self.ui.lcdmaxW.display(str(self.ui.horizontalSliderMaxW.value()))
        self.ui.lcdminW.display(str(self.ui.horizontalSliderMinW.value()))
        self.setup_combo_box()

    def set_slider_values(self, width, height):
        self.ui.horizontalSliderMaxH.setMaximum(height)
        self.ui.horizontalSliderMaxH.setMinimum(0)
        self.ui.horizontalSliderMinH.setMaximum(height)
        self.ui.horizontalSliderMinH.setMinimum(0)
        self.ui.horizontalSliderMaxW.setMaximum(width)
        self.ui.horizontalSliderMaxW.setMinimum(0)
        self.ui.horizontalSliderMinW.setMaximum(width)
        self.ui.horizontalSliderMinW.setMinimum(0)

    def open_file_dialog(self):
        self.filename = QFileDialog().getOpenFileName(options=QFileDialog.DontUseNativeDialog)[0]
        if len(self.filename) != 0:
            width, height = self.ui.labelImage.width(), self.ui.labelImage.height()
            self.ui.labelImage.setPixmap(QPixmap(self.filename).scaled(width, height))
            width, height = cv2.imread(self.filename).shape[:2]
            self.set_slider_values(width, height)

    def display_lcd_value(self):
        self.ui.lcdmaxH.display(str(self.ui.horizontalSliderMaxH.value()))
        self.ui.lcdminH.display(str(self.ui.horizontalSliderMinH.value()))
        self.ui.lcdmaxW.display(str(self.ui.horizontalSliderMaxW.value()))
        self.ui.lcdminW.display(str(self.ui.horizontalSliderMinW.value()))

    def get_text_from_model(self, model):
        self.reader = DatasetReader()
        self.reader.classNo = self.class_no
        self.reader.create_readable_labels()
        s = ""
        for word in self.extractor.characters_from_word:
            for char in word:
                print_image_array(char)
                s += self.reader.readableLabels[model.predict(char)]
            s += " "
        return s[:-2]  # don't include space at the end

    def extract_text_from_selected_file(self):
        self.extractor.word_extraction(self.ui.horizontalSliderMaxH.value(), self.ui.horizontalSliderMinH.value(),
                                       self.ui.horizontalSliderMaxW.value(), self.ui.horizontalSliderMinW.value(),
                                       display_images=False)
        self.extractor.character_extraction(display_images=False, verbose=False)
        self.extractor.reverse_everything()

    def fill_textbox(self, s):
        self.ui.textEdit.clear()
        self.ui.textEdit.append(s)

    def do_ocr(self):
        if len(self.filename) == 0:
            return
        start = time()
        self.image = cv2.imread(self.filename)
        self.extractor.read_from_image(deskew(self.image))
        self.extract_text_from_selected_file()
        width, height = self.ui.labelImageAfterOCR.width(), self.ui.labelImageAfterOCR.height()
        if self.ui.comboBoxAlgorithms.currentIndex() == 0:  # CNN
            qimg = cvmat_to_qimg(self.extractor.image).scaled(width, height)
            txt = self.get_text_from_model(self.cnn)

        if self.ui.comboBoxAlgorithms.currentIndex() == 1:  # NearestNeighbor
            qimg = cvmat_to_qimg(self.extractor.image).scaled(width, height)
            txt = self.get_text_from_model(self.kNN)

        if self.ui.comboBoxAlgorithms.currentIndex() == 2:  # Tesseract
            qimg = cvmat_to_qimg(tesseract_bounding_box(self.image)).scaled(width, height)
            img = cv2.cvtColor(deskew(self.image), cv2.COLOR_BGR2RGB)
            txt = tesserocr.image_to_text(Image.fromarray(img))

        self.ui.labelImageAfterOCR.setPixmap(QPixmap.fromImage(qimg))
        self.fill_textbox(txt)
        stop = time()
        self.ui.statusBar.showMessage("The operation took: " + str(stop - start) + " seconds")


def main(path_to_models):
    app = QApplication(sys.argv)
    m = MyGUI(path_to_models)
    sys.exit(app.exec_())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_models", help="Directory to stored algorithm models")
    args = parser.parse_args()
    main(args.path_to_models)

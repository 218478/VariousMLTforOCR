# TODO: consider using PyQt5
import PyQt5
import sys, design, cv2


class ExampleApp(QtGui.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        img = cv2.imread("converted.jpg")
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        # qPixMap = QtGui.QPixmap(qImg)
        # self.frame = qPixMap

        


def main():
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
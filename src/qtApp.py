from __future__ import print_function

import numpy as np
import sys, cv2, os, argparse, logging
from PyQt4 import QtGui


# app = QtGui.QApplication(sys.argv)

# window = QtGui.QWidget()

# window.show()
# sys.exit(app.exec_()) 


def main(pathToImage):
    img = cv2.imread(pathToImage, 0)
    img = cv2.rectangle(img,(4,6),(23,25),0,1)
    img = cv2.bitwise_not(img)
    # img /= 32
    kernel = np.ones((2,2),np.uint8)
    # img = cv2.erode(img, kernel)
    # img = cv2.bitwise_not(img)
    cv2.namedWindow(pathToImage, cv2.WINDOW_NORMAL)

    scanningLetter = False
    zeroColumn = False
    initialHeight = 0
    for row in range(img.shape[0]):
        if np.isclose(np.dot(np.transpose(np.ones(img.shape[0])),img[:,row]),0): # if the first letter is encountered
            for col in range(img.shape[1]):
                print(col)

    # print(height)

    # TODO: sth is wrong with labeling cause img.shape[0] is a row not col
    for col in range(img.shape[0]):
        if np.isclose(np.dot(np.transpose(np.ones(img.shape[0])),img[:,col]),0): # if the column is full of zeros this is True
            zeroColumn = True       
        if not zeroColumn and not scanningLetter: # if encountered next character
            beginning = col
            scanningLetter = True
        if zeroColumn and scanningLetter: # if character is finished
            img = cv2.rectangle(img,(4,6),(20,30),(0,255,0),1)



    # for row in img:
    #     for cell in row:
    #         sys.stdout.write("%d" % cell)
    #     sys.stdout.write("\n")

    cv2.imshow(pathToImage, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # from appJar import gui
    # # create a GUI variable called app
    # app = gui()
    # app.addLabel("title", "Welcome to appJar")
    # app.setLabelBg("title", "red")
    # app.go()

    parser = argparse.ArgumentParser()
    parser.add_argument("pathToImage", help="Directory to stored datasets")
    parser.add_argument("pathToLogFile", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFile, 'logFile.log'))#, level = logging.INFO)
    main(args.pathToImage)
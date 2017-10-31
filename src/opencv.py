import numpy as np
import cv2, os, argparse, logging


class TextExtractor():
    def __init__(self, pathToImage = None):
        self.pathToImage = pathToImage
        self.words = []
        self.characters = []

    def addWord(self, i, img):
        """
        Adds i-th word. Function supports substitution.
        """
        if (i >= len(self.words)):
            self.words.append(img)
            self.characters.append([])
        else:
            self.words[i] = img

    def addChar(self, i, j, img):
        """
        Adds j-th letter to i-th word. Function supports substitution. Does not check
        i bounds.
        """
        if (j >= len(self.characters[i])):
            self.characters[i].append(img)
        else:
            self.characters[i][j] = img

    def erodeImg(self, img, kernel):
        return cv2.erode(img, kernel)

    # one of the next post of the same user states that you do not
    # have to reverse the bytes anymore
    # https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
    def reversBGRtoRGB(self, img):
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img

    def convertOpenCVtoQImageFormat(self):
        """
        Needs a colored image
        https://stackoverflow.com/questions/33741920/convert-opencv-3-iplimage-to-pyqt5-qimage-qpixmap-in-python
        Accessed: 25.10.2017 15:57      Answered by: AdityaIntwala
        """
        for i in range(len(self.words)):
            for c in self.characters[i]:
                height, width = c.shape
                bytesPerLine = 3 * width
                c = self.reversBGRtoRGB(c)
                c = QImage(char.data, width, height, bytesPerLine,QImage.Format_RGB888)

    def characterExtraction(self):
        '''
        TODO: make it work with color images (opencv can convert it on the fly)
        TODO: include verification by average width of a character. I think, after
            eroding letter "H", the algorithm might split it into two letter "I"s
        Needs a contoured box in order to properly scan and a MAYBE greyscale [0,255].
        It then inverts the image, divides every pixel by 32, so that it's between values [0,7]
        Then it erodes it with 2x2 kernel of uint8 and scans it.

        The whole algorithm scans every column of an image until the right side is reached.
        Scanning begins by multiplying the column with 35x1 vector of ones. If the result is not
        zero, then the letter has been encountered. It's then looking for the moment when the
        result of multiplication is zero again. It means that the end of a letter has been recognized.
        It then draws a rectangle over it and saves into a new image.
        '''
        for idxWord, word, in enumerate(self.words):
            img = word
            origImg = img
            imgCopy = img.copy()
            img = cv2.bitwise_not(img)
            imgCopy = cv2.bitwise_not(imgCopy)
            # img /= 32 it was not working on python3
            for row in img:
                for pixel in row:
                    pixel /= 32
            # img = np.divide(img,32) # divided to get data into [0,7]
            kernel = np.ones((2,2),np.uint8)

            img = cv2.erode(img, kernel)
            # imgCopy = pool.map(self.erodeImg, img, kernel)
            # imgCopy = cv2.erode(imgCopy, kernel)
            imgCopy = cv2.bitwise_not(imgCopy)
            # cv2.namedWindow(self.pathToImage, cv2.WINDOW_NORMAL) # lets you resize the window

            # TODO: remove printing when sure it's doing proper OCR
            # TODO: add contoured model with those boxes, cause it looks nice

            # for row in img:
            #     for cell in row:
            #         sys.stdout.write("%d" % cell)
            #     sys.stdout.write("\n")

            scanningLetter = False
            zeroResult = False
            upperBorder = 0
            lowerBorder = 0
            margin = 2

            # detect the height of the words start
            for row in range(img.shape[0]):
                # print(np.dot(np.ones((1 ,img.shape[1])),img[row,:]))
                zeroResult = np.isclose(np.dot(np.ones((1 ,img.shape[1])),img[row,:]), 0)
                if not zeroResult and not scanningLetter:
                    upperBorder = row
                    if row > margin: # add margin to make rectangle bigger than the letter
                        upperBorder -= margin
                    scanningLetter = True
                if scanningLetter and zeroResult:
                    lowerBorder = row
                    if row < img.shape[0]-margin:
                        lowerBorder += margin
                    scanningLetter = False
                    break

            x1 = 0
            x2 = 0
            letters = () # tuple for storing recognized letters
            # detect breaks between letters
            for col in range(img.shape[1]):
                # print(np.dot(np.ones((1 ,img.shape[0])),img[:,col]))
                zeroResult = np.isclose(np.dot(np.ones((1 ,img.shape[0])),img[:,col]), 0)
                if not zeroResult and not scanningLetter: # letter start
                    x1 = col
                    if x1 > margin:
                        x1 -= margin
                    scanningLetter = True
                    # print("poczatek literki")
                if zeroResult and scanningLetter: # letter end
                    x2 = col
                    if col < img.shape[1] - margin:
                        x2 += margin
                    scanningLetter = False
                    # print("koniec literki")
                    # print("szerokosc: %d" % (x2-x1))

                    cv2.rectangle(imgCopy, (x1,upperBorder),(x2,lowerBorder),150,1)
                    letters = letters + (imgCopy[upperBorder:lowerBorder, x1:x2],)


            for idxChar, letter in enumerate(letters):
                # cv2.imshow("cropped_" + str(idxChar), letter)
                self.addChar(idxWord, idxChar, letter)
                # cv2.imwrite(os.path.join("/home/kkuczaj/Praca_inzynierska/build","cropped_") + str(idxChar) + ".jpg", letter)
                # cv2.moveWindow("cropped_" + str(idxChar), 30*(1+idxChar), 30*(1+idxChar))

    # TODO: make tuning easier and add cropping
    def contourExample(self, maxH=100, maxW=200, minH=40, minW=40):
        '''
        Grayscales images and contours an image.
        https://stackoverflow.com/questions/23506105/extracting-text-opencv

        answered May 9 '14 at 5:07 by anana
        accessed Oct 10 '17 at 18:46
        '''
        image = cv2.imread(self.pathToImage)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
        _,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
        _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

        self.contours = ()
        # for each contour found, draw a rectangle around it on original image
        for idx, contour in enumerate(contours):
            # get rectangle bounding contour
            [x,y,w,h] = cv2.boundingRect(contour)
            # discard areas that are too large
            if h>maxH and w>maxW:
                continue
            # discard areas that are too small
            if h<minH or w<minW:
                continue

            # draw rectangle around contour on original image
            # cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
            self.contours = self.contours + (gray[y:y+h,x:x+w],)

        # TODO: encapsulate this in multiprocess
        for idx, c in enumerate(self.contours):
            # TODO: it stopped working when single image is provided
            # cv2.imshow("boundingRectangle_" + str(idx), c)
            # cv2.moveWindow("boundingRectangle_" + str(idx), 1400, (1000-50*idx))
            self.addWord(idx, c)
        if (len(self.words) is 0):
            self.addWord(0, gray)



def main(singleWord, multipleWords):
    print("OpenCV Version: {}". format(cv2. __version__))
    t = TextExtractor(singleWord)
    t.contourExample( 100, 200, 40, 40) # original values 300, 300, 40, 40
    t.characterExtraction()
    # t.convertOpenCVtoQImageFormat()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToSingleWordImage", help="Path to single word image")
    parser.add_argument("pathToMultipleWordsImage", help="Path to multiple words image")
    parser.add_argument("pathToLogFile", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFile, 'logFile.log'))#, level = logging.INFO)
    main(args.pathToSingleWordImage, args.pathToMultipleWordsImage)
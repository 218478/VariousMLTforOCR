import numpy as np
import cv2, os, argparse, logging, sys

class TextExtractor():
    def __init__(self, pathToImage, grayValue = 150, boxThickness = 5):
        if not os.path.exists(pathToImage):
            sys.exit("Bad filename provide. File: " + str(pathToImage) + " does not exist!")
        self.pathToImage = pathToImage
        self.grayValue = grayValue
        self.boxThickness = boxThickness
        self.image = cv2.imread(self.pathToImage, flags=cv2.IMREAD_GRAYSCALE)
        self.words = []
        self.charactersFromWord = []

    def reverseEverything(self):
        self.charactersFromWord = self.charactersFromWord[::-1]
        for word in self.charactersFromWord:
            word = reversed(word)
        self.words = reversed(self.words)

    def addWord(self, i, img):
        """
        Adds i-th word. Function supports substitution.
        """
        if (i >= len(self.words)):
            self.words.append(img)
            self.charactersFromWord.append([])
        else:
            self.words[i] = img

    def addCharToIthWord(self, i, j, img):
        """
        Adds j-th letter to i-th word. Function supports substitution. Does not check
        i bounds.
        """
        if (j >= len(self.charactersFromWord[i])):
            self.charactersFromWord[i].append(img)
        else:
            self.charactersFromWord[i][j] = img

    def characterExtraction(self, displayImages=False, verbose=False):
        '''
        TODO: include verification by average width of a character.
        Perform wordExtraction() method first.
        It then inverts the image, divides every pixel by 32, so that it's between values [0,7]
        Then it erodes it with 2x2 kernel of uint8 and scans it.
        '''
        for idxWord, word, in enumerate(self.words):
            img, imgCopy = word.copy(), word.copy()

            img = cv2.bitwise_not(img)
            imgCopy = cv2.bitwise_not(imgCopy)

            img = np.floor(np.divide(img,32)) # divided to get data into [0,7]
            kernel = np.ones((2,2),np.uint8)

            img = cv2.erode(img, kernel)
            imgCopy = cv2.erode(imgCopy, kernel)

            imgCopy = cv2.bitwise_not(imgCopy)

            if verbose == True:
                for row in img:
                    for cell in row:
                        sys.stdout.write("%d" % cell)
                    sys.stdout.write("\n")

            scanningLetter, zeroResult = False, False
            upperBorder, lowerBorder, margin = 0, 0, 2

            # detect the height of the words start
            for row in range(img.shape[0]):
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

            x1, x2 , letters = 0, 0, ()
            # detect breaks between letters
            for col in range(img.shape[1]):
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

                    letters = letters + (imgCopy[upperBorder:lowerBorder, x1:x2],)
                    cv2.rectangle(word, (x1,upperBorder),(x2,lowerBorder),self.grayValue,self.boxThickness)


            for idxChar, letter in enumerate(letters):
                if displayImages == True:
                    cv2.imshow("cropped_" + str(idxChar), letter)
                    cv2.moveWindow("cropped_" + str(idxChar), 30*(1+idxChar), 30*(1+idxChar))
                self.addCharToIthWord(idxWord, idxChar, letter)
            if displayImages == True:
                cv2.imshow("whole_word", word)
                cv2.waitKey()
                cv2.destroyAllWindows()

    def wordExtraction(self, maxH=100, maxW=200, minH=40, minW=40, displayImages=False):
        '''
        Grayscales images and contours an image.
        https://stackoverflow.com/questions/23506105/extracting-text-opencv

        answered May 9 '14 at 5:07 by anana
        accessed Oct 10 '17 at 18:46
        '''
        image = self.image.copy()
        _,thresh = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        dilated = cv2.dilate(thresh,kernel,iterations = 13)
        _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        self._contours = ()
        for idx, contour in enumerate(contours):
            [x,y,w,h] = cv2.boundingRect(contour)
            # discard areas that are too large
            if h>maxH and w>maxW:
                continue
            # discard areas that are too small
            if h<minH or w<minW:
                continue
            cv2.rectangle(self.image,(x,y),(x+w,y+h),self.grayValue,self.boxThickness)
            self._contours = self._contours + (image[y:y+h,x:x+w],)

        for idx, img in enumerate(self._contours):
            if displayImages == True:
                cv2.imshow("boundingRectangle_" + str(idx), img)
                cv2.moveWindow("boundingRectangle_" + str(idx), 1400, (1000-50*idx))
            self.addWord(idx, img)

        # single word case
        if (len(self.words) is 0):
            self.addWord(0, image)
            if displayImages == True:
                cv2.imshow("boundingRectangle_0", image)

        if displayImages == True:
            cv2.imshow("whole_text", self.image)
            cv2.waitKey()
            cv2.destroyAllWindows()



def main(singleWord, multipleWords):
    print("OpenCV Version: {}". format(cv2. __version__))
    t = TextExtractor(multipleWords)
    t.wordExtraction( 100, 200, 40, 40) # original values 300, 300, 40, 40
    t.characterExtraction()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToSingleWordImage", help="Path to single word image")
    parser.add_argument("pathToMultipleWordsImage", help="Path to multiple words image")
    parser.add_argument("pathToLogFile", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFile, 'logFile.log'))#, level = logging.INFO)
    main(args.pathToSingleWordImage, args.pathToMultipleWordsImage)
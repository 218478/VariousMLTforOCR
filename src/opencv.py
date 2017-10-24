import numpy as np
import sys, cv2, os, argparse, logging, glob
import pathlib2

# TODO: promote to opencv3
class TextExtractor():
    def __init__(self, pathToImage = None):
        self.pathToImage = pathToImage

    def setPathToImage(self, newPath):
        self.pathToImage = newPath

    def characterExtraction(self, pathToSave):
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
        # TODO: clear up this mess
        # if self.contours is None:
        #     throw
        # img = cv2.imread(self.pathToImage, 0)
        if not os.path.exists(self.pathToImage):
            raise ValueError
        img = cv2.imread(self.pathToImage, 0)
        origImg = img
        imgCopy = img.copy()
        img = cv2.bitwise_not(img)
        imgCopy = cv2.bitwise_not(imgCopy)
        img /= 32
        # img = np.divide(img,32) # divided to get data into [0,7]
        kernel = np.ones((2,2),np.uint8)
        img = cv2.erode(img, kernel)
        imgCopy = cv2.erode(imgCopy, kernel)
        imgCopy = cv2.bitwise_not(imgCopy)
        # thread problem cv2.namedWindow(self.pathToImage, cv2.WINDOW_NORMAL) # lets you resize the window

        # TODO: remove printing when sure it's doing proper OCR

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

                # cv2.rectangle(imgCopy, (x1,upperBorder),(x2,lowerBorder),150,1)
                letters = letters + (imgCopy[upperBorder:lowerBorder, x1:x2],)

        
        dir_name = self.pathToImage.split('.')[0] # switch to self.contours later
        for idx, letter in enumerate(letters):
            # cv2.imshow("cropped_" + str(idx), letter)
            pathlib2.Path(pathToSave, dir_name).mkdir(parents=True, exist_ok=True) # mkdir -p
            cv2.imwrite(os.path.join(pathToSave,dir_name,"cropped_" + str(idx)+".jpg"), letter)
            # cv2.moveWindow("cropped_" + str(idx), 30*(1+idx), 30*(1+idx))
        # cv2.imshow(self.pathToImage, imgCopy)
        # cv2.waitKey()
        cv2.destroyAllWindows()
        self.letters = letters

    # TODO: make tuning easier and add cropping
    def contourExample(self, pathToMultipleWordsImage, outPath, maxH=100, maxW=200, minH=40, minW=40):
        '''
        https://stackoverflow.com/questions/23506105/extracting-text-opencv

        answered May 9 '14 at 5:07 by anana
        accessed Oct 10 '17 at 18:46
        '''
        self.pathToMultipleWordsImage = pathToMultipleWordsImage
        image = cv2.imread(self.pathToMultipleWordsImage)
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
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
            self.contours = self.contours + (gray[y:y+h,x:x+w],)


        for idx, c in enumerate(self.contours):
            # cv2.imshow("boundingRectangle_" + str(idx), c)
            cv2.moveWindow("boundingRectangle_" + str(idx), 1400, (1000-50*idx))
            # TODO: make saving more elastic
            cv2.imwrite(os.path.join(outPath,"boundingRectangle_" + str(len(self.contours)-idx-1) + ".jpg"), c) # it was writing in reverse order
        # write original image with added contours to disk  
        cv2.imwrite("contoured_models.jpg", image)
        # cv2.imshow(self.pathToMultipleWordsImage, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

def main(singleWord, multipleWords):
    t = TextExtractor(singleWord)
    mypath = "/home/kkuczaj/Praca_inzynierska/VariousMLTforOCR/app_data/separated_words"
    t.contourExample(multipleWords, mypath, 100, 200, 40, 40) # original values 300, 300, 40, 40
    onlyfiles = glob.glob(os.path.join(mypath,"*.jpg"))
    # t.characterExtraction(mypath)
    for c in onlyfiles:
        print("Separating letters in " + str(c))
        t.setPathToImage(str(os.path.abspath(c)))
        t.characterExtraction("/home/kkuczaj/Praca_inzynierska/VariousMLTforOCR/app_data/separated_chars")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToSingleWordImage", help="Path to single word image")
    parser.add_argument("pathToMultipleWordsImage", help="Path to multiple words image")
    parser.add_argument("pathToLogFile", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFile, 'logFile.log'))#, level = logging.INFO)
    main(args.pathToSingleWordImage, args.pathToMultipleWordsImage)
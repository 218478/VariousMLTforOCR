

import numpy as np
import cv2
import sys, cv2, os, argparse, logging

# TODO: make tuning easier
def contourExample(pathToImage):
    '''
    https://stackoverflow.com/questions/23506105/extracting-text-opencv

    answered May 9 '14 at 5:07 by anana
    accessed Oct 10 '17 at 18:46
    '''
    image = cv2.imread(pathToImage)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
    _,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
    _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)

        # discard areas that are too large
        if h>300 and w>300:
            continue

        # discard areas that are too small
        if h<40 or w<40:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)

    # write original image with added contours to disk  
    cv2.imwrite("contoured_models.jpg", image)
    cv2.imshow(pathToImage, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def characterExtraction(pathToImage):
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
    img = cv2.imread(pathToImage, 0)
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
    # thread problem cv2.namedWindow(pathToImage, cv2.WINDOW_NORMAL) # lets you resize the window

    # TODO: remove printing when sure it's doing proper OCR

    for row in img:
        for cell in row:
            sys.stdout.write("%d" % cell)
        sys.stdout.write("\n")

    scanningLetter = False
    zeroResult = False
    upperBorder = 0
    lowerBorder = 0
    margin = 2

    # detect the height of the words start
    for row in range(img.shape[0]):
        print(np.dot(np.ones((1 ,img.shape[1])),img[row,:]))
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
        print(np.dot(np.ones((1 ,img.shape[0])),img[:,col]))
        zeroResult = np.isclose(np.dot(np.ones((1 ,img.shape[0])),img[:,col]), 0)
        if not zeroResult and not scanningLetter: # letter start
            x1 = col
            if x1 > margin:
                x1 -= margin
            scanningLetter = True
            print("poczatek literki")
        if zeroResult and scanningLetter: # letter end
            x2 = col
            if col < img.shape[1] - margin:
                x2 += margin
            scanningLetter = False
            print("koniec literki")
            print("szerokosc: %d" % (x2-x1))

            # cv2.rectangle(imgCopy, (x1,upperBorder),(x2,lowerBorder),150,1)
            letters = letters + (imgCopy[upperBorder:lowerBorder, x1:x2],)

    # for function verification
    for idx, letter in enumerate(letters):
        cv2.imshow("cropped_" + str(idx), letter)
        cv2.imwrite("cropped_" + str(idx)+".jpg", letter)
    # cv2.imshow(pathToImage, imgCopy)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return letters

def main(pathToImage):
    # contourExample(pathToImage)
    characterExtraction(pathToImage)
    exit()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToImage", help="Path to scanned image")
    parser.add_argument("pathToLogFile", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFile, 'logFile.log'))#, level = logging.INFO)
    main(args.pathToImage)
from keras import backend as K
import keras
from cnn import modelCNN
# from mlp import modelMLP
import numpy as np
import argparse, logging, os, sys, math, json, cv2
from PIL import Image, ImageOps
import cv2


# TODO: making it a general reader??
class Reader_Chars74K:
    def setFilepaths(self, filepath, classNo):
        """
        Reads Chars74K dataset and returns 2D array of filepaths. The first
        value is the class no (derived from the directory name) and the second
        one holds a vector of filepaths to the files.
        """
        dirs = os.listdir(filepath)
        print(("Read " + str(len(dirs)) + " classes"))
        self.classNo = classNo
        filepaths = [[]]*self.classNo # hard coded
        i = 0
        for root, dirs, files in os.walk(filepath):
            path = root.split(os.sep)
            print(((len(path) - 1) * '---', os.path.basename(root)))
            filepathsForSpecificClass = []
            for file in files:
                file = os.path.join(root, file)
                filepathsForSpecificClass.append(file)
            if len(filepathsForSpecificClass) is not 0:
                currentScannedClass = int(path[-1][-3:]) - 1 # because the number is starting from 1
                filepaths[currentScannedClass] = filepathsForSpecificClass
        self.filepaths = filepaths
        self.createReadableLabels()

    def createReadableLabels(self):
        """
        This function describes, and assigns the class to the number.
        Specific to Chars74K dataset.
        """
        self.readableLabels = [[]]*self.classNo
        for i in range(0,10):
            self.readableLabels[i] = str(i)
        for i in range(65,91):
            self.readableLabels[i-55] =  chr(i)
        for i in range(97,123):
            self.readableLabels[i-61] =  chr(i)

    def getWhiteImageBlackBackground(self, image):
        """
        This function asserts if image has a black (0) or white (255) background.
        And then either inverts the colors, or leaves the black background.
        """
        if np.bincount(np.array(image).flatten()).argmax() == 255:
            return ImageOps.invert(image)
        else:
            return image

    def loadImagesIntoMemory(self, trainSetProportion, maxsize):
        """
        Returns a tuple of images. trainSetProportion must be between (0; 1.0). It describes
        what part of the images are used for training and what part for testing.

        Color inversion happens on the fly.

        It has the fancy progress bar which is 37 characters wide.
        !!! This function also makes sure the images are converted to maxsize (usually 16x16) but
        this can be changed by setting the maxsize variable !!!
        """
        counts = np.empty((len(self.filepaths),1))
        for idx, val in enumerate(self.filepaths): # counting images in every class
            counts[idx] = len(val) # TODO: use this value in the for loop below and use list comprehension
        print(("Filenames array size: " + str((sys.getsizeof(self.filepaths[0]) + sys.getsizeof(self.filepaths[1]))*self.classNo/1024) + " kB"))
        print(("Read: " + str(len(self.filepaths[1]*len(self.filepaths)))))
        print ("Reading images into memory")
        print(("I have %d classes" % self.classNo))

        toolbar_width = self.classNo - 1
        self.trainCountPerClass = np.ceil(counts*trainSetProportion).astype(int)
        self.testCountPerClass = (counts - self.trainCountPerClass).astype(int)

        self.trainSet = np.empty((sum(self.trainCountPerClass)[0],maxsize[0],maxsize[1]))
        self.trainLabels = np.empty((sum(self.trainCountPerClass)[0]))
        self.testSet = np.empty((sum(self.testCountPerClass)[0],maxsize[0],maxsize[1]))
        self.testLabels = np.empty((sum(self.testCountPerClass)[0]))
        print(("Shape of trainDataset before reading: " + str(self.trainSet.shape)))
        print(("Shape of testDataset before reading: " + str(self.testSet.shape)))
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        for imgClass in range(0, self.classNo-1):
            idx = 0
            for filepath in self.filepaths[imgClass]:
                image = Image.open(filepath, mode="r").convert('LA')
                image = cv2.imread(filepath)
                # image.thumbnail(maxsize, Image.ANTIALIAS)
                # image = self.getWhiteImageBlackBackground(image) # negates
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("not negated", image)
                # self.printImageArray(image)
                # print("\n")
                _,image = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)
                # cv2.imshow("negated", image)
                image = np.array(image)
                # self.printImageArray(image)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                # print(image.shape)
                # exit()
                if idx < self.trainCountPerClass[imgClass]:
                    # print(self.trainSet[imgClass*self.trainCountPerClass[imgClass]+idx].shape)
                    # print(image.shape)
                    self.trainSet[imgClass*self.trainCountPerClass[imgClass]+idx] = image
                    # print(self.trainSet[imgClass*self.trainCountPerClass[imgClass]+idx])
                    self.trainLabels[imgClass*self.trainCountPerClass[imgClass]+idx] = imgClass
                else:
                    self.testSet[imgClass*self.testCountPerClass[imgClass] + idx-self.trainCountPerClass[imgClass]] = image
                    self.testLabels[imgClass*self.testCountPerClass[imgClass] + idx-self.trainCountPerClass[imgClass]] = imgClass
                idx += 1
            sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("\n")
        print(("Shape of read trainDataset: " + str(self.trainSet.shape)))
        print(("Shape of read testDataset: " + str(self.testSet.shape)))

    def saveArrayToFile(self, outfile):
        for_saving = np.array((self.trainLabels, self.trainSet, self.testLabels, self.testSet))
        outfile = open(outfile, 'wb')
        np.save(outfile, for_saving)
        outfile.close()

    def loadArraysFromFile(self, infile):
        infile = open(infile, 'rb')
        self.trainLabels, self.trainSet, self.testLabels, self.testSet = np.load(infile)
        print ("Loaded")
        print(("Length of training set: " + str(len(self.trainSet))))
        print(("Length of test set: " + str(len(self.testSet))))

    def reshapeData(self, maxsize):
        img_rows = maxsize[0]
        img_cols = maxsize[1]
        if K.image_data_format() == 'channels_first':
            self.trainSet = self.trainSet.reshape(self.trainSet.shape[0], 1, img_rows, img_cols)
            self.testSet = self.testSet.reshape(self.testSet.shape[0], 1, img_rows, img_cols)
        else:
            self.trainSet = self.trainSet.reshape(self.trainSet.shape[0], img_rows, img_cols, 1)
            self.testSet = self.testSet.reshape(self.testSet.shape[0], img_rows, img_cols, 1)

        # print(("Shape after reshape: " + str(self.trainSet.shape[0])))
        self.trainSet = self.trainSet.astype('float32')
        self.testSet = self.testSet.astype('float32')
        self.trainSet /= 255.0 # this was 255.0
        self.testSet /= 255.0 # this was 255.0
        print(('self.trainSet shape:', self.trainSet.shape))
        print((self.trainSet.shape[0], 'train samples'))
        print(('self.testSet shape:', self.testSet.shape))
        print((self.testSet.shape[0], 'test samples'))

        self.trainLabels = keras.utils.to_categorical(self.trainLabels, self.classNo)
        self.testLabels = keras.utils.to_categorical(self.testLabels, self.classNo)

    def printImageArray(self, img):
        for row in img:
            for cell in row:
                sys.stdout.write("%d " % cell)
            sys.stdout.write("\n")

def main(filepath):
    # CONSTANTS
    batch_size = 256
    epochs = 6
    maxsize = (16, 16)
    classNo = 62

    r = Reader_Chars74K()
    r.setFilepaths(filepath, classNo)
    r.loadImagesIntoMemory(0.9, maxsize)
    outfile = "temp_to_save_np_array_for_my_dataset.temp"
    r.saveArrayToFile(outfile)
    # exit()
    # r.loadArraysFromFile(outfile)
    r.reshapeData(maxsize)

    # temp1 = r.testSet[200]
    # for row in temp1:
    #     for cell in row:
    #         sys.stdout.write("%d " % cell)
    #     sys.stdout.write("\n")
    # print(r.testLabels[200])
    # exit()
    # temp2 = r.testSet[4816]/32
    # for row in temp2:
    #     for cell in row:
    #         if np.isclose(cell, 0):
    #             sys.stdout.write("  ")
    #         else:
    #             sys.stdout.write("%d " % cell)
    #     sys.stdout.write("\n")
    # print(r.testLabels[4816])

    model = modelCNN(maxsize, classNo)#,"trained_model_for_my_dataset.h5")
    # model = modelMLP(maxsize, classNo)#, "trained_model.h5") it works but needs file to be imported
    model.fit(r.trainSet, r.testSet, r.trainLabels, r.testLabels, batch_size, epochs)
    model.saveKerasModel("trained_model_for_my_dataset5.h5")
    values = model.predict(r.testSet[2040].reshape(16,16))
    print(values)
    print((values.argmax()))
    print((r.testLabels[2040].argmax()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToDatasets", help="Directory to stored datasets")
    parser.add_argument("pathToLogFileDir", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFileDir, 'logFile.log'))#, level = logging.INFO)
    main(args.pathToDatasets)

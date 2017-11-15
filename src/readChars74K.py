from keras import backend as K
from cnn import modelCNN
from mlp import modelMLP
from cnn_different import modelCNN2
import numpy as np
import argparse, logging, os, sys, math, json, cv2, keras
from PIL import Image, ImageOps
from knn import kNN


# TODO: making it a general reader??
class Reader_Chars74K:
    def setFilepaths(self, filepath, classNo):
        """
        Reads Chars74K dataset and returns 2D array of filepaths. The first
        value is the class no (derived from the directory name) and the second
        one holds a vector of filepaths to the files. The reader assumes that
        the images read have white background and black font
        """
        dirs = os.listdir(filepath)
        print(("Read " + str(len(dirs)) + " classes"))
        self.classNo = classNo
        filepaths = [[]]*self.classNo # hard coded
        i = 0
        for root, dirs, files in os.walk(filepath):
            path = root.split(os.sep)
            # print(((len(path) - 1) * '---', os.path.basename(root)))
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
        print("len(self.filepaths[1]) = " + str(len(self.filepaths[1])))
        print("len(self.filepaths) = " + str(len(self.filepaths)))
        print(("Read: " + str(len(self.filepaths[1])*len(self.filepaths))))
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
            for idx, filepath in enumerate(self.filepaths[imgClass]):
                image = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)

                # IMPORTANT!!! EXPECTING BLACK FONT WITH WHITE BACKGROUND
                _,image = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)
                image = cv2.resize(image,(maxsize[1],maxsize[0]), interpolation = cv2.INTER_AREA)
                image = np.array(image)
                if self.imageIsNotValid(image):
                    pass

                if idx < self.trainCountPerClass[imgClass]:
                    self.trainSet[imgClass*self.trainCountPerClass[imgClass]+idx] = image
                    self.trainLabels[imgClass*self.trainCountPerClass[imgClass]+idx] = imgClass
                else:
                    self.testSet[imgClass*self.testCountPerClass[imgClass] + idx-self.trainCountPerClass[imgClass]] = image
                    self.testLabels[imgClass*self.testCountPerClass[imgClass] + idx-self.trainCountPerClass[imgClass]] = imgClass
                idx += 1
            sys.stdout.write("-")
            sys.stdout.flush()
            # self.printImageArray(image)
            # cv2.imshow("test",image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

        sys.stdout.write("\n")
        print(("Shape of read trainDataset: " + str(self.trainSet.shape)))
        print(("Shape of read testDataset: " + str(self.testSet.shape)))

    def imageIsNotValid(self, image):
        """
        Checks if the image contains NaN or Inf records.
        """
        return np.isinf(image).any() or np.isnan(image).any()

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

    def reshapeDataForMLP(self, maxsize):
        self.trainSet = self.trainSet.reshape(len(self.trainSet),maxsize[0]*maxsize[1])
        self.testSet = self.testSet.reshape(len(self.testSet),maxsize[0]*maxsize[1])

        self.trainSet = self.trainSet.astype('float32')
        self.testSet = self.testSet.astype('float32')
        self.trainSet /= 255
        self.testSet /= 255

        self.trainLabels = keras.utils.to_categorical(self.trainLabels, self.classNo)
        self.testLabels = keras.utils.to_categorical(self.testLabels, self.classNo)

    def reshapeDataForCNN(self, maxsize):
        img_rows, img_cols = maxsize
        if K.image_data_format() == 'channels_first':
            self.trainSet = self.trainSet.reshape(self.trainSet.shape[0], 1, img_rows, img_cols)
            self.testSet = self.testSet.reshape(self.testSet.shape[0], 1, img_rows, img_cols)
        else:
            self.trainSet = self.trainSet.reshape(self.trainSet.shape[0], img_rows, img_cols, 1)
            self.testSet = self.testSet.reshape(self.testSet.shape[0], img_rows, img_cols, 1)

        # print(("Shape after reshape: " + str(self.trainSet.shape[0])))
        self.trainSet = self.trainSet.astype('float32')
        self.testSet = self.testSet.astype('float32')
        self.printImageArray(self.testSet[0])
        self.trainSet /= 255
        self.testSet /= 255
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

    def testLetterFromTestSet(self, model, n):
        image = self.testSet[n]
        cv2.imshow("test",image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        values = model.predict(image, using_training_set=True)
        print("Predicted: " + str(values))
        print("Should be: " + str((self.testLabels[n].argmax())))

def main(filepath):
    batch_size = 128
    epochs = 5
    maxsize = (64, 64)
    classNo = 62

    r = Reader_Chars74K()
    r.setFilepaths(filepath, classNo)
    # r.loadImagesIntoMemory(0.9, maxsize)
    outfile = "temp_to_save_np_array.temp"
    # r.saveArrayToFile(outfile)
    r.loadArraysFromFile(outfile)
    r.reshapeDataForCNN(maxsize)
    # r.reshapeDataForMLP(maxsize)
    # k = kNN()
    # r.printImageArray(k._createPatternSetForKNN(r.trainSet, classNo)[2])
    # exit()

    model = modelCNN(maxsize, classNo,"cnn_model_for_my_dataset_64x64_2.h5")
    # model = modelMLP(maxsize, classNo)#, "mlp_model_for_my_dataset.h5")
    # model.fit(r.trainSet, r.testSet, r.trainLabels, r.testLabels, batch_size, epochs)
    # model.saveKerasModel("cnn_model_for_my_dataset_64x64_2.h5")
    r.testLetterFromTestSet(model, 281)
    r.testLetterFromTestSet(model, 28)
    r.testLetterFromTestSet(model, 57)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToDatasets", help="Directory to stored datasets")
    parser.add_argument("pathToLogFileDir", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFileDir, 'logFile.log'))#, level = logging.INFO)
    main(args.pathToDatasets)

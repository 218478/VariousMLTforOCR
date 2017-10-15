from __future__ import print_function

from tempfile import TemporaryFile
import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import argparse, logging, os, sys, math, json, shutil
from PIL import Image, ImageOps


# CONSTANTS
batch_size = 128
epochs = 4

maxsize = (16, 16)
classNo = 62
toolbar_width = classNo


MYPATH = '/home/kkuczaj/Praca_inzynierska/VariousMLTforOCR/datasets/Chars74K/English/Fnt'
TEST_PATH = "C:\Users\kamil\Pictures\database2\/test"
TRAIN_PATH = "C:\Users\kamil\Pictures\database2\/train"
    



class Reader_Chars74K:
    def __init__(self, filepath):
        """
        Reads Chars74K dataset and returns 2D array of filepaths. The first
        value is the class no (derived from the directory name) and the second
        one holds a vector of filepaths to the files.
        """
        dirs = os.listdir(filepath)
        print ("Read " + str(len(dirs)) + " classes")
        global classNo # because it was referenced before assignment
        filepaths = [[]]*classNo # hard coded
        i = 0
        for root, dirs, files in os.walk(filepath):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            filepathsForSpecificClass = []
            for file in files:
                # print(len(path) * '---', file)
                file = os.path.join(root, file)
                filepathsForSpecificClass.append(file)
            if len(filepathsForSpecificClass) is not 0:
                currentScannedClass = int(path[-1][-3:]) - 1 # because the number is starting from 1
                filepaths[currentScannedClass] = filepathsForSpecificClass
        self.filepaths = filepaths
    
    def createLabels(self):
        """
        This function describes, and assigns the class to the number.
        Specific to Chars74K dataset.
        """
        myLabels = [str(i) for i in range(0,10)]
        myLabels.append(["capital_" + chr(i) for i in range(65,91)])
        myLabels.append(["small_" + chr(i) for i in range(97,123)])
        self.labels = myLabels

    def getWhiteImageBlackBackground(self, image):
        """
        This function asserts if image has a black (0) or white (255) background.
        And then either inverts the colors, or leaves the black background.
        """
        if np.bincount(np.array(image).flatten()).argmax() == 255:
            return ImageOps.invert(image)
        else:
            return image

    def getTupleOfImages(self, trainSetProportion):
        """
        Returns a tuple of images. trainSetProportion must be between (0; 1.0). It describes
        what part of the images are used for training and what part for testing.

        Color inversion happens on the fly.

        It has the fancy progress bar which is 37 characters wide.
        !!! This function also makes sure the images are converted to maxsize (usually 16x16) but
        this can be changed by setting the maxsize variable !!!
        """
        counts = np.zeros([len(self.filepaths[0]),1])
        global classNo # because it was referenced before assignment
        for idx, val in enumerate(self.filepaths): # counting images in every class
            counts[idx] = len(val) # TODO: use this value in the for loop below and use list comprehension
        print ("Filenames array size: " + str((sys.getsizeof(self.filepaths[0]) + sys.getsizeof(self.filepaths[1]))*classNo/1024) + " kB")
        print ("Read: " + str(len(self.filepaths[1]*len(self.filepaths))))
        print ("Reading images into memory")
        print("I have %d classes" % classNo)

        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        # trainCount = int(math.floor(int(sum(counts))*trainSetProportion))
        # testCount = int(sum(counts)) - trainCount
        # TODO: please change it
        trainCountPerClass = 900
        testCountPerClass = 116

        trainSet = np.empty((classNo*trainCountPerClass,maxsize[0],maxsize[1])) # TODO: remove hard coding
        trainLabels = np.empty((trainCountPerClass*classNo))
        testSet = np.empty((classNo*testCountPerClass,maxsize[0],maxsize[1])) # TODO: remove hard coding
        testLabels = np.empty((testCountPerClass*classNo))
        for imgClass in range(0, classNo-1): # TODO: change 
            idx = 0
            for filepath in self.filepaths[imgClass]:
                image = Image.open(filepath, mode="r")
                image.thumbnail(maxsize, Image.ANTIALIAS)
                image = self.getWhiteImageBlackBackground(image) # negates
                if idx < trainCountPerClass:
                    trainSet[imgClass*trainCountPerClass+idx] = np.array(image)
                    trainLabels[imgClass*trainCountPerClass+idx] = imgClass
                else:
                    testSet[imgClass*testCountPerClass + idx-trainCountPerClass] = np.array(image)
                    testLabels[imgClass*testCountPerClass + idx-trainCountPerClass] = imgClass
                idx += 1
            sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("\n")
        print ("Length of read trainDataset: " + str(trainSet.shape))
        print ("Length of read testDataset: " + str(testSet.shape))
        return trainLabels, trainSet, testLabels, testSet

    def loadImages(self, (trainLabels, trainSet, testLabels, testSet)):
        self.trainLabels = trainLabels
        self.trainSet = trainSet
        self.testLabels = testLabels
        self.testSet = testSet

    def saveArrayToFile(self, outfile):
        for_saving = np.array((self.trainLabels, self.trainSet, self.testLabels, self.testSet))
        outfile = file(outfile, 'w')
        np.save(outfile, for_saving)
        outfile.close()

    def loadArraysFromFile(self, infile):
        inffile = file(infile, 'r')
        self.trainLabels, self.trainSet, self.testLabels, self.testSet = np.load(infile)


def main():
    r = Reader_Chars74K(MYPATH)
    # r.loadImages(r.getTupleOfImages(0.8))
    # outfile = "temp_to_save_np_array.temp"
    # r.saveArrayToFile(outfile)
    # exit()
    outfile = "temp_to_save_np_array.temp"
    # outfile = file(outfile, 'r')
    # trainLabels, trainSet, testLabels, testSet = np.load(outfile)
    r.loadArraysFromFile(outfile)
    print ("Loaded")
    print ("Length of training set: " + str(len(r.trainSet)))
    print ("Length of test set: " + str(len(r.testSet)))
    print ("Shape before reshape: " + str(r.trainSet.shape[0]))
    img_rows = maxsize[0]
    img_cols = maxsize[1]
    if K.image_data_format() == 'channels_first':
        r.trainSet = r.trainSet.reshape(r.trainSet.shape[0], 1, img_rows, img_cols)
        r.testSet = r.testSet.reshape(r.testSet.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        r.trainSet = r.trainSet.reshape(r.trainSet.shape[0], img_rows, img_cols, 1)
        r.testSet = r.testSet.reshape(r.testSet.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print ("Shape after reshape: " + str(r.trainSet.shape[0]))
    r.trainSet = r.trainSet.astype('float32')
    r.testSet = r.testSet.astype('float32')
    r.trainSet /= 255
    r.testSet /= 255
    print('r.trainSet shape:', r.trainSet.shape)
    print(r.trainSet.shape[0], 'train samples')
    print('r.testSet shape:', r.testSet.shape)
    print(r.testSet.shape[0], 'test samples')

    r.trainLabels = keras.utils.to_categorical(r.trainLabels, classNo)
    r.testLabels = keras.utils.to_categorical(r.testLabels, classNo)


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classNo, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(r.trainSet, r.trainLabels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(r.testSet, r.testLabels))
    score = model.evaluate(r.testSet, r.testLabels, verbose=0)
    model.save("trained_model.h5")
    json_string = model.to_json()
    with open("data.txt",'w') as jfile:
        json.dump(json_string, jfile)        
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    values = model.predict(r.trainSet[2040].reshape(1,16,16,1), verbose=1)
    print(values.argmax())
    exit()

if __name__ == '__main__':
    # from appJar import gui
    # # create a GUI variable called app
    # app = gui()
    # app.addLabel("title", "Welcome to appJar")
    # app.setLabelBg("title", "red")
    # app.go()
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToDatasets", help="Directory to stored datasets")
    parser.add_argument("pathToLogFileDir", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFileDir, 'logFile.log'))#, level = logging.INFO)
    main()

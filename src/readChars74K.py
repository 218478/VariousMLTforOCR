from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import argparse, logging, os, sys
import shutil
import PIL
from PIL import Image


# CONSTANTS
batch_size = 128
num_classes = 10
epochs = 1

maxsize = (16, 16)
classNo = 62
toolbar_width = 37


MYPATH = '/home/kkuczaj/Praca_inzynierska/VariousMLTforOCR/datasets/Chars74K/English/Fnt'
TEST_PATH = "C:\Users\kamil\Pictures\database2\/test"
TRAIN_PATH = "C:\Users\kamil\Pictures\database2\/train"

def getFilepathsToFilesInChars74(training = True):
    """
    Reads Chars74K dataset and returns 2D array of filepaths. The first
    value is the class no (derived from the directory name) and the second
    one holds a vector of filepaths to the files.
    """
    dirs = os.listdir(MYPATH)
    print ("Read " + str(len(dirs)) + " classes")
    global classNo # because it was referenced before assignment
    filepaths = [[]]*classNo # hard coded
    i = 0
    for root, dirs, files in os.walk(MYPATH):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        filepathsForSpecificClass = []
        for file in files:
            # print(len(path) * '---', file)
            file = os.path.join(root, file)
            filepathsForSpecificClass.append(file)
        if len(filepathsForSpecificClass) is not 0:
            classNo = int(path[-1][-3:]) - 1 # because the number is starting from 1
            filepaths[classNo] = filepathsForSpecificClass
    return filepaths

def getTupleOfImages(trainSetProportion):
    """
    Returns a tuple of images. trainSetProportion must be between (0; 1.0). It describes
    what part of the images are used for training and what part for testing.

    It has the fancy progress bar which is 37 characters wide.
    !!! This function also makes sure the images are converted to maxsize (usually 16x16) but
    this can be changed by setting the maxsize variable !!!
    """
    filepaths = getFilepathsToFilesInChars74()
    print ("Filenames array size: " + str((sys.getsizeof(filepaths[0]) + sys.getsizeof(filepaths[1]))*62/1024) + " kB")
    print ("Reading images into memory")
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    images = list()
    labels = list()
    global classNo # because it was referenced before assignment
    for classNo in xrange(0,3):#classNo-1): # TODO: change 
        for filepath in filepaths[classNo]:
            image = Image.open(filepath, mode="r")
            image.thumbnail(maxsize, Image.ANTIALIAS)
            labels.append(classNo)
            images.append(np.array(image))
        sys.stdout.write("-")
        sys.stdout.flush()
    
    sys.stdout.write("\n")
    # length is ok when dividing this way
    print ("Length of read dataset: " + str(len(images)))
    # print len(images[0:int(len(images)*trainSetProportion)])+len(images[int(len(images)*trainSetProportion):])
    return labels[0:int(len(labels)*trainSetProportion)], images[0:int(len(images)*trainSetProportion)], \
           labels[int(len(labels)*trainSetProportion):], images[int(len(images)*trainSetProportion):]

# hard-coded label creation TODO: I think this can be deleted
def createLabels():
    """
    This function describes, and assigns the class to the number.
    Specific to Chars74K dataset.
    """
    myLabels = []
    for i in range(0,10):
        myLabels.append(str(i))
    for i in range(65,91):
        myLabels.append("capital_" + chr(i))
    for i in range(97,123):
        myLabels.append("small_" + chr(i))
    return myLabels


# TODO: how about making the letter white and background black????
def main():
    trainLabels, trainSet, testLabel, testSet = getTupleOfImages(0.8)
    print ("Shape: " + trainSet.shape)
    print ("Length of training set: " + str(len(trainSet)))
    print ("Length of test set: " + str(len(testSet)))
    if K.image_data_format() == 'channels_first':
        trainSet = trainSet.reshape(trainSet.shape[0], 1, img_rows, img_cols)
        testSet = testSet.reshape(testSet.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        trainSet = trainSet.reshape(trainSet.shape[0], img_rows, img_cols, 1)
        testSet = testSet.reshape(testSet.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    trainSet = trainSet.astype('float32')
    testSet = testSet.astype('float32')
    trainSet /= 255
    testSet /= 255
    print('trainSet shape:', trainSet.shape)
    print(trainSet.shape[0], 'train samples')
    print(testSet.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

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
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(trainSet, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(testSet, y_test))
    score = model.evaluate(testSet, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    

if __name__ == '__main__':
    # from appJar import gui
    # # create a GUI variable called app
    # app = gui()
    # app.addLabel("title", "Welcome to appJar")
    # app.setLabelBg("title", "red")
    # app.go()

    parser = argparse.ArgumentParser()
    parser.add_argument("pathToDatasets", help="Directory to stored datasets")
    parser.add_argument("pathToLogFile", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFile, 'logFile.log'))#, level = logging.INFO)
    main()

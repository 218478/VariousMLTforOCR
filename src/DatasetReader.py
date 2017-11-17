import argparse
import cv2
import keras
import os
import sys

import numpy as np
from keras import backend as K

from NearestNeighbor import NearestNeighbor


def image_invalid(image):
    """
    Checks if the image contains NaN or Inf records.
    """
    return np.isinf(image).any() or np.isnan(image).any()


def print_image_array(img):
    for row in img:
        for cell in row:
            sys.stdout.write("%d " % cell)
        sys.stdout.write("\n")


class DatasetReader:
    def set_filepaths(self, filepath, class_no):
        """
        Reads Chars74K dataset and returns 2D array of filepaths. The first
        value is the class no (derived from the directory name) and the second
        one holds a vector of filepaths to the files. The reader assumes that
        the images read have white background and black font
        """
        dirs = os.listdir(filepath)
        print(("Read " + str(len(dirs)) + " classes"))
        self.classNo = class_no
        filepaths = [[]] * self.classNo  # hard coded
        for root, dirs, files in os.walk(filepath):
            path = root.split(os.sep)
            # print(((len(path) - 1) * '---', os.path.basename(root)))
            filepaths_for_class = []
            for file in files:
                file = os.path.join(root, file)
                filepaths_for_class.append(file)
            if len(filepaths_for_class) is not 0:
                current_class = int(path[-1][-3:]) - 1  # because the number is starting from 1
                filepaths[current_class] = filepaths_for_class
        self.filepaths = filepaths
        self.create_readable_labels()

    def create_readable_labels(self):
        """
        This function describes, and assigns the class to the number.
        Specific to Chars74K dataset.
        """
        self.readableLabels = [[]] * self.classNo
        for i in range(0, 10):
            self.readableLabels[i] = str(i)
        for i in range(65, 91):
            self.readableLabels[i - 55] = chr(i)
        for i in range(97, 123):
            self.readableLabels[i - 61] = chr(i)

    def load_images_into_memory(self, train_to_test_proportion, maxsize):
        """
        Returns a tuple of images. train_setProportion must be between (0; 1.0). It describes
        what part of the images are used for training and what part for testing.

        Color inversion happens on the fly.

        It has the fancy progress bar which is 37 characters wide.
        !!! This function also makes sure the images are converted to maxsize (usually 16x16) but
        this can be changed by setting the maxsize variable !!!
        """
        counts = np.empty((len(self.filepaths), 1))
        for idx, val in enumerate(self.filepaths):  # counting images in every class
            counts[idx] = len(val)  # TODO: use this value in the for loop below and use list comprehension
        print(("Filenames array size: " + str(
            (sys.getsizeof(self.filepaths[0]) + sys.getsizeof(self.filepaths[1])) * self.classNo / 1024) + " kB"))
        print("len(self.filepaths[1]) = " + str(len(self.filepaths[1])))
        print("len(self.filepaths) = " + str(len(self.filepaths)))
        print(("Read: " + str(len(self.filepaths[1]) * len(self.filepaths))))
        print("Reading images into memory")
        print(("I have %d classes" % self.classNo))

        toolbar_width = self.classNo - 1
        self.trainCountPerClass = np.ceil(counts * train_to_test_proportion).astype(int)
        self.testCountPerClass = (counts - self.trainCountPerClass).astype(int)

        self.train_set = np.empty((sum(self.trainCountPerClass)[0], maxsize[0], maxsize[1]))
        self.train_labels = np.empty((sum(self.trainCountPerClass)[0]))
        self.test_set = np.empty((sum(self.testCountPerClass)[0], maxsize[0], maxsize[1]))
        self.test_labels = np.empty((sum(self.testCountPerClass)[0]))
        print(("Shape of trainDataset before reading: " + str(self.train_set.shape)))
        print(("Shape of testDataset before reading: " + str(self.test_set.shape)))
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

        for imgClass in range(0, self.classNo - 1):
            for idx, filepath in enumerate(self.filepaths[imgClass]):
                image = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)

                # IMPORTANT!!! EXPECTING BLACK FONT WITH WHITE BACKGROUND
                _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
                image = cv2.resize(image, (maxsize[1], maxsize[0]), interpolation=cv2.INTER_AREA)
                image = np.array(image)
                if image_invalid(image):
                    pass

                if idx < self.trainCountPerClass[imgClass]:
                    self.train_set[imgClass * self.trainCountPerClass[imgClass] + idx] = image
                    self.train_labels[imgClass * self.trainCountPerClass[imgClass] + idx] = imgClass
                else:
                    self.test_set[
                        imgClass * self.testCountPerClass[imgClass] + idx - self.trainCountPerClass[imgClass]] = image
                    self.test_labels[imgClass * self.testCountPerClass[imgClass] + idx - self.trainCountPerClass[
                        imgClass]] = imgClass
                idx += 1
            sys.stdout.write("-")
            sys.stdout.flush()
            # self.printImageArray(image)
            # cv2.imshow("test",image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

        sys.stdout.write("\n")
        print(("Shape of read trainDataset: " + str(self.train_set.shape)))
        print(("Shape of read testDataset: " + str(self.test_set.shape)))

    def save_array_to_file(self, outfile):
        for_saving = np.array((self.train_labels, self.train_set, self.test_labels, self.test_set))
        outfile = open(outfile, 'wb')
        np.save(outfile, for_saving)
        outfile.close()

    def load_arrays_from_file(self, infile):
        infile = open(infile, 'rb')
        self.train_labels, self.train_set, self.test_labels, self.test_set = np.load(infile)
        print("Loaded")
        print(("Length of training set: " + str(len(self.train_set))))
        print(("Length of test set: " + str(len(self.test_set))))

    def reshape_data_for_mlp(self, maxsize):
        self.train_set = self.train_set.reshape(len(self.train_set), maxsize[0] * maxsize[1])
        self.test_set = self.test_set.reshape(len(self.test_set), maxsize[0] * maxsize[1])

        self.train_set = self.train_set.astype('float32')
        self.test_set = self.test_set.astype('float32')
        self.train_set /= 255
        self.test_set /= 255

        self.train_labels = keras.utils.to_categorical(self.train_labels, self.classNo)
        self.test_labels = keras.utils.to_categorical(self.test_labels, self.classNo)

    def reshape_data_for_knn(self, maxsize):
        img_rows, img_cols = maxsize
        self.train_set = self.train_set.reshape(self.train_set.shape[0], img_rows * img_cols)
        self.test_set = self.test_set.reshape(self.test_set.shape[0], img_rows * img_cols)
        self.train_set = self.train_set.astype(np.float32)
        self.test_set = self.test_set.astype(np.float32)
        self.train_labels = self.train_labels.astype(np.float32)
        self.test_labels = self.test_labels.astype(np.float32)
        self.train_labels = self.train_labels.reshape(self.train_labels.shape[0], 1)
        self.test_labels = self.test_labels.reshape(self.test_labels.shape[0], 1)
        print("self.train_labels[0] = " + str(self.train_labels[0]))
        print(('self.train_labels shape:', self.train_labels.shape))
        print(('self.test_labels shape:', self.test_labels.shape))
        print(('self.train_set shape:', self.train_set.shape))
        print((self.train_set.shape[0], 'train samples'))
        print(('self.test_set shape:', self.test_set.shape))
        print((self.test_set.shape[0], 'test samples'))

    def reshape_data_for_cnn(self, maxsize):
        img_rows, img_cols = maxsize
        if K.image_data_format() == 'channels_first':
            self.train_set = self.train_set.reshape(self.train_set.shape[0], 1, img_rows, img_cols)
            self.test_set = self.test_set.reshape(self.test_set.shape[0], 1, img_rows, img_cols)
        else:
            self.train_set = self.train_set.reshape(self.train_set.shape[0], img_rows, img_cols, 1)
            self.test_set = self.test_set.reshape(self.test_set.shape[0], img_rows, img_cols, 1)

        # print(("Shape after reshape: " + str(self.train_set.shape[0])))
        self.train_set = self.train_set.astype('float32')
        self.test_set = self.test_set.astype('float32')
        print_image_array(self.test_set[0])
        self.train_set /= 255
        self.test_set /= 255
        print(('self.train_set shape:', self.train_set.shape))
        print((self.train_set.shape[0], 'train samples'))
        print(('self.test_set shape:', self.test_set.shape))
        print((self.test_set.shape[0], 'test samples'))

        self.train_labels = keras.utils.to_categorical(self.train_labels, self.classNo)
        self.test_labels = keras.utils.to_categorical(self.test_labels, self.classNo)

    def test_letter_from_test_set_nn(self, model, n):
        image = self.test_set[n]
        cv2.imshow("test", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        values = model.predict(image, using_training_set=True)
        print("Predicted: " + str(values))
        print("Should be: " + str((self.test_labels[n].argmax())))

    def test_letter_from_test_set_knn(self, model, n):
        image = self.test_set[n]
        values = model.predict(image, using_training_set=True)
        print("Predicted: " + str(values))
        print("Should be: " + str(self.test_labels[n]))
        image = cv2.resize(image, (64, 64))
        cv2.imshow("test", image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main(filepath):
    batch_size = 128
    epochs = 5
    maxsize = (64, 64)
    class_no = 62

    r = DatasetReader()
    r.set_filepaths(filepath, class_no)
    r.load_images_into_memory(0.9, maxsize)
    outfile = "temp_to_save_np_array.temp"
    r.save_array_to_file(outfile)
    # r.load_arrays_from_file(outfile)
    r.reshape_data_for_knn(maxsize)
    # r.reshape_data_for_cnn(maxsize)
    # r.reshape_data_for_mlp(maxsize)


    # bestK = 1
    # prevAcc = 0
    # for k in range(1,14):
    #     model = NearestNeighbor(maxsize, k)
    #     model.train(r.train_set, r.train_labels)
    #     acc = model.accuracy(r.test_set, r.test_labels)
    #     if acc > prevAcc:
    #         bestK = k
    #         prevAcc = acc
    # print("best acc (" + str(prevAcc) + ") for " + str(bestK) + " neighbors")


    # best acc (91.45161290322581) for 11 neighbors
    model = NearestNeighbor(maxsize, 11)
    model.train(r.train_set, r.train_labels)
    np.savez('knn_data.npz', train_set=r.train_set, train_labels=r.train_labels)

    with np.load('knn_data.npz') as data:
        print(data.files)

    # model = CNN(maxsize, class_no,"cnn_model_for_my_dataset_64x64_2.h5")
    # model = MLP(maxsize, class_no)#, "mlp_model_for_my_dataset.h5")
    # model.fit(r.train_set, r.test_set, r.train_labels, r.test_labels, batch_size, epochs)
    # model.saveKerasModel("cnn_model_for_my_dataset_64x64_2.h5")
    r.test_letter_from_test_set_knn(model, 281)
    r.test_letter_from_test_set_knn(model, 28)
    r.test_letter_from_test_set_knn(model, 57)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToDatasets", help="Directory to stored datasets")
    args = parser.parse_args()
    main(args.pathToDatasets)

import cv2
import numpy as np
import random

class kNN():
    def __init__(self, maxsize, k = 5):
        self.maxsize = maxsize
        self.knn = cv2.ml.KNearest_create()
        self.k = k

    def save(self, filepath):
        self.knn.save(filepath)

    def load(self, filepath):
        self.knn.load(filepath)

    def train(self, trainSet, trainLabels):
        self.knn.train(trainSet, cv2.ml.ROW_SAMPLE, trainLabels)

    def reshape(self, img):
        img =img.reshape(1, self.maxsize[0]*self.maxsize[1])
        img = img.astype(np.float32)
        return img

    def predict(self, img, using_training_set=False):
        img = self.reshape(img)
        print("My img.shape = " + str(img.shape))
        ret, result, neighbours, dist = self.knn.findNearest(img, self.k)
        return int(result)

    def accuracy(self, testSet, testLabels):
        ret,result,neighbours,dist = self.knn.findNearest(testSet, self.k)
        matches = result==testLabels
        correct = np.count_nonzero(matches)
        accuracy = correct*100.0/result.size
        return accuracy
import cv2
import numpy as np
import random

class kNN():
    def _createPatternSetForKNN(self, trainSet, classNo, n=10):
        """
        Creates pattern for kNN image recognition. Parameter n tells
        how many role models should the algorithm base for each class.

        Expects (n,width,height) trainSet vector
        """
        print("shape of trainSet = " + str(trainSet.shape))
        class_span =int(len(trainSet)/classNo)
        pattern_set = np.empty((classNo*n,trainSet.shape[1],trainSet.shape[2]))
        for i in range(classNo):
            for j in range(n):
                random_sample_for_jth_class = random.randint(i*class_span,(i+1)*class_span-1)
                pattern_set[i*n+j] = trainSet[random_sample_for_jth_class]

        print("pattern_set shape = " + str(pattern_set.shape))
        cv2.imshow("test2",pattern_set[2])
        cv2.imshow("test12",pattern_set[12])
        cv2.imshow("test22",pattern_set[22])
        cv2.waitKey()
        cv2.destroyAllWindows()
        return pattern_set
from nn import myNN
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import *
from keras.regularizers import *
import cv2


class modelMLP(myNN):
    def __init__(self, maxsize, classNo, filepath=None):
        """
        Filename -> link to .h5 file.
        maxsize -> imsize height by width (16 16)
        """
        self.maxsize = maxsize
        if filepath is not None:
            self.loadKerasModel(filepath)
            print("Loaded model from file")
        else:
            img_rows = self.maxsize[0]
            img_cols = self.maxsize[1]
            if K.image_data_format() == 'channels_first':
                input_shape = (1, img_rows*img_cols)
            else:
                input_shape = (img_rows*img_cols, 1)
            self.model = Sequential()
            self.model.add(Dense(classNo, activation='relu',
                            input_shape=(maxsize[0]*maxsize[1],)))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(classNo, activation='softmax'))

            self.model.summary()

            self.model.compile(loss='categorical_crossentropy',
                        optimizer=RMSprop(lr=1),
                        metrics=['accuracy'])

    def fit(self, trainSet, testSet, trainLabels, testLabels, batch_size, epochs):
        self.model.fit(trainSet, trainLabels,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=True,
                  validation_data=(testSet, testLabels))
        score = self.model.evaluate(testSet, testLabels, verbose=0)
        print(('Test loss:', score[0]))
        print(('Test accuracy:', score[1]))

    def predict(self, image):
        """
        image is an opencv image. This function will resize it to the size, when it was trained
        """
        image = cv2.resize(image, (self.maxsize[0],self.maxsize[1]), interpolation=cv2.INTER_AREA)
        values = self.model.predict(image.reshape(1,self.maxsize[0]*self.maxsize[1]), verbose=False)
        return values.argmax()
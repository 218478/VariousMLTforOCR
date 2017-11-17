import cv2
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import *
from keras.regularizers import *

from NNinterface import NNinterface


class MLP(NNinterface):
    def __init__(self, maxsize, class_no, filepath=None):
        """
        Filename -> link to .h5 file.
        maxsize -> imsize height by width (16 16)
        """
        self.maxsize = maxsize
        if filepath is not None:
            self.load_keras_model(filepath)
            print("Loaded model from file")
        else:
            img_rows, img_cols = self.maxsize[0], self.maxsize[1]
            input_shape = img_rows * img_cols
            self.model = Sequential()
            self.model.add(Dense(class_no, activation='relu',
                                 input_shape=(input_shape,)))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(class_no, activation='softmax'))

            self.model.summary()

            self.model.compile(loss='categorical_crossentropy',
                               optimizer=RMSprop(lr=1),
                               metrics=['accuracy'])

    def fit(self, train_set, test_set, train_labels, test_labels, batch_size, epochs):
        self.model.fit(train_set, train_labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=True,
                       validation_data=(test_set, test_labels))
        score = self.model.evaluate(test_set, test_labels, verbose=0)
        print(('Test loss:', score[0]))
        print(('Test accuracy:', score[1]))

    def predict(self, image):
        """
        image is an opencv image. This function will resize it to the size, when it was trained
        """
        image = cv2.resize(image, (self.maxsize[0], self.maxsize[1]), interpolation=cv2.INTER_AREA)
        values = self.model.predict(image.reshape(1, self.maxsize[0] * self.maxsize[1]), verbose=False)
        return values.argmax()

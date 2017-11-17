import keras
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from NNinterface import NNinterface


class CNN(NNinterface):
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
            img_rows = self.maxsize[0]
            img_cols = self.maxsize[1]
            if K.image_data_format() == 'channels_first':
                input_shape = (1, img_rows, img_cols)
            else:
                input_shape = (img_rows, img_cols, 1)
            print(input_shape)
            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=(3, 3),
                                  activation='relu',
                                  input_shape=input_shape, ))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(class_no, activation='softmax'))
            self.model.summary()
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.Adadelta(),
                               metrics=['accuracy'])

    def fit(self, train_set, test_set, train_labels, test_labels, batch_size, epochs):
        self.model.fit(train_set, train_labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(test_set, test_labels))
        score = self.model.evaluate(test_set, test_labels, verbose=0)
        print(('Test loss:', score[0]))
        print(('Test accuracy:', score[1]))

    def predict(self, image, using_training_set=False):
        """
        image is an opencv image. This function will not resize it to the size. It expects inverted
        image and will only.
        """
        img_rows, img_cols = self.maxsize
        if K.image_data_format() == 'channels_first':
            image = image.reshape(1, 1, img_rows, img_cols)
        else:
            image = image.reshape(1, img_rows, img_cols, 1)
        if not using_training_set:
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
        values = self.model.predict(image, verbose=True)
        return values.argmax()

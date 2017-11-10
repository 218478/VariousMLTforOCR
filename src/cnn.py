from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras, cv2
from nn import myNN

class modelCNN(myNN):
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
                input_shape = (1, img_rows, img_cols)
            else:
                input_shape = (img_rows, img_cols, 1)
            print(input_shape)
            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=(3, 3),
                      activation='relu',
                      input_shape=input_shape,
                      kernel_regularizer=keras.regularizers.l2(l=0.001),
                      activity_regularizer=keras.regularizers.l2(l=0.001)))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(classNo, activation='softmax'))
            self.model.summary()
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer = keras.optimizers.Adadelta(),
                               metrics=['accuracy'])

    def fit(self, trainSet, testSet, trainLabels, testLabels, batch_size, epochs):
        self.model.fit(trainSet, trainLabels,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(testSet, testLabels))
        score = self.model.evaluate(testSet, testLabels, verbose=0)
        print(('Test loss:', score[0]))
        print(('Test accuracy:', score[1]))

    def predict(self, image, using_training_set=False):
        """
        image is an opencv image. This function will resize it to the size, when it was trained
        and inverse
        """
        if not using_training_set:
            image = cv2.resize(image, (self.maxsize[0],self.maxsize[1]), interpolation=cv2.INTER_AREA)
            image = cv2.bitwise_not(image)
            image = image.astype('float32')
            image /= 255
        values = self.model.predict(image.reshape(1,self.maxsize[0],self.maxsize[1],1), verbose=False)
        return values.argmax()
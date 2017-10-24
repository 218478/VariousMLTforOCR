from  nn import myNN
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# TODO: neurons do not match
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
                input_shape = (1, img_rows, img_cols)
            else:
                input_shape = (img_rows, img_cols, 1)
            self.model = Sequential()
            self.model.add(Dense(512, activation='relu',
                            input_shape=input_shape))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(classNo, activation='softmax'))

            self.model.summary()

            self.model.compile(loss='categorical_crossentropy',
                        optimizer=RMSprop(),
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
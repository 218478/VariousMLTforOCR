from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras, cv2
from nn import myNN

class modelCNN2(myNN):
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
            # For an explanation on conv layers see http://cs231n.github.io/convolutional-networks/#conv
            # By default the stride/subsample is 1 and there is no zero-padding.
            # If you want zero-padding add a ZeroPadding layer or, if stride is 1 use border_mode="same"
            self.model.add(Conv2D(12, 5, 5, activation = 'relu', input_shape=input_shape, init='he_normal'))

            # For an explanation on pooling layers see http://cs231n.github.io/convolutional-networks/#pool
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(25, 5, 5, activation = 'relu', init='he_normal'))

            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # Flatten the 3D output to 1D tensor for a fully connected layer to accept the input
            self.model.add(Flatten())
            self.model.add(Dense(180, activation = 'relu', init='he_normal'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(100, activation = 'relu', init='he_normal'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(classNo, activation = 'softmax', init='he_normal')) #Last layer with one output per class
            self.model.summary()
            # The function to optimize is the cross entropy between the true label and the output (softmax) of the model
            # We will use adadelta to do the gradient descent see http://cs231n.github.io/neural-networks-3/#ada
            self.model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])

    def fit(self, trainSet, testSet, trainLabels, testLabels, batch_size, epochs):
        self.model.fit(trainSet, trainLabels,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(testSet, testLabels))
        score = self.model.evaluate(testSet, testLabels, verbose=0)
        print(('Test loss:', score[0]))
        print(('Test accuracy:', score[1]))

    def predict(self, image):
        """
        image is an opencv image. This function will resize it to the size, when it was trained
        """
        image = cv2.resize(image, (self.maxsize[0],self.maxsize[1]), interpolation=cv2.INTER_AREA)
        values = self.model.predict(image.reshape(1,self.maxsize[0],self.maxsize[1],1), verbose=False)

        print("wartosci: " + str(values))
        return values.argmax()
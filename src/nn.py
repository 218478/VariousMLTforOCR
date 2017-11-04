import keras, cv2

class myNN:
    def loadKerasModel(self, filepath):
        self.model = keras.models.load_model(filepath)

    def saveKerasModel(self, filepath = "trained_model.h5"):
        self.model.save(filepath)

    def predict(self, image):
        """
        image is an opencv image. This function will resize it to the size, when it was trained

        TODO: how to know the size when image is loaded
        """
        image = cv2.resize(image, (16,16), interpolation=cv2.INTER_AREA)
        image = cv2.bitwise_not(image)
        # cv2.imwrite("converted.jpg", image)
        values = self.model.predict(image.reshape(1,16,16,1), verbose=False)
        return values.argmax()
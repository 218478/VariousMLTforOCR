import keras

class myNN:
    def loadKerasModel(self, filepath):
        self.model = keras.models.load_model(filepath)

    def saveKerasModel(self, filepath = "trained_model.h5"):
        self.model.save(filepath)
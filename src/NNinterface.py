import keras


class NNinterface:
    def load_keras_model(self, filepath):
        self.model = keras.models.load_model(filepath)

    def save_keras_model(self, filepath="trained_model.h5"):
        self.model.save(filepath)

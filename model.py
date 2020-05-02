from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from config import *
from plot import plot_loss, plot_acc


class Net:
    def __init__(self):
        self.model = None
        self.history = None

    def build_net(self):
        self.model = Sequential()
        self.model.add(Dense(50, activation="relu", input_shape=(10000,)))
        self.model.add(Dropout(0.2, noise_shape=None, seed=None))
        self.model.add(Dense(50, activation="linear", kernel_regularizer=regularizers.l2()))
        self.model.add(Dropout(0.5, noise_shape=None, seed=None))
        self.model.add(Dense(100, activation="relu", kernel_regularizer=regularizers.l2()))
        self.model.add(Dropout(0.5, noise_shape=None, seed=None))
        self.model.add(Dense(50, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.summary()

    def compile(self):
        self.model.compile(Adam(lr=ADAM_LR), loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, x_train, y_train, x_test, y_test):
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_test, y_test)
        )

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def demonstration(self):
        H = self.history
        plot_loss(H.history['loss'], H.history['val_loss'])
        plot_acc(H.history['accuracy'], H.history['val_accuracy'])

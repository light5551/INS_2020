from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam

from config import *
from plot import plot_loss, plot_acc


class Net:
    def __init__(self):
        self.model = None
        self.history = None

    def build_net(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=INPUT_SHAPE))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(NUM_CLASSES, activation='softmax'))

    def compile(self):
        self.model.compile(Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

    def fit(self, x_train, y_train):
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_split=VALIDATION_SPLIT
        )

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def demonstration(self):
        H = self.history
        plot_loss(H.history['loss'], H.history['val_loss'])
        plot_acc(H.history['accuracy'], H.history['val_accuracy'])

from keras import Input, Model
from keras.layers import MaxPooling2D, Convolution2D
from keras.layers import Dense, Dropout, Flatten
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

from me.config import *
from me.plot import plot_loss, plot_acc


class Net:
    def __init__(self):
        self.model = None
        self.history = None

    def build_net(self, depth, height, width, num_classes):
        inp = Input(shape=(depth, height, width))  # N.B. depth goes first in Keras
        conv_1 = Convolution2D(CONV_DEPTH_1, KERNEL_SIZE, KERNEL_SIZE, border_mode='same', activation='relu')(inp)
        conv_2 = Convolution2D(CONV_DEPTH_1, KERNEL_SIZE, KERNEL_SIZE, border_mode='same', activation='relu')(conv_1)
        pool_1 = MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE))(conv_2)
        drop_1 = Dropout(DROP_PROB_1)(pool_1)
        conv_3 = Convolution2D(CONV_DEPTH_2, 3, 3, border_mode='same', activation='relu')(drop_1)
        conv_4 = Convolution2D(CONV_DEPTH_2, 3, 3, border_mode='same', activation='relu')(conv_3)
        pool_2 = MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE))(conv_4)
        drop_2 = Dropout(DROP_PROB_1)(pool_2)
        flat = Flatten()(drop_2)
        hidden = Dense(HIDDEN_SIZE, activation='relu')(flat)
        drop_3 = Dropout(DROP_PROB_2)(hidden)
        out = Dense(num_classes, activation='softmax')(drop_3)
        self.model = Model(input=inp, output=out)  # To define a model, just specify its input and output layers

    def compile(self):
        self.model.compile(Adam(lr=0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])

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

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU, LSTM
from keras.optimizers import Adam

from config import *
from plot import plot_loss, plot_sequence


class Net:
    def __init__(self):
        self.model = None
        self.history = None

    def build_net(self):
        self.model = Sequential()
        self.model.add(GRU(32, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
        self.model.add(LSTM(32, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2))
        self.model.add(Dropout(0.5))
        self.model.add(GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
        self.model.add(Dense(1))

    def compile(self):
        self.model.compile(Adam(), loss='mse')

    def fit(self, x_train, y_train, x_val, y_val):
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_val, y_val)
        )

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def demonstration(self, test_data, test_res):
        H = self.history
        plot_loss(H.history['loss'], H.history['val_loss'])
        plot_sequence(self.model.predict(test_data), test_res)


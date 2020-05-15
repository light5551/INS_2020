from keras import Sequential
from keras.datasets import imdb
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, LSTM, Dropout
import numpy
from keras_preprocessing import sequence

from config import *
from plot import plot_loss, plot_acc


class Net:
    def __init__(self):
        self.model = None
        self.model_2 = None
        self.history = None
        self.history_2 = None

    def build_net(self):
        self.model = Sequential()
        self.model.add(Embedding(10000, EMBEDING_VECOR_LENGTH, input_length=MAX_REVIEW_LENGTH))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(100))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model_2 = Sequential()
        self.model_2.add(Embedding(10000, EMBEDING_VECOR_LENGTH, input_length=MAX_REVIEW_LENGTH))
        self.model_2.add(LSTM(100))
        self.model_2.add(Dense(1, activation='sigmoid'))


    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, x_test, y_test):
        epoch = 3
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=epoch,
            verbose=1,
            validation_split=0.1
        )
        self.history_2 = self.model_2.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=epoch,
            verbose=1,
            validation_split=0.1
        )

    def smart_predict(self, input):
        pred1 = self.model.predict(input)
        pred2 = self.model_2.predict(input)
        pred = [1 if (pred1[i] + pred2[i]) / 2 > 0.5 else 0 for i in range(len(pred1))]
        return numpy.array(pred)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def encode_review(self, rev):
        res = []
        for i, el in enumerate(rev):
            el = el.lower()
            delete_el = [',', '!', '.', '?']
            for d_el in delete_el:
                el = el.replace(d_el, '')
            el = el.split()
            for j, word in enumerate(el):
                code = imdb.get_word_index().get(word)
                if code is None:
                    code = 0
                el[j] = code
            res.append(el)
        for i, r in enumerate(res):
            res[i] = sequence.pad_sequences([r], maxlen=MAX_REVIEW_LENGTH)
        res = numpy.array(res)
        return res.reshape((res.shape[0], res.shape[2]))

    def review(self, review):
        data = self.encode_review(review)
        print(self.smart_predict(data))

    def demonstration(self):
        H = self.history
        plot_loss(H.history['loss'], H.history['val_loss'])
        plot_acc(H.history['accuracy'], H.history['val_accuracy'])

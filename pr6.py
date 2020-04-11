from model import Net
from var4 import gen_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from config import *


def generation_data():
    x, y = gen_data(size=DATASET_SIZE, img_size=28)
    x, y = np.asarray(x), np.asarray(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_train = to_categorical(y_train, NUM_CLASSES)

    encoder.fit(y_test)
    y_test = encoder.transform(y_test)
    y_test = to_categorical(y_test, NUM_CLASSES)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = generation_data()
net = Net()
net.build_net()
net.compile()
net.fit(x_train, y_train)
_, acc = net.evaluate(x_train, y_train)
print('Train', acc)
_, acc = net.evaluate(x_test, y_test)
print('Test', acc)
net.demonstration()

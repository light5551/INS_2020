from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
from me.model import Net

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

num_train, depth, height, width = X_train.shape # there are 50000 training examples in CIFAR-10
num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # there are 10 image classes

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_train) # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels


print(num_classes)
net = Net()
net.build_net(depth, height, width, num_classes)
net.compile()
net.fit(X_train, Y_train)

_, acc = net.evaluate(X_test, Y_test)
print('Test', acc)
net.demonstration()

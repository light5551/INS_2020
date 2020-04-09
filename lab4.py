import keras
import matplotlib.pyplot as plt
from keras import Sequential
from keras import optimizers as opt
from keras.layers import Dense
from keras.preprocessing import image
from keras.utils import to_categorical

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)


class Opt:
    def __init__(self, name, opt):
        self.opt = opt
        self.name = name


lr = .001
OPTIMIZERS = [
    Opt("Adam", opt.Adam(lr=lr)),
    Opt("SGD", opt.SGD(lr=lr, momentum=.2)),
    Opt("RMSprop", opt.RMSprop(lr=lr))
]


def upload_image(path):
    img = image.load_img(path=path, grayscale=True, target_size=(28, 28, 1))
    img = image.img_to_array(img)
    return img.reshape((1, 784))


def predict_image(model, img):
    img_class = model.predict_classes(img)
    prediction = img_class[0]
    return prediction


def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


def create_graphic(H, title_name):
    plt.figure(1, figsize=(8, 5))
    plt.title(title_name)
    plt.plot(H.history['accuracy'], 'r', label='train')
    plt.plot(H.history['val_accuracy'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()
    plt.figure(1, figsize=(8, 5))

    plt.title("{} Training and test loss".format(title_name))
    plt.plot(H.history['loss'], 'r', label='train')
    plt.plot(H.history['val_loss'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()


model = build_model()

answer = {}
prediction_model = {}
for optimizer in OPTIMIZERS:
    model.compile(optimizer=optimizer.opt, loss='categorical_crossentropy', metrics=['accuracy'])
    H = model.fit(train_images, train_labels, epochs=7, batch_size=100, validation_split=0.1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    create_graphic(H, optimizer.name)
    answer[optimizer.name] = test_acc
    prediction_model[optimizer.name] = predict_image(model, upload_image('./testimage.jpg'))
    model = build_model()

for i in answer.keys():
    print("{}:  {}".format(i, answer[i]))

for i in prediction_model.keys():
    print("{}:  {}".format(i, prediction_model[i]))


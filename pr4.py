import numpy as np
from keras.layers import Dense
from keras.models import Sequential


def relu(x):
    return np.maximum(x, 0.)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def result_of_operation(a, b, c) -> int:
    return (a or b) and (b or c)


def get_matrix_truth():
    return np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]])


def result_of_matrix():
    return np.array([result_of_operation(*i) for i in get_matrix_truth()])


def tensor_result(dataset, weights):
    result = dataset.copy()
    layers = [relu for i in range(len(weights) - 1)]
    layers.append(sigmoid)
    for i in range(len(weights)):
        result = layers[i](np.dot(result, weights[i][0]) + weights[i][1])
    return result


def each_element_of_tensor_result(dataset, weights):
    result = dataset.copy()
    layers = [relu for i in range(len(weights) - 1)]
    layers.append(sigmoid)
    for weight in range(len(weights)):
        len_current_weight = len(weights[weight][1])
        step_result = np.zeros((len(result), len_current_weight))
        for i in range(len(result)):
            for j in range(len_current_weight):
                sum = 0
                for k in range(len(result[i])):
                    sum += result[i][k] * weights[weight][0][k][j]
                step_result[i][j] = layers[weight](sum + weights[weight][1][j])
        result = step_result
    return result


def smart_print(model, dataset):
    weights = [layer.get_weights() for layer in model.layers]
    tensor_res = tensor_result(dataset, weights)
    each_el = each_element_of_tensor_result(dataset, weights)
    model_res = model.predict(dataset)
    print(tensor_res)
    print(model_res)
    assert np.isclose(tensor_res, model_res).all()
    assert np.isclose(each_el, model_res).all()
    print("Результат тензорного вычисления:")
    print(tensor_res)
    print("Результат поэлементного вычисления:")
    print(each_el)
    print("Результат прогона через обученную модель:")
    print(model_res)
    print('Predict')
    #print(model.predict(np.array([[1,1,1]])))


def custom_print(model):
    weights = [layer.get_weights() for layer in model.layers]
    def a(x):
        print(x[0][0])
        return 1 if x[0][0] > .5 else 0
    datas = [
        np.array([[1, 1, 1]]),
        np.array([[1, 0, 1]]),
        np.array([[0, 0, 1]])
    ]
    for data in datas:
        print('------START ITERATION {}-------'.format(data))
        print('Keras model:', a(model.predict(data)))
        print('Tensor model:', a(tensor_result(data, weights)))
        print('each element:', a(each_element_of_tensor_result(data, weights)))
        print('----END ITERATION------')


def create_model():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(3,)))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_model(model, train, validation):
    return model.fit(train, validation, epochs=150, batch_size=1)  # нет разницы от кол-во эпох, все равно потом берутся ее же слои


def start():
    train_data = get_matrix_truth()
    validation_data = result_of_matrix()
    model = create_model()
    print('NOT fitting')
    smart_print(model, train_data)
    custom_print(model)
    print('fitting')
    fit_model(model, train_data, validation_data)
    smart_print(model, train_data)
    custom_print(model)
    print(get_matrix_truth())
    print(result_of_matrix())


if __name__ == '__main__':
    start()

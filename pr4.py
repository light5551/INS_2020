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
    for w in range(len(weights)):
        step_result = np.zeros((len(result), len(weights[w][1])))
        for i in range(len(result)):
            for j in range(len(weights[w][1])):
                sum = 0
                for k in range(len(result[i])):
                    sum += result[i][k] * weights[w][0][k][j]
                step_result[i][j] = layers[w](sum + weights[w][1][j])
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


def create_model():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(3,)))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_model(model, train, validation):
    return model.fit(train, validation, epochs=1, batch_size=1)  # нет разницы от кол-во эпох, все равно потом берутся
                                                                 # ее же слои


def start():
    train_data = get_matrix_truth()
    validation_data = result_of_matrix()
    model = create_model()
    print('NOT fitting')
    smart_print(model, train_data)
    print('fitting')
    fit_model(model, train_data, validation_data)
    smart_print(model, train_data)


if __name__ == '__main__':
    start()


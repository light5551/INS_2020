from data_manipulation import gen_data_from_sequence
from model import Net


def create_dataset():
    data, res = gen_data_from_sequence()

    dataset_size = len(data)
    train_size = (dataset_size // 10) * 7
    val_size = (dataset_size - train_size) // 2

    train_data, train_res = data[:train_size], res[:train_size]
    val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]
    test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]
    return train_data, train_res, val_data, val_res, test_data, test_res


train_data, train_res, val_data, val_res, test_data, test_res = create_dataset()

net = Net()
net.build_net()
net.compile()
net.fit(train_data, train_res, val_data, val_res)
net.evaluate(test_data, test_res)
net.demonstration(test_data, test_res)

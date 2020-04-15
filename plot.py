import matplotlib.pyplot as plt


def plot_loss(loss, v_loss):
    plt.figure(1, figsize=(8, 5))
    plt.plot(loss, 'b', label='train')
    plt.plot(v_loss, 'r', label='validation')
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    plt.clf()


def plot_sequence(predicted_res, test_res):
    pred_length = range(len(predicted_res))
    plt.title('Sequence')
    plt.ylabel('Sequence')
    plt.xlabel('x')
    plt.plot(pred_length, predicted_res)
    plt.plot(pred_length, test_res)
    plt.show()


def plot_acc(acc, val_acc):
    plt.plot(acc, 'b', label='train')
    plt.plot(val_acc, 'r', label='validation')
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    plt.clf()
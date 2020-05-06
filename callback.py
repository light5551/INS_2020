from keras.callbacks import Callback
from datetime import datetime


class CB(Callback):
    def __init__(self, epoch, prefix="prefix"):
        super(CB, self).__init__()
        self.epoch = epoch
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        now = datetime.now().strftime('%m-%d')
        if epoch in self.epoch:
            self.model.save("{}_{}_{}".format(now, self.prefix, epoch))

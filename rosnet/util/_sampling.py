import numpy as np

def train_test_split(x, y, validation_split, is_shuffle):
    unit = int(x.shape[0] * validation_split)
    shuffle = np.arange(x.shape[0])
    if is_shuffle:
      np.random.shuffle(shuffle)

    x_val = x[shuffle[:unit]]
    y_val = y[shuffle[:unit]]
    x_train = x[shuffle[unit:]]
    y_train = y[shuffle[unit:]]

    return x_train, x_val, y_train, y_val
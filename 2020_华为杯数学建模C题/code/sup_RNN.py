import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers, initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K

from sklearn.utils import class_weight


str_matrix = np.array([["A", "B", "C", "D", "E", "F"], \
                       ["G", "H", "I", "J", "K", "L"], \
                       ["M", "N", "O", "P", "Q", "R"], \
                       ["S", "T", "U", "V", "W", "X"], \
                       ["Y", "Z", "1", "2", "3", "4"], \
                       ["5", "6", "7", "8", "9", "10"]])

test_str = ["M", "F", "5", "2", "I"]


class GetLossesCallback(Callback):
    def __init__(self, epoch_learning_rate_lists=None):
        super(GetLossesCallback, self).__init__()
        self.epoch_learning_rate_lists = epoch_learning_rate_lists
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_learning_rate_lists is not None:
            lr = float(K.get_value(self.model.optimizer.learning_rate))
            for ep_lr in self.epoch_learning_rate_lists:
                if epoch == ep_lr[0]:
                    lr = ep_lr[1]
                    break
            K.set_value(self.model.optimizer.lr, lr)


class SaveBestModelCallback(Callback):
    def __init__(self, min_delta, patience):
        super(SaveBestModelCallback, self).__init__()

        self.min_delta = min_delta
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.Inf
        self.best_epoch = None
        self.best_weights = None
        self.best_flag = False
    
    def on_epoch_end(self, epoch, logs=None):
        if not self.best_flag and epoch > 50:
            current = logs["val_loss"]

            if np.less(current + self.min_delta, self.best):
                self.wait = 0
                self.best = current
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.best_epoch = epoch - self.wait
                    self.best_flag = True


def build_model(input_dim, unit, learning_rate=0.001):
    weight_init = initializers.RandomNormal(0, 0.5)
    bias_init = initializers.Constant(0.1)

    model = Sequential()
    model.add(LSTM(unit, input_shape=input_dim, recurrent_activation="sigmoid", \
            kernel_initializer=weight_init, bias_initializer=bias_init))
    model.add(Dense(1, activation="sigmoid", \
            kernel_initializer=weight_init, bias_initializer=bias_init))
    model.compile(optimizer=optimizers.Adam(learning_rate), \
            loss="binary_crossentropy", metrics=["accuracy"])
    return model


def disassemble_data(data):
    label_index = np.where(pd.isna(data["label"]) == False)[0]
    label = data.iloc[label_index, 0].to_numpy().astype("int64")

    time_step = label_index[1] - label_index[0]
    x = np.zeros((label.shape[0], time_step, data.shape[1]-1))
    for i in range(label.shape[0]):
        x[i] = data.iloc[i*time_step : (i+1)*time_step, 1:].to_numpy()
        x[i] -= np.mean(x[i], axis=0)
        x[i] /= np.std(x[i], axis=0)

    return x, label, time_step


def get_one_person_data(filename):
    data = pd.read_excel(filename, None, index_col=0)
    
    x = []
    label = []
    for i in data.values():
        tmp_x, tmp_label, time_step = disassemble_data(i)
        x.append(tmp_x)
        label.append(tmp_label)
    x = np.vstack(x)
    label = np.hstack(label)

    return x, label, time_step


def mean_predict(predict, test_label):
    letter_index = int(test_label.shape[0] / 10)
    mean = np.zeros(120)
    for i in range(10):
        for j in range(12):
            letter_range = range(i*letter_index, (i+1)*letter_index)
            index = np.where(test_label[letter_range] == j+1)[0]
            mean[i*12+j] = predict[letter_range][index].sum().mean()
    label = np.array([i+1 for i in range(12)] * 10)
    return mean, label


def caculate_label_and_letter(predict, test_label):
    predict_label = np.zeros_like(predict).astype("int32")
    predict_letter = []

    letter_index = int(test_label.shape[0] / 10)
    for i in range(10):
        for j in range(int(letter_index / 12)):
            index = range(i*letter_index + j*12, i*letter_index + (j+1)*12)
            rows_index = np.where(test_label[index] <= 6)[0]
            cols_index = np.where(test_label[index] > 6)[0]

            row_max_index = predict[index][rows_index].argmax()
            col_max_index = predict[index][cols_index].argmax()

            row = test_label[index][rows_index][row_max_index]
            col = test_label[index][cols_index][col_max_index]
            
            label = np.zeros(12)
            for k in [row, col]:
                label_index = np.where(test_label[index] == k)[0]
                label[label_index] = 1
            predict_label[index] = label

            predict_letter.append(str_matrix[row-1, col-7] \
                    + " / " + str(row) + " / " + str(col))

    return predict_label, predict_letter


def save_test_result(predict, test_label, writer, sheet):
    predict_label, predict_letter = \
            caculate_label_and_letter(predict, test_label)
    
    event = np.zeros_like(predict).astype("str")
    predict_event = np.zeros_like(predict).astype("str")
    event[:], predict_event[:] = "", ""

    index = int(test_label.shape[0] / 10)
    for i, letter in enumerate(test_str):
        row, col = np.where(letter == str_matrix)
        event[i*index : (i+1)*index : 12] = [letter + " / " + \
                str(row[0]+1) +  " / " + str(col[0]+7)] * int(index / 12)
    for i, letter in enumerate(predict_letter):
        predict_event[i*12] = letter

    pd.DataFrame({"event": event, "predict event": predict_event, \
            "event label": test_label, "predict": predict, \
            "predict label": predict_label}) \
            .to_excel(writer, sheet_name=sheet)


if __name__ == "__main__":
    train_data_file = "../result/generated_train_data/S1_train_generated.xlsx"
    test_data_file = "../result/generated_test_data/S1_test_generated.xlsx"
    model_save_file = "../model/RNN.h5"

    train_x, train_y, time_step = get_one_person_data(train_data_file)
    test_x, test_label, _ = get_one_person_data(test_data_file)

    index = int(train_y.shape[0] * 8 / 10)
    val_x, val_y = train_x[index:], train_y[index:]
    train_x, train_y = train_x[:index], train_y[:index]


    if test_label.shape[0] == 540:
        test_x = np.vstack((test_x, test_x[-60:]))
        test_label = np.hstack((test_label, test_label[-60:]))

    model = build_model((time_step, 20), 20, 2e-4)
    
    epoch_learning_rate_lists = None#[[50, 1e-5], [100, 5e-6], [200, 1e-6]]
    custom_callbacks = []
    custom_callbacks.append(GetLossesCallback(epoch_learning_rate_lists))
    custom_callbacks.append(SaveBestModelCallback(0, 20))

    weight = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)
    max_epoch = 500
    history = model.fit(train_x, train_y, batch_size=32, epochs=max_epoch, \
            validation_data=(val_x, val_y), callbacks=custom_callbacks, \
            class_weight={0: weight[0], 1: weight[1]})

    writer = pd.ExcelWriter("../result/test_result.xlsx")

    predict = model.predict(test_x).reshape(-1, )
    save_test_result(predict, test_label, writer, "test")
    save_test_result(*mean_predict(predict, test_label), writer, "mean")

    if custom_callbacks[1].best_flag:
        best_epoch = custom_callbacks[1].best_epoch
        model.set_weights(custom_callbacks[1].best_weights)
        model.save(model_save_file)
        predict = model.predict(test_x).reshape(-1, )
        save_test_result(predict, test_label, writer, "best")
        save_test_result(*mean_predict(predict, test_label), writer, "best mean")
        writer.close()
    else:
        best_epoch = max_epoch - custom_callbacks[1].wait - 1

    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.plot(best_epoch, history.history["val_loss"][best_epoch], \
            "x", label="best model", color="black")
    plt.xlabel("epoches")
    plt.ylabel("mse loss")
    plt.legend()
    plt.grid(True)
    plt.show()

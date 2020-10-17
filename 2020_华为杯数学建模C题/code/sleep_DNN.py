import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

from tensorflow.keras import optimizers, initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalCrossentropy


class GetLossesCallback(Callback):
    def __init__(self, test_x, test_y, epoch_learning_rate_lists=None):
        super(GetLossesCallback, self).__init__()

        self.train_loss = []
        self.validation_loss = []
        self.test_loss = []
        self.test_x = test_x
        self.test_y = test_y
        self.epoch_learning_rate_lists = epoch_learning_rate_lists
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_learning_rate_lists is not None:
            lr = float(K.get_value(self.model.optimizer.learning_rate))
            for ep_lr in self.epoch_learning_rate_lists:
                if epoch == ep_lr[0]:
                    lr = ep_lr[1]
                    break
            K.set_value(self.model.optimizer.lr, lr)
    
    def on_epoch_end(self, epoch, logs=None):
        test_predict = self.model.predict_on_batch(self.test_x)
        cce = CategoricalCrossentropy()
        test_loss = cce(self.test_y, test_predict)
        
        self.train_loss.append(logs["loss"])
        self.validation_loss.append(logs["val_loss"])
        self.test_loss.append(test_loss)


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
    weight_init = initializers.RandomNormal(0, 0.3)
    bias_init = initializers.Constant(0.1)

    model = Sequential()
    model.add(Dense(unit, input_dim=input_dim, activation="relu", \
            kernel_initializer=weight_init, bias_initializer=bias_init))
    model.add(Dense(unit, activation="relu", kernel_initializer=weight_init, bias_initializer=bias_init))
    model.add(Dense(5, activation="sigmoid", \
            kernel_initializer=weight_init, bias_initializer=bias_init))
    model.compile(optimizer=optimizers.RMSprop(learning_rate), \
            loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def split_data_set(data_split, data_set):
    train = []
    validation = []
    test = []
    for data in data_set.values():
        train_end_index = int(data.shape[0] * data_split[0])
        validation_end_index = train_end_index + int(data.shape[0] * data_split[1])

        data = data.iloc[:, :5].to_numpy()
        np.random.shuffle(data)

        train.append(data[:train_end_index, :])
        validation.append(data[train_end_index : validation_end_index, :])
        test.append(data[validation_end_index:, :])

    return np.vstack(train), np.vstack(validation), np.vstack(test)


def generate_data(read=True, norm=True):
    data_file = "../data/附件2-睡眠脑电数据.xlsx"
    save_data_file = "../result/sleep_data/data_split.xlsx"

    if read and os.path.exists(save_data_file):
        data_set = pd.read_excel(save_data_file, None)
        train = data_set["train"].to_numpy()
        validation = data_set["validation"].to_numpy()
        test = data_set["test"].to_numpy()
    else:
        data_set = pd.read_excel(data_file, None)

        data_split = [0.6, 0.2, 0.2]
        train, validation, test = split_data_set(data_split, data_set)

        feature_list = ["Alpha", "Beta", "Theta", "Delta"]
        writer = pd.ExcelWriter(save_data_file)
        pd.DataFrame(train, columns=["train label"]+feature_list). \
                to_excel(writer, sheet_name="train")
        pd.DataFrame(validation, columns=["validation label"]+feature_list). \
                to_excel(writer, sheet_name="validation")
        pd.DataFrame(test, columns=["test label"]+feature_list). \
                to_excel(writer, sheet_name="test")
        writer.close()

    if norm:
        train[:, 1:] -= np.mean(train[:, 1:], axis=0)
        train[:, 1:] /= np.std(train[:, 1:], axis=0)
        validation[:, 1:] -= np.mean(validation[:, 1:], axis=0)
        validation[:, 1:] /= np.std(validation[:, 1:], axis=0)
        test[:, 1:] -= np.mean(test[:, 1:], axis=0)
        test[:, 1:] /= np.std(test[:, 1:], axis=0)

    return train, validation, test


def label2y(label):
    label = (label - label.min()).reshape(-1, ).astype("int64")
    return np.eye(label.shape[0])[label, :label.max()+1]


def plot_class_result(y_true, y_pred):
    #font = FontProperties(fname=r"/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc")
    #label = ["深睡眠期", "深睡眠2期", "深睡眠1期", "快速动眼期", "清醒期"]
    true_num = [0] * int(y_true.max() - y_true.min() + 1)
    pred_num = [0] * int(y_true.max() - y_true.min() + 1)
    for i, y in enumerate(y_true):
        true_num[int(y - y_true.min())] += 1
        if y == y_pred[i]:
            pred_num[int(y - y_true.min())] += 1

    label = [i for i in range(2, 7)]
    for i in range(len(pred_num)):
        accuracy = pred_num[i] / true_num[i]
        label[i] = str(label[i]) + " (" + "{:.3f}".format(accuracy) + ")"
        plt.bar(i, accuracy)
    plt.xticks(range(5), label)
    plt.ylabel("accuracy")
    plt.xlabel("labels")
    plt.title("every label accuracy result")
    plt.grid(True)
    plt.show()


def plot_losses(losses, best_epoch):
    x = range(len(losses.train_loss))
    plt.plot(x, losses.train_loss, label="train loss")
    plt.plot(x, losses.validation_loss, label="validation loss")
    plt.plot(x, losses.test_loss, label="test loss")
    plt.plot(best_epoch, losses.validation_loss[best_epoch], \
            "x", label="best epoch", color="black")
    plt.xlabel("epoches")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    model_dir = "../model/"
    sleep_result_dir = "../result/sleep_data/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(sleep_result_dir):
        os.mkdir(sleep_result_dir)

    model = build_model(4, 300, 5e-5)
    
    train, validation, test = generate_data(read=False)
    
    epoch_learning_rate_lists = None#[[50, 1e-5], [100, 5e-6], [200, 1e-6]]
    custom_callbacks = []
    custom_callbacks.append(GetLossesCallback(test[:, 1:], \
            label2y(test[:, 0]), epoch_learning_rate_lists))
    custom_callbacks.append(SaveBestModelCallback(0, 30))

    max_epoch = 500
    model.fit(train[:, 1:], label2y(train[:, 0]), batch_size=32, epochs=max_epoch, \
            validation_data=(validation[:, 1:], label2y(validation[:, 0])), \
            callbacks=custom_callbacks)
    print("model test predict")
    model.evaluate(test[:, 1:], label2y(test[:, 0]))
    predict = model.predict(test[:, 1:]).argmax(axis=1).astype("int32")
    plot_class_result(test[:, 0].astype("int32"), predict+2)
    model.save(model_dir + "sleep_DNN.h5")

    if custom_callbacks[1].best_flag:
        best_epoch = custom_callbacks[1].best_epoch
        model.set_weights(custom_callbacks[1].best_weights)
        print("model test predict at best epoch")
        model.evaluate(test[:, 1:], label2y(test[:, 0]))
        best_predict = model.predict(test[:, 1:]).argmax(axis=1)
        plot_class_result(test[:, 0], best_predict+2)
        model.save(model_dir + "best_sleep_DNN.h5")
    else:
        best_epoch = max_epoch - custom_callbacks[1].wait - 1

    plot_losses(custom_callbacks[0], best_epoch)

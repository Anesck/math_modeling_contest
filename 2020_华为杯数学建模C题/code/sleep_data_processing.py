import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal


def filtering():
    pass


def data_analysis(method):
    plt.rcParams["figure.figsize"] = [13.66, 7.2]
    plt.rcParams["figure.subplot.bottom"] = 0.05
    plt.rcParams["figure.subplot.left"] = 0.03
    plt.rcParams["figure.subplot.right"] = 0.97
    plt.rcParams["figure.subplot.top"] = 0.95

    feature_data = pd.read_excel("../data/附件2-睡眠脑电数据.xlsx", None)

    if method == "feature":
        for sheet, data in feature_data.items():
            x = range(data.shape[0]-1)
            for i in range(1, 5):
                #data.iloc[:, :] -= np.mean(data.iloc[:, :])
                #data.iloc[:, :] /= np.std(data.iloc[:, :])
                plt.scatter(x, np.diff(data.iloc[:, i]), label=data.columns[i])
            plt.title(sheet)
            plt.legend()
            plt.show()
    elif method == "label":
        for i in range(1, 5):
            for sheet, data in feature_data.items():
                x = range(data.shape[0])
                plt.scatter(x, data.iloc[:, i], label=sheet)
            plt.title(data.columns[i])
            plt.legend()
            plt.show()


if __name__ == "__main__":
    data_analysis("label")

import numpy as np
import pandas as pd

from tensorflow.keras import optimizers, initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_model(input_dim, unit, learning_rate=0.001):
    weight_init = initializers.RandomNormal(0, 0.3)
    bias_init = initializers.Constant(0.1)

    model = Sequential()
    model.add(Dense(unit, input_dim=input_dim, activation="sigmoid", \
            kernel_initializer=weight_init, bias_initializer=bias_init))
    model.add(Dense(2, activation="sigmoid", \
            kernel_initializer=weight_init, bias_initializer=bias_init))
    model.compile(optimizer=optimizers.Adam(learning_rate), \
            loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def get_data():
    train_data_file = "../result/train_data/z-score_processed_train.xlsx"
    test_data_file = "../result/test_data/z-score_processed_test.xlsx"

    train_data = pd.read_excel(train_data_file).drop(columns="Unnamed: 0")
    test_data = pd.read_excel(test_data_file).drop(columns="Unnamed: 0")

    train_x = train_data.iloc[:, 1:].to_numpy()
    train_y = train_data.iloc[:, 0].to_numpy()
    test_x = test_data.iloc[:, 1:].to_numpy()

    train_y = np.eye(train_y.shape[0])[train_y, :train_y.max()+1] 
    return train_x, train_y, test_x


if __name__ == "__main__":
    model = build_model(20, 500)
    
    train_x, train_y, test_x = get_data()

    model.fit(train_x, train_y, batch_size=36, epochs=1000)

    test_predict = model.predict(test_x).reshape(-1, )

    pd.Series(test_predict).to_excel("../result/test_predict.xlsx")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal


data_filenum = 5
data_dir = "../data/附件1-P300脑机接口数据/"
result_dir = "../result/"

str_matrix = np.array([["A", "B", "C", "D", "E", "F"], \
                       ["G", "H", "I", "J", "K", "L"], \
                       ["M", "N", "O", "P", "Q", "R"], \
                       ["S", "T", "U", "V", "W", "X"], \
                       ["Y", "Z", "1", "2", "3", "4"], \
                       ["5", "6", "7", "8", "9", "10"]])
num_matrix = np.arange(101, 137).reshape(str_matrix.shape)

columns_name = ["Fz", "F3", "F4", "Cz", "C3", "C4", "T7", "T8", "CP3", "CP4", \
                "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8", "Oz", "O1", "O2"]

plt.rcParams["figure.figsize"] = [19.20, 10.80]
plt.rcParams["figure.subplot.bottom"] = 0.05
plt.rcParams["figure.subplot.left"] = 0.07
plt.rcParams["figure.subplot.right"] = 0.97
plt.rcParams["figure.subplot.top"] = 0.95


def name_read(prename, dtype, person):
    if prename == "raw":
        return data_dir + "/S" + str(person) + "/S" \
                + str(person) + dtype + "data.xlsx"
    elif prename == "label":
        return data_dir + "/S" + str(person) + "/S" \
                + str(person) + dtype + "event.xlsx"
    else:
        return result_dir + prename + dtype + "data/S" \
                + str(person) + dtype + prename + ".xlsx"


def name_write(prename, dtype, person):
    save_path = result_dir + prename + dtype + "data/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    return save_path + "/S" + str(person) + dtype + prename + ".xlsx"


def data_filtering():
    b, a = signal.butter(6, [2/250, 20/250], btype="bandpass")
    for dtype in ["_train_", "_test_"]:
        for s in range(data_filenum):
            feature_data = pd.read_excel(name_read("raw", dtype, s+1), \
                    None, header=None, names=columns_name)
            
            writer = pd.ExcelWriter(name_write("filtered", dtype, s+1))
            for sheet, data in feature_data.items():
                for i in range(data.shape[1]):
                    data.iloc[:, i] = signal.filtfilt(b, a, data.iloc[:, i])
                data.to_excel(writer, sheet_name=sheet)
            writer.close()


def data_select():
    for dtype in ["_train_", "_test_"]:
        for s in range(data_filenum):
            feature_data = pd.read_excel(name_read("filtered", \
                    dtype, s+1), None, index_col=0)
            event_data = pd.read_excel(name_read("label", \
                    dtype, s+1), None, header=None)
            
            writer = pd.ExcelWriter(name_write("selected", dtype, s+1))
            for sheet, data in feature_data.items():
                if sheet in event_data.keys():
                    data.iloc[event_data[sheet].iloc[0, 1] \
                            : event_data[sheet].iloc[-1, 1]+1]. \
                            to_excel(writer, sheet_name=sheet)
            writer.close()


def data_weighted_mean(block=5):
    for dtype in ["_train_", "_test_"]:
        for s in range(data_filenum):
            feature_data = pd.read_excel(name_read("selected", dtype, s+1), \
                    None, index_col=0, header=None, names=columns_name)
            event_data = pd.read_excel(name_read("label", \
                    dtype, s+1), None, header=None)

            feature_data_all.append(feature_data)
            for sheet, data in feature_data.items():
                if sheet in event_data.keys():
                    block_split_index = event_data[sheet].iloc[::13, 1]
                    for i in range(block):
                        std.append(1 / np.var(data.loc[ \
                                event_data[sheet].iloc[::13, 1].iloc[i] : \
                                event_data[sheet].iloc[::13, 1].iloc[i]+1]))


def dataset_generate(section=np.arange(70, 150)):
    label_step = section[-1] - section[0] + 1
    for dtype in ["_train_", "_test_"]:
        for s in range(data_filenum):
            feature_data = pd.read_excel(name_read("filtered", \
                    dtype, s+1), None, index_col=0)
            event_data = pd.read_excel(name_read("label", \
                    dtype, s+1), None, header=None)
            
            writer = pd.ExcelWriter(name_write("generated", dtype, s+1))
            for sheet, data in feature_data.items():
                if sheet in event_data.keys():
                    event = event_data[sheet]
                    if dtype == "_train_":
                        row, col = np.where(num_matrix == event.iloc[0, 0])
                    
                    event = event.drop(index=range(event.shape[0])[::13])
                    labeled_data = pd.DataFrame([], columns=["label"] + columns_name)
                    for i in range(event.shape[0]):
                        labeled_data = labeled_data.append( \
                                data.loc[event.iloc[i, 1] + section])
                        if dtype == "_train_":
                            if event.iloc[i, 0] == row+1 or \
                                    event.iloc[i, 0] == col+7:
                                labeled_data.iloc[i*label_step, 0] = 1
                            else:
                                labeled_data.iloc[i*label_step, 0] = 0
                        else:
                            labeled_data.iloc[i*label_step, 0] = event.iloc[i, 0]
                    labeled_data.to_excel(writer, sheet_name=sheet)
            writer.close()


def generated_mean_data(method, block_num=5, section=np.arange(70, 150)):
    label_step = section[-1] - section[0] + 1
    for dtype in ["_train_", "_test_"]:
        for s in range(data_filenum):
            feature_data = pd.read_excel(name_read("filtered", \
                    dtype, s+1), None, index_col=0)
            event_data = pd.read_excel(name_read("label", \
                    dtype, s+1), None, header=None)

            x_letters = []
            writer = pd.ExcelWriter(name_write("generated_mean", dtype, s+1))
            for sheet, data in feature_data.items():
                if sheet in event_data.keys():
                    event = event_data[sheet]
                    if dtype == "_train_":
                        row, col = np.where(num_matrix == event.iloc[0, 0])
                    
                    event = event.drop(index=range(event.shape[0])[::13])
                    labeled_data = pd.DataFrame([], columns=["label"] + columns_name)
                    for i in range(12):
                        label_index = np.where(event == i+1)[0][:block_num]
                        labeled_data_index = event.iloc[label_index, 1]

                        tmp_labeled_data = np.zeros((label_step, 20))
                        if method == "best":
                            sum_data_var = np.zeros(20)
                            for k in labeled_data_index:
                                tmp_data = data.loc[k + section].to_numpy()
                                tmp_data_var = np.var(tmp_data, axis=0)
                                tmp_labeled_data += tmp_data / tmp_data_var
                                sum_data_var += 1 / tmp_data_var
                            tmp_labeled_data /= sum_data_var
                        elif method == "normal":
                            for k in labeled_data_index:
                                tmp_labeled_data += data.loc[k + section].to_numpy()
                            tmp_labeled_data /= block_num

                        labeled_data = labeled_data.append(pd.DataFrame( \
                                tmp_labeled_data, columns=columns_name))
                        if dtype == "_train_":
                            if i == row+1 or i == col+7:
                                labeled_data.iloc[i*label_step, 0] = 1
                            else:
                                labeled_data.iloc[i*label_step, 0] = 0
                        else:
                            labeled_data.iloc[i*label_step, 0] = i + 1
                    labeled_data.to_excel(writer, sheet_name=sheet)
            writer.close()
                

def generated_data_analysis(mean):
    for dtype in ["_train_", "_test_"]:
        for s in range(data_filenum):
            feature_data = pd.read_excel(name_read("generated" + mean, \
                    dtype, s+1), None, index_col=0)

            #x_letters = []
            for sheet, data in feature_data.items():
                label_index = np.where(pd.isna(data["label"]) == False)[0]
                label = data.iloc[label_index, 0].to_numpy().astype("int64")
                time_step = label_index[1] - label_index[0]

                x = np.zeros((label.shape[0], time_step, data.shape[1]-1))
                for i in range(label.shape[0]):
                    x[i] = data.iloc[i*time_step : (i+1)*time_step, 1:].to_numpy()
                    x[i] -= np.mean(x[i], axis=0)
                    x[i] /= np.std(x[i], axis=0)
                #x_letters.append(x)

                for i in range(x.shape[2]):
                    j = 0
                    ax = plt.subplot(10, 2, i+1)
                    for k in range(x.shape[0])[j*12 : (j+1)*12]:
                        if label[k] == 1:
                            plt.plot(x[k, :, i], color="red")
                        else:
                            plt.plot(x[k, :, i], "--", color="black")
                    plt.xlim([0, 80])
                    plt.ylabel(columns_name[i])
                    plt.grid(True)
                figname = "S" + str(s+1) + ": " + sheet
                plt.suptitle(figname)
                #plt.show()

                plt.savefig("../picture/" + figname + ".png")
                plt.close()


def data_analysis():
    b, a = signal.butter(6, [2/250, 20/250], btype="bandpass")
    dtype = "_train_"
    for s in range(data_filenum):
        feature_data_label = pd.read_excel(name_read("label", \
                    dtype, s+1), None, header=None)
        feature_data = pd.read_excel(name_read("raw", dtype, s+1), \
                None, header=None, names=columns_name)

        for sheet, data in feature_data_label.items():
            if sheet in feature_data.keys():
                index = np.where(num_matrix == data.iloc[0, 0])
                data = data.drop(index=range(data.shape[0])[::13])
                feature_data[sheet] -= np.mean(feature_data[sheet])
                feature_data[sheet] /= np.std(feature_data[sheet])

                for i in range(feature_data[sheet].shape[1]):
                    ax = plt.subplot(10, 2, i+1)
                    plt.plot(range(feature_data[sheet].shape[0]), \
                            feature_data[sheet].iloc[:, i], label="norm raw")
                    filtered_data = signal.filtfilt(b, a, feature_data[sheet].iloc[:, i])
                    plt.plot(range(feature_data[sheet].shape[0]), \
                            filtered_data, label="filtered")
                    filtered_data -= np.mean(filtered_data)
                    filtered_data /= np.std(filtered_data)
                    plt.plot(range(feature_data[sheet].shape[0]), \
                            filtered_data, label="norm filtered")

                    index_label = []
                    y_max = feature_data[sheet].iloc[:, i].max()
                    y_min = feature_data[sheet].iloc[:, i].min()
                    for j in range(data.shape[0]):
                        if data.iloc[j, 0] == index[0][0]+1 or data.iloc[j, 0] == index[1][0]+1:
                            color = "red"
                        else:
                            color = "black"
                        x = data.iloc[j, 1]
                        plt.plot([x, x], [y_min, y_max], linestyle="--", color=color)
                        #index_label.append(str(data.iloc[j, 1]) + " (" + str(data.iloc[i, 0]) + ")")
                    #plt.xticks(ticks=data.iloc[:, 1], labels=index_label)
                    plt.xlim([0, 3150])
                    plt.ylabel(columns_name[i])
                    plt.grid(True)
                plt.legend()
                figname = "S" + str(s+1) + ": " + sheet
                plt.suptitle(figname)
                #plt.show()

                plt.savefig("../picture/" + figname + ".png")
                plt.close()


if __name__ == "__main__":
    #data_filtering()

    #data_select()
    
    #dataset_generate()

    #generated_mean_data("normal")

    #generated_mean_data("best")

    #generated_data_analysis("")
    
    #generated_data_analysis("_mean")

    #data_analysis()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_img(x, y, num, img, str_title):
    plt.subplot(x, y, num)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title(str_title)

if __name__ == "__main__":
    # 读取图片并画出来
    img = plt.imread("../imgs/PCA_imgs/one_peice.jpeg")
    plot_img(1, 2, 1, img, "before")

    # 将图片数据转化为二维
    img = img.reshape(img.shape[0], -1)
    print(img.shape)

    # 对转化后的数据进行 PCA 降维
    pca = PCA(n_components=10).fit(img)
    #pca = PCA(n_components=100).fit(img)
    print("保留的信息量百分比：{} %".format(pca.explained_variance_ratio_.sum()*100))
    img = pca.transform(img)
    print(img.shape)

    # 将降维后的数据还原至原空间
    img = pca.inverse_transform(img)

    # 将还原的数据处理为 RGB 图片数据 0~255 的整数格式，并画出
    img = img.reshape(450, 450, 3)
    img = img.astype(int)
    img = np.clip(img, 0, 255)
    plot_img(1, 2, 2, img, "after")

    plt.show()

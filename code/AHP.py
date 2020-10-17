import numpy as np


RI = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]


def hierarchical_single_sort_and_check(judgment_matrix):
    lam, v = np.linalg.eig(judgment_matrix)

    n = judgment_matrix.shape[0]
    CI = (lam.max() - n) / (n - 1)
    CR = CI / RI[n-1]

    if CR >= 0.1:
        print("判断矩阵\n{}\n的一致性检验未通过！".format(judgment_matrix))
        exit()

    max_index = lam.argmax()
    weight = v[:, max_index] / v[:, max_index].sum()
    return weight, CI


if __name__ == "__main__":
    A = np.array([[1,   1/2, 4,   3,   3  ], \
                  [2,   1,   7,   5,   5  ], \
                  [1/4, 1/7, 1,   1/2, 1/3], \
                  [1/3, 1/5, 2,   1,   1  ], \
                  [1/3, 1/5, 3,   1,   1  ]])

    B1 = np.array([[1,   2,   5  ], \
                   [1/2, 1,   2  ], \
                   [1/5, 1/2, 1  ]])

    B2 = np.array([[1,   1/3, 1/8], \
                   [3,   1,   1/3], \
                   [8,   3,   1  ]])

    B3 = np.array([[1,   1,   3  ], \
                   [1,   1,   3  ], \
                   [1/3, 1/3, 1  ]])

    B4 = np.array([[1,   3,   4  ], \
                   [1/3, 1,   1  ], \
                   [1/4, 1,   1  ]])

    B5 = np.array([[1,   1,   1/4], \
                   [1,   1,   1/4], \
                   [4,   4,   1  ]])

    judgment_matrixes = [A, B1, B2, B3, B4, B5]
    weights = []
    CIs = []
    for matrix in judgment_matrixes:
        weight, CI = hierarchical_single_sort_and_check(matrix)
        weights.append(np.array(weight))
        CIs.append(CI)

    # 层次总排序及一致性检验
    CR_B = np.matmul(weights[0], CIs[1:]) / \
            (weights[0] * RI[weights[1].shape[0]-1]).sum()
    if CR_B >= 0.1:
        print("层次总排序一致性检验未通过！")
    else:
        print("最终权重结果为：{}".format(np.matmul(\
            weights[0], np.vstack((weights[1:]))).real))

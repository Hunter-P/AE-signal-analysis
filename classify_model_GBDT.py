# -*- coding:utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
import heapq
import xlwt
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from feature_construction import FeatureConstruction

file = r"G:\声发射试验\节段胶拼轴拉AE试验\节段胶拼轴拉试验201801-王少帅-AE数据\试验4#数据处理\能量大于40_已分类.xls"

pd.set_option('display.line_width', 300)


def data_processing(file, sheet=0, l1=500, l2=890, l3=1180):
    """
    数据准备，能量与信号强度的相关性为1，去掉信号强度这个特征
    :param file:
    :return:
    """
    data = xlrd.open_workbook(file)
    table = data.sheet_by_index(sheet)  # 获取表
    nrows = table.nrows  # 获取行数
    data = []
    for i in range(1, nrows):
        if i <= l1:
            data.append(table.row_values(i)[3:15] + [table.row_values(i)[15]] + [0])
        elif i <= l2:
            data.append(table.row_values(i)[3:15] + [table.row_values(i)[15]] + [1])
        elif i <= l3:
            data.append(table.row_values(i)[3:15] + [table.row_values(i)[15]] + [1])
        else:
            data.append(table.row_values(i)[3:15] + [table.row_values(i)[15]] + [2])
    return pd.DataFrame(data=data, columns=table.row_values(0)[3:15]+[table.row_values(0)[15]]+['label'])

data0 = data_processing(file, 0)
data1 = data_processing(file, 1, 337, 544, 703)
data2 = data_processing(file, 2, 357, 578, 702)
data3 = data_processing(file, 3, 408, 693, 927)


def select_test_sets(scale_data_X, data_Y, start, num_samples, gap):
    """
    从总样本集中抽取一部分样本作为测试集， n个信号组成一个测试  样本
    :param scale_scale_X: np.ndarray
    :param data_Y:  pd.series
    :param start:  切片起始点, 0
    :param num_samples: 信号个数, 9
    :param gap: 间隔, 30
    :return:
    """
    num_sum = scale_data_X.shape[0] - num_samples
    delete_rows = []  # 删除的行索引
    test_scale_X = scale_data_X[start:start + num_samples, :]
    test_Y = data_Y[start:start + num_samples]

    for i in range(start + gap, num_sum, gap):
        test_scale_X = np.concatenate((test_scale_X, scale_data_X[i:i + num_samples, :]), axis=0)
        test_Y = pd.concat((test_Y, data_Y[i:i + num_samples]), axis=0)
        for j in range(i, i + num_samples):
            delete_rows.append(j)

    test_scale_X = np.array(test_scale_X)
    test_Y = np.array(test_Y)
    train_scale_X = np.delete(scale_data_X, delete_rows, axis=0)
    train_Y = np.delete(np.array(data_Y), delete_rows, axis=0)
    return train_scale_X, train_Y, test_scale_X, test_Y

print(data0.columns)
print([i for i in data0.columns[:13]]+[i for i in data0.columns[14:]])
for distance in range(11):
    print("采样信号离端部距离：", distance)
    data0_X = data0[[i for i in data0.columns[:13]]+[i for i in data0.columns[14:]]]
    data0_Y = data0['label']
    scale_data0_X = StandardScaler().fit_transform(data0_X)
    data1_X = data1[[i for i in data1.columns[:13]]+[i for i in data1.columns[14:]]]
    data1_Y = data1['label']
    scale_data1_X = StandardScaler().fit_transform(data1_X)
    data2_X = data2[[i for i in data2.columns[:13]]+[i for i in data2.columns[14:]]]
    data2_Y = data2['label']
    scale_data2_X = StandardScaler().fit_transform(data2_X)
    data3_X = data3[[i for i in data3.columns[:13]]+[i for i in data3.columns[14:]]]
    data3_Y = data3['label']
    scale_data3_X = StandardScaler().fit_transform(data3_X)

    # 整合四个通道的数据
    scale_data_X = np.concatenate((scale_data0_X, scale_data1_X, scale_data2_X, scale_data3_X), axis=0)
    data_Y = pd.concat((data0_Y, data1_Y, data2_Y, data3_Y), axis=0)

    # 产生测试集， 训练集
    num_samples = 8  # 信号个数, 8
    gap = 30
    train_scale_X, train_Y, test_scale_X, test_Y = select_test_sets(scale_data1_X, data1_Y, distance, num_samples=num_samples, gap=gap)
    print('num_samples：', num_samples)
    print('gap:', gap)
    # print(scale_data0_X.shape)
    # print(test_scale_X.shape)
    # print(train_scale_X.shape)
    # 随机化数据集
    train_scale_X, validation_scale_X, train_Y, validation_Y = train_test_split(train_scale_X, train_Y,
                                                                                test_size=0.0, random_state=6)
    num_folds = 10  # 10折交叉验证
    seed = 7
    scoring = 'accuracy'
    best_param = 300
    GBC = GradientBoostingClassifier(n_estimators=best_param, max_depth=3, random_state=1, learning_rate=0.1)
    GBC.fit(train_scale_X, train_Y)
    predictions = GBC.predict(test_scale_X)

    test_y = []
    for i in range(0, len(test_Y), num_samples):
        test_y.append(np.argmax(np.bincount(test_Y[i:i+num_samples])))

    predict_y = []
    for i in range(0, len(predictions), num_samples):
        predict_y.append(np.argmax(np.bincount(predictions[i:i+num_samples])))

    def create_predict_y(predictions, num_samples):
        """
        为处理样本不均衡问题，给初步预测结果进行加权
        """
        weighted_predict_y = []
        for i in range(0, len(predictions), num_samples):
            c_0, c_1, c_2 = 0, 0, 0
            for j in predictions[i:i+num_samples]:
                if j == 0:
                    c_0 += 1
                elif j == 1:
                    c_1 += 1
                else:
                    c_2 += 1
            c = [c_0*0.4, c_1*0.4, c_2*0.4]
            weighted_predict_y.append(c.index(max(c)))
        return weighted_predict_y

    # predict_y = create_predict_y(predictions, num_samples)

    # print("   test_y:", test_y)
    # print("predict_y:", predict_y)
    print("准确率：", accuracy_score(test_y, predict_y))
    print("总数：", len(test_y))
    cout_1 = 0
    cout_2 = 0

    for i in test_y:
        if i == 1:
            cout_1 += 1
        if i == 2:
            cout_2 += 1
    print("0类数：", len(test_y)-cout_1-cout_2)
    print("1类数：", cout_1)
    print("2类数：", cout_2)


# 比较重要的特征
# imp_f = heapq.nlargest(10, GBC.feature_importances_)
# for i in imp_f:
#     index_f = list(GBC.feature_importances_)
#     if index_f.index(i)<12:
#         print(index_f.index(i), i, data0.columns[index_f.index(i)])
#     else:
#         print(index_f.index(i), i, data0.columns[index_f.index(i)+1])






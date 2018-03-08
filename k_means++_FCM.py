# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
import xlrd
import xlwt
import math


# 获取文件某个通道的数据集，并输出pandas.dataframe数据结构
def get_dataset(file, tongdao_num=0):
    """
    获取文件某个通道的数据集，并输出pandas.dataframe数据结构
    :param file:
    :param tongdao_num:0,1,2,3
    :return: data_frame
    """
    data = xlrd.open_workbook(file)
    table = data.sheet_by_index(tongdao_num)
    head = table.row_values(0)
    data_list = []
    nrows = table.nrows
    for i in range(1, nrows):
        if type(table.row_values(i)[2]) == float:
            data_list.append(table.row_values(i))

    data_frame = pd.DataFrame(data_list, columns=head)
    return data_frame


# 数据预处理，归一化，输出numpy.array数据结构
def data_processing(data):
    # 归一化，均值0，方差1
    data_scaled = preprocessing.scale(data.iloc[:, 3:])   # 前3列剔除
    print("归一化后：", data_scaled)
    return data_scaled


# k-means,k-means++,FCM算法
class K_Means(object):

    # FLOAT_MAX = 1e10  # 设置一个较大的值作为初始化的最小的距离

    def __init__(self, dataset, cluster_center_number):
        self.dataset = dataset
        self.cluster_center_number = cluster_center_number

    # 随机化生成[-1,1]之间的中心点
    def get_random_center_point(self):
        center_point = np.array([random.uniform(-1, 1) for i in range(self.dataset.shape[1])])
        return center_point

    # 随机选择一个样本点作为初始中心
    def random_point_as_center_point(self):
        n = self.dataset.shape[0]
        return self.dataset[random.randint(0, n), ]

    # 计算一个样本点与已经初始化的各个聚类中心之间的距离，并选择其中最短的距离输出
    def nearest_distance(self, point, center_points):
        FLOAT_MAX = 1e10  # 设置一个较大的值作为初始化的最小的距离
        min_dist = FLOAT_MAX
        m = center_points.shape[0]
        for i in range(m):
            d = self.distance(center_points[i, ], point)
            if min_dist > d:
                min_dist = d
        return min_dist

    def get_initial_center_points(self):
        """
        生成所有初始聚类中心点，
        :return: 输出中心点矩阵，np.array
        """
        center_points = self.random_point_as_center_point()  # 获取第一个初始化中心点
        for i in range(self.cluster_center_number):
            nearset_distances = []
            for sample in self.dataset:
                min_dist = self.nearest_distance(sample, center_points)
                nearset_distances.append(min_dist)
            index = nearset_distances.index(max(nearset_distances))
            center_point = self.dataset[index, ]
            center_points = np.vstack((center_points, center_point))

        return center_points

    def distance(self, point_A, point_B):
        """
        # 计算两点之间的距离
        :param point_A:
        :param point_B:
        :return:
        """
        return sum((point_A-point_B)*(point_A-point_B).T)

    def k_means_plus_plus(self):
        """
        k-means++算法
        :return: 样本分类列表point_classify，list
        """
        n = self.dataset.shape[0]  # 样本的个数
        center_points = self.get_initial_center_points()  # 初始化中心点
        point_classify = [0]*n  # 样本点的分类
        change = True  # 判断是否需要重新计算聚类中心
        while change is True:
            for i in range(n):  # 对样本点进行分类
                distance_list = [self.distance(point, self.dataset[i, ]) for point in center_points]
                point_classify[i] = distance_list.index(min(distance_list))

            C = []
            for i in range(self.cluster_center_number):
                c_list = [point for point, classify in zip(self.dataset, point_classify) if classify == i]
                C.append(sum(c_list)/len(c_list))
            C = np.array(C)

            if (C == center_points).all() is not True:
                # 更新中心点
                center_points = C
            else:
                change = False

        return point_classify, center_points

    def FCM(self, m, epsilon=0.000000001):
        """
        FCM算法，m的最佳取值范围为[1.5，2.5]
        :param m:m 权值
        :param epsilon:终止条件
        :return: 聚类中心矩阵C，np.array
        """
        U = self.initialise_U()  # 隶属度矩阵
        # print("U: ", U)
        J = 0  # 目标函数
        change = True  # 判断是否需要重新计算聚类中心
        # C = 0
        # 计算聚类中心c
        while change is True:
            C = []  # C为聚类中心矩阵，np.array
            for i in range(self.cluster_center_number):
                c = sum([U[j, i]**m*self.dataset[j, ] for j in range(self.dataset.shape[0])])/sum(U[:, i]**m)
                C.append(c)
            C = np.array(C)
            # print('聚类中心矩阵C: ', "\n", C)
            # 计算目标函数J
            for i in range(self.cluster_center_number):
                j = sum([U[j, i]**m*(self.distance(C[i, ], self.dataset[j, ])**2) for j in
                         range(self.dataset.shape[0])])
                J += j
            # print('J: ', J)
            # 更新U矩阵
            new_U = np.array([[0]*self.cluster_center_number]*self.dataset.shape[0], dtype='float')
            for i in range(self.cluster_center_number):
                for j in range(self.dataset.shape[0]):
                    new_U[j, i] = 1/sum([(self.distance(self.dataset[j], C[i])/self.distance(self.dataset[j], C[l]))
                                         ** (2/(m-1)) for l in range(self.cluster_center_number)])
            # print('new U: ', new_U)
            count = 0
            for i in range(self.dataset.shape[0]):
                for j in range(self.cluster_center_number):
                    if abs(new_U[i, j]-U[i, j]) <= epsilon:
                        count += 1
            if float(count) >= self.dataset.shape[0]*self.cluster_center_number*0.6:
                change = False

            U = new_U
        # print('c: ', C)

        return C, U

    def FCM_classify(self, C):
        """
        FCM算法的聚类结果
        :param C:聚类中心矩阵
        :return: 分类结果,list
        """
        point_classify = [0] * self.dataset.shape[0]  # 样本点的分类
        for i in range(self.dataset.shape[0]):
            mid = [self.distance(self.dataset[i, ], point) for point in C]
            point_classify[i] = mid.index(min(mid))

        if sum(point_classify) < len(point_classify)/2:  # 0比1多，交换0,1
            for i in range(len(point_classify)):
                if point_classify[i] == 0:
                    point_classify[i] = 1
                else:
                    point_classify[i] = 0

        return point_classify

    def initialise_U(self):
        """
        生成初始化隶属度矩阵U，每行加起来是1，输出为np.array结构
        """
        m = self.dataset.shape[0]
        n = self.cluster_center_number
        U = []
        for i in range(m):
            u = [random.randint(1, 10) for j in range(n)]
            u = np.array(u) / sum(u)
            U.append(u)

        return np.array(U)

    def validity_index(self, C, U):
        """
        计算聚类有效性指标
        :param C:聚类中心矩阵
        :param U:隶属度矩阵
        :return: v
        """
        n = self.dataset.shape[0]
        k = self.cluster_center_number
        # 计算模糊分测度,gamma
        fenzi = 0.
        for i in range(n):
            fenzi += sum([(U[i, j]-1/k)**2*(self.distance(self.dataset[i, ], C[j, ])) for j in range(k)])
        fenmu = min([self.distance(C[0], C[i]) for i in range(1, k)])*n
        gamma = fenzi/fenmu
        # 计算有效性指标v
        v = gamma*(-1./n)*sum([sum([U[i, j]*math.log(U[i, j]) for i in range(n) for j in range(k)])])

        return v

"""
----------------------------------------------------------------------------------------------------------------
"""


def write_classify(point_classifies, filename):
    """
    将聚类结果写入文件中
    :param point_classifies: (point_classify1,point_classify2,point_classify3,point_classify4)
    :param filename: excel文件
    :return: excel文件
    """
    data = xlrd.open_workbook(filename)
    # 创建一个Excel文件
    book = xlwt.Workbook(encoding='utf-8')

    for sheet_num, point_classify in zip(range(4), point_classifies):
        table = data.sheet_by_index(sheet_num)
        # 添加表
        sheet = book.add_sheet(str(sheet_num), cell_overwrite_ok=True)  # 创建表

        # 写入标题
        head = table.row_values(0)
        head.append('分类')
        print("写入表头： ", head)
        for i in range(len(head)):
            sheet.write(0, i, head[i])

        nrows = table.nrows  # 行数
        ncols = table.ncols+1  # 列数
        row_count = 1

        # 写入数据
        for row_num in range(1, nrows):
            if row_count <= len(point_classify):
                row = table.row_values(row_num)
                if row[0] != '':
                    row.append(point_classify[row_count-1])
                    for col_num in range(ncols):
                        sheet.write(row_count, col_num, row[col_num])
                        print(row_count, col_num, row[col_num])
                    row_count += 1

    book.save(r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\试验2#数据处理\能量大于40_已分类.xlsx")  # 保存文件


#  构造测试数据集,100个数据，四个类
def create_dataset():
    data1 = []
    for i in range(-5, 0, 1):
        for j in range(1, 6, 1):
            data1.append([i, j])

    data2 = []
    for i in range(1, 6, 1):
        for j in range(1, 6, 1):
            data2.append([i, j])

    data3 = []
    for i in range(-5, 0, 1):
        for j in range(-5, 0, 1):
            data3.append([i, j])

    data4 = []
    for i in range(1, 6, 1):
        for j in range(-5, 0, 1):
            data4.append([i, j])

    return np.array(data1+data2+data3+data4)


def main():
    filename = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\试验2#数据处理\各通道数据分离_滤波_能量大于40.xls"
    # data_tongdao = get_dataset(filename, 2)
    # data_tongdao_scale = data_processing(data_tongdao)
    #
    # for i in range(2, 8):
    #     k_means = K_Means(data_tongdao_scale[:, [1, 2, 3, 4, 8]], i)
    #     C, U = k_means.FCM(2)
    #     v = k_means.validity_index(C, U)
    #     print(i, v)

    point_classifies = []
    for sheet in range(4):
        data_tongdao = get_dataset(filename, sheet)
        data_tongdao_scale = data_processing(data_tongdao)
        k_means = K_Means(data_tongdao_scale[:, [1, 2, 3, 4, 8]], 2)
        C, U = k_means.FCM(2)
        point_classify = k_means.FCM_classify(C)
        print("分类： ", sum(point_classify))
        point_classifies.append(point_classify)
        print("point_classifies: ", point_classifies)

    write_classify(point_classifies, filename)


main()




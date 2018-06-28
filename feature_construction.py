# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureConstruction(object):
    def __init__(self, data):
        """
        输入数据集
        :param data: pandas.dataframe
        """
        self.data = data

    def sum_of_features(self, *args):
        """
        将多个特征相加组成新特征
        注意：需要先将特征进行0-1标准化，以消除量纲的影响
        :param args: 特征名称
        :return: np.ndarray
        """
        new_feature = StandardScaler().fit_transform(self.data[args[0]].values.reshape(-1,1))
        for i in args[1:]:
            new_feature += StandardScaler().fit_transform(self.data[i].values.reshape(-1,1))
        return new_feature

    def subtraction_of_features(self, *args):
        """
        将多个特征相减组成新特征
        :param args: 特征名称
        :return:  np.ndarray
        """
        new_feature = StandardScaler().fit_transform(self.data[args[0]].values.reshape(-1,1))
        for i in args[1:]:
            new_feature -= StandardScaler().fit_transform(self.data[i].values.reshape(-1,1))
        return new_feature

    def multiplication_of_features(self, *args):
        """
        将多个特征相乘组成新特征
        :param args: 特征名称
        :return:  np.ndarray
        """
        new_feature = StandardScaler().fit_transform(self.data[args[0]].values.reshape(-1,1))
        for i in args[1:]:
            new_feature *= StandardScaler().fit_transform(self.data[i].values.reshape(-1,1))
        return new_feature

    def division_of_features(self, *args):
        """
        将多个特征相除组成新特征
        :param args: 特征名称
        :return:  np.ndarray
        """
        new_feature = MinMaxScaler(feature_range=(1, 2)).fit_transform(self.data[args[0]].values.reshape(-1,1))
        for i in args[1:]:
            new_feature /= MinMaxScaler(feature_range=(1, 2)).fit_transform(self.data[i].values.reshape(-1,1))
        return new_feature

    def log_of_feature(self, arg):
        """
        自然对数特征
        :param arg: 特征名称
        :return: np.ndarray
        """
        return np.log(MinMaxScaler(feature_range=(1, 2)).fit_transform(self.data[arg].values.reshape(-1, 1)))

    def sqrt_of_feature(self, arg):
        """开方"""
        return np.square(self.data[arg])

    def square_of_feature(self, arg):
        """ 平方"""
        return np.square(MinMaxScaler(feature_range=(1, 2)).fit_transform(self.data[arg].values.reshape(-1, 1)))

    def exp_of_feature(self, arg):
        """ exp"""
        return np.exp(StandardScaler().fit_transform(self.data[arg].values.reshape(-1, 1)))

    def accumulation_of_feature(self, arg):
        """
        累加
        :param arg: 特征名称，能量
        :return: np.ndarray
        """
        return self.data[arg].cumsum(axis=0)






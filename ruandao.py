# -*- coding:utf-8 -*-

import struct
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import heapq
import xlrd
import xlwt
import csv

"""
----------*---------软岛公司仪器--------------*----------------------------*---------------------------*--------------
-------------------------------------------------------------------------------------------------------------
"""
# file = r"F:\DS180125_143037M\DS180125_143037_D000220A.dat"


# 打开一个软岛全波形文件，并找出1,2,3,4通道的数据，存为字典格式
def open_one_file(file):
    with open(file, 'rb') as fl:
        tongdao_1 = []
        i = 0
        read_data = fl.read()
        while i < 52428800:
            data = struct.unpack('H', read_data[i:2+i])[0]
            i += 16
            tongdao_1.append(data)

        tongdao_2 = []
        i = 4
        while i < 52428800:
            data = struct.unpack('H', read_data[i:2+i])[0]
            i += 16
            tongdao_2.append(data)

        tongdao_3 = []
        i = 8
        while i < 52428800:
            data = struct.unpack('H', read_data[i:2+i])[0]
            i += 16
            tongdao_3.append(data)

        tongdao_4 = []
        i = 12
        while i < 52428800:
            data = struct.unpack('H', read_data[i:2+i])[0]
            i += 16
            tongdao_4.append(data)

    data_dict = {}
    data_dict['1'] = tongdao_1
    data_dict['2'] = tongdao_2
    data_dict['3'] = tongdao_3
    data_dict['4'] = tongdao_4

    # for i, j in enumerate(data_dict.keys()):
    #     plt.subplot(411+i)
    #     plt.plot(range(len(data_dict[j])), data_dict[j], len(data_dict[j]), 33094)
    #     # plt.yscale = [30000, 34000]
    #     # xlim(0, 3500000)
    #     plt.axis([0, 3500000, 30000, 34000])
    #     new_ticks = np.arange(30000, 34000, 1000)
    #     plt.yticks(new_ticks)
    #     plt.grid(True)
    #
    # plt.show()

    return data_dict

# open_one_file(file)

path = r'F:\DS180125_143037M'


# 找出软岛全波形文件里有信号的文件
def find_signal(path):
    filelist = os.listdir(path)
    have_signal = []
    for count, file in enumerate(filelist[4:-5]):
        with open(os.path.join(path, file), 'rb') as fl:
            i = 0
            j = 0
            read_data = fl.read()
            while i < 52428800:
                data = struct.unpack('H', read_data[i:2+i])[0]
                if data >= 33094:  # 门槛为40db
                    print(count, True)
                    have_signal.append(count)
                    break
                i += 4
                j += 1
    return have_signal

# find_signal(path)


# 截取一个文件里的AE信号
def cut_out_AE(data_dict):
    for i in heapq.nlargest(1, data_dict['1']):
        if i >= 33100:
            max_index = data_dict['1'].index(i)
            data_AE = data_dict['1'][(max_index-3500):(max_index+16000)]

            plt.plot(range(len(data_AE)), data_AE, range(len(data_AE)), [33000]*len(data_AE))
            plt.show()


# data_dict = open_one_file(file)
# cut_out_AE(data_dict)

"""
----------*---------物理声公司仪器--------------*----------------------------*---------------------------*--------------
---------------------------------------------------------------------------------------------------------------
"""

# file = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\试验2#数据处理\胶拼轴拉试验2#20180123.xlsx"


def tongdao_AE_divide(file):
    """
    将1,2,3,4通道的信号分别提取出来，并保存到Excel文件
    :param file:
    :return:
    """
    data = xlrd.open_workbook(file)
    table = data.sheet_by_index(0)  # 获取表
    nrows = table.nrows  # 获取行数
    ncols = table.ncols  # 获取列数

    # 创建一个Excel文件
    book = xlwt.Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('tongdao1', cell_overwrite_ok=True)  # 创建表
    sheet2 = book.add_sheet('tongdao2', cell_overwrite_ok=True)
    sheet3 = book.add_sheet('tongdao3', cell_overwrite_ok=True)
    sheet4 = book.add_sheet('tongdao4', cell_overwrite_ok=True)

    # 写入标题
    head = table.row_values(0)
    for i in range(len(head)):
        sheet1.write(0, i, head[i])
        sheet2.write(0, i, head[i])
        sheet3.write(0, i, head[i])
        sheet4.write(0, i, head[i])

    row_count1 = 1  # 行计数
    row_count2 = 1
    row_count3 = 1
    row_count4 = 1

    # 写入数据
    for row_num in range(1, nrows):
        # # 转化date格式
        # for col_num in range(ncols):
        #     if table.cell(row_num, col_num).ctype == 3:  # 3表示date格式
        #         data = xlrd.xldate_as_datetime(table.cell(row_num, col_num).value, 1)
        #         # print(data)
        row = table.row_values(row_num)
        # if type(row[1]) == float:
        #     row[1] = data

        # 写入tongdao1
        if row[2] == 1.:
            for col_num in range(ncols):
                print("写入通道1: ", row)
                sheet1.write(row_count1, col_num, row[col_num])
            row_count1 += 1
        # 写入tongdao2
        if row[2] == 2.:
            for col_num in range(ncols):
                print("写入通道2: ", row)
                sheet2.write(row_count2, col_num, row[col_num])
            row_count2 += 1
        # 写入tongdao3
        if row[2] == 3.:
            for col_num in range(ncols):
                print("写入通道3: ", row)
                sheet3.write(row_count3, col_num, row[col_num])
            row_count3 += 1
        # 写入tongdao4
        if row[2] == 4.:
            for col_num in range(ncols):
                print("写入通道4: ", row)
                sheet4.write(row_count4, col_num, row[col_num])
            row_count4 += 1

        # 写入空行
        if row[2] == '':
            print("写入空行:", row)
            sheet1.write(row_count1, 0, '')
            sheet2.write(row_count2, 0, '')
            sheet3.write(row_count3, 0, '')
            sheet4.write(row_count4, 0, '')
            row_count1 += 1
            row_count2 += 1
            row_count3 += 1
            row_count4 += 1

    book.save(r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\试验2#数据处理\各通道数据分离.xlsx")  # 保存文件

# tongdao_AE_divide(file)

# file = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\试验1#数据处理\胶拼轴拉试验1#20180122.xlsx"
# file = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\试验2#数据处理\各通道数据分离.xls"


# 滤波，基于能量滤波
def lv_bo(file, threshold=0):
    """
    滤波
    :param file:
    :param threshold: 门槛
    :return:
    """
    data = xlrd.open_workbook(file)
    book = xlwt.Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('tongdao1', cell_overwrite_ok=True)
    sheet2 = book.add_sheet('tongdao2', cell_overwrite_ok=True)
    sheet3 = book.add_sheet('tongdao3', cell_overwrite_ok=True)
    sheet4 = book.add_sheet('tongdao4', cell_overwrite_ok=True)

    for sheet_num, new_sheet_name in zip(range(4), [sheet1, sheet2, sheet3, sheet4]):
        table = data.sheet_by_index(sheet_num)
        # 写入标题
        head = table.row_values(0)
        for i in range(len(head)):
            new_sheet_name.write(0, i, head[i])

        nrows = table.nrows  # 获取行数
        ncols = table.ncols  # 获取列数
        row_count = 1  # 行计数

        # 写入数据
        for row_num in range(1, nrows):
            row = table.row_values(row_num)
            if type(table.cell(row_num, 5).value) == float and table.cell(row_num, 5).value >= threshold:
                for col_num in range(ncols):
                    print("写入通道%d: " % sheet_num, row)
                    new_sheet_name.write(row_count, col_num, row[col_num])
                row_count += 1
            elif table.cell(row_num, 5).value == '':
                print("写入空行:", row)
                new_sheet_name.write(row_count, 0, '')
                row_count += 1

    book.save(r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\试验2#数据处理\各通道数据分离_滤波_能量大于40.xlsx")  # 保存文件


# lv_bo(file, 40)


def read_boxing(*files):
    """
    显示物理声多个文件波形
    :param file: csv波形文件
    :return:
    """
    # file1, file2, file3 = file
    for i, file in enumerate(files):
        with open(file, 'r') as fl:
            reader = csv.reader(fl)
            y = []
            count = 0
            for row in reader:
                count += 1
                if count > 12:
                    y.append(float(row[0]))
            plt.subplot(len(files)*100+10+i+1)
            plt.plot(range(len(y)), y)

    # plt.subplot(312)
    # plt.plot(range(len(y2)), y2)
    # plt.subplot(313)
    # plt.plot(range(len(y3)), y3)
    plt.show()

file1 = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_1_15_1021999143.csv"  # 1
file2 = f"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_1_35_1038671228.csv"  # 1
file3 = f"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_1_59_1045426596.csv"  # 1
file4 = f"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_1_66_1047696998.csv"  # 1

file5 = f"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_1_19_1030831100.csv"  # 0
file6 = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_1_31_1036953879.csv"  # 0
file7 = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_1_264_1089104945.csv"  # 0
file8 = f"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_1_274_1090163173.csv"  # 0


# read_boxing(file1, file2, file3, file4, file5, file6, file7, file8)


def draw_boxing(file):
    """
    画出典型模式的波形图
    :param file:  .csv
    :return:
    """
    with open(file, 'r') as fl:
        reader = csv.reader(fl)
        y = []
        count = 0
        for row in reader:
            count += 1
            if count > 12:
                y.append(float(row[0]))
        # fig = plt.figure()
        plt.plot(range(len(y)), y, color='b', lw=1)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.xlim(0, 1024)
        plt.xlabel('time/μs', fontsize=18, family='Times New Roman')
        plt.ylabel('voltage/V', fontsize=18, family='Times New Roman')
        plt.xticks(fontsize=15, family='Times New Roman')
        plt.yticks(fontsize=15, family='Times New Roman')
        plt.show()

file9 = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_2_7371_2284342323.csv"  # 0
file10 = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\胶拼轴拉试验1#20180122\分级加载破坏正式加载_1_3452_2222068654.csv"  # 1
# draw_boxing(file10)


# 绘制折线图
def draw_time_deformation_count_linegrah(file_yinbian, file_fenlei):
    """
    画出时间-应变-分类累积计数 双纵坐标轴图----折线图
    :param file_yinbian: file = r"G:\声发射资料\应变数据-郑强\1#\20180122-1-分级加载9T破坏.xls"
    :param file_fenlei:  file=r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\试验1#数据处理\能量大于40_已分类.xls"
    :return:
    """
    # 获取应变数据
    data_yinbian = xlrd.open_workbook(file_yinbian)
    table_yinbian = data_yinbian.sheet_by_index(0)  # 获取表
    nrows_yinbian = table_yinbian.nrows

    y_yinbian = []
    for row in range(1, nrows_yinbian):
        try:
            y_yinbian.append(float(table_yinbian.row_values(row)[-1]))
        except:
            print(file_yinbian, table_yinbian.row_values(row))

    x = np.array([i for i in range(len(y_yinbian[:-1000]))])
    x = x / 100
    # 将应变数据向右偏移200s
    x = [i for i in np.arange(0, 250, 0.01)] + [i + 250 for i in x]
    y_yinbian = [0] * 25000 + y_yinbian

    # 获取分类累积计数数据
    data_fenlei = xlrd.open_workbook(file_fenlei)
    table_fenlei = data_fenlei.sheet_by_index(3)  # 获取表
    nrows_fenlei = table_fenlei.nrows

    fenlei = []
    time = []  # 时间
    for row in range(1, nrows_fenlei):
        try:
            fenlei.append(float(table_fenlei.row_values(row)[-2]))
            time.append(float(table_fenlei.row_values(row)[-1]))
        except:
            print(file_fenlei, table_fenlei.row_values(row))

    y_fenlei_0 = [len(fenlei[:i + 1]) - sum(fenlei[:i + 1]) for i in range(len(fenlei))]  # 0的累积计数
    y_fenlei_1 = [sum(fenlei[:i + 1]) for i in range(len(fenlei))]  # 1的累积计数

    matplotlib.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
    matplotlib.rcParams['ytick.direction'] = 'in'  # 刻度线朝内

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l1, = ax1.plot(x, y_yinbian[:-1000], 'k', linewidth=2.5, label='deformation')  # 采样频率1/0.01s
    ax1.set_ylabel('deformation/με', fontsize=18, family='Times New Roman')
    ax1.set_xlabel('time/s', fontsize=18, family='Times New Roman')
    ax1.set_xlim(0, 2500)
    ax1.set_ylim(0, 250)
    ax1.tick_params(axis='x', size=7)
    ax1.tick_params(axis='y', size=7)

    ax2 = ax1.twinx()
    l2, = ax2.plot(time, y_fenlei_0, 'r', linewidth=2.5, label='the first classical AE signal')
    l3, = ax2.plot(time, y_fenlei_1, 'g', linewidth=2.5, label='the second classical AE signal')
    ax2.set_ylabel("count", fontsize=18, family='Times New Roman')
    ax2.set_ylim(0, 650)
    ax2.tick_params(axis='y', size=7)

    plt.legend(handles=[l1, l2, l3],
               labels=['deformation', 'the first classical AE signal', 'the second classical AE signal'],
               loc='upper left', frameon=False)  # 增加图例
    plt.show()

    # plt.xlim(0, x[-1].max()*1.1)
    # plt.ylim(0, max(y[:-1000])*1.1)
    # plt.xticks([0, 500, 1000, 1500, 2000])
    # plt.xlabel('time/s', fontsize=18, family='Times New Roman')
    # plt.ylabel('deformation/με', fontsize=18, family='Times New Roman')
    # plt.xticks(fontsize=15, family='Times New Roman')
    # plt.yticks(fontsize=15, family='Times New Roman')
    # plt.show()


file_yinbian = r"G:\声发射资料\应变数据-郑强\1#\20180123-2-分级加载.xls"
file_fenlei = r"G:\声发射资料\节段胶拼轴拉试验201801-王少帅\试验2#数据处理\能量大于40_已分类.xls"
# draw_time_deformation_count_linegrah(file_yinbian, file_fenlei)


# 绘制柱状图
def draw_time_count_histogram(file_fenlei):
    """
    绘制0/1分类柱状图，时间划分为4个区间
    :param file:
    :return:
    """
    # 获取分类累积计数数据
    data_fenlei = xlrd.open_workbook(file_fenlei)
    table_fenlei = data_fenlei.sheet_by_index(3)  # 获取表，0,1,2,3
    nrows_fenlei = table_fenlei.nrows

    fenlei = []  # 分类，0,1
    time = []  # 时间
    for row in range(1, nrows_fenlei):
        try:
            fenlei.append(float(table_fenlei.row_values(row)[-2]))
            time.append(float(table_fenlei.row_values(row)[-1]))
        except:
            print(file_fenlei, table_fenlei.row_values(row))

    y_fenlei_0 = []
    y_fenlei_1 = []
    for i in range(1, len(time)):
        if time[i-1] < 700 <= time[i+1]:
            y_fenlei_0.append(len(fenlei[:i])-sum(fenlei[:i]))
            y_fenlei_1.append(sum(fenlei[:i]))
            mark_1300 = i

        elif time[i-1] < 900 <= time[i+1]:
            y_fenlei_0.append(len(fenlei[mark_1300-1:i])-sum(fenlei[mark_1300-1:i]))
            y_fenlei_1.append(sum(fenlei[mark_1300-1:i]))
            mark_1600 = i

        elif time[i-1] < 1200 <= time[i+1]:
            y_fenlei_0.append(len(fenlei[mark_1600-1:i])-sum(fenlei[mark_1600-1:i]))
            y_fenlei_1.append(sum(fenlei[mark_1600-1:i]))

            y_fenlei_0.append(len(fenlei[i:]) - sum(fenlei[i:]))
            y_fenlei_1.append(sum(fenlei[i:]))

    matplotlib.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
    matplotlib.rcParams['ytick.direction'] = 'in'  # 刻度线朝内

    y_fenlei_0 = [y_fenlei_0[i] for i in [0, 2, 4, 5]]  # 挑选出非重复的部分
    y_fenlei_1 = [y_fenlei_1[i] for i in [0, 2, 4, 5]]

    name_list = ['I', 'II', 'III', 'IV']
    x = [0, 1, 2, 3]
    width = 0.2
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, y_fenlei_0, width, color='w', edgecolor='b', hatch="//")
    rects2 = ax.bar([i+width for i in x], y_fenlei_1, width, color='w', edgecolor='r', hatch="//")
    ax.set_xticks(np.array(x)+width/2)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xticklabels(name_list)
    ax.set_ylabel("frequency", fontsize=18, family='Times New Roman')
    ax.set_xlabel("phases", fontsize=18, family='Times New Roman')
    ax.legend([rects1[0], rects2[1]], ['the first classical AE signal', 'the second classical AE signal'],
              frameon=False, fontsize=13)
    plt.show()

    # print("y_fenlei_1:")
    # for i in y_fenlei_1:
    #     print(i)
    # print("y_fenlei_0:")
    # for i in y_fenlei_0:
    #     print(i)


draw_time_count_histogram(file_fenlei)






import os
import re

import pandas as pd

from mytools import CommanTools


def find_missing_y():
    """找到缺失的y,发现文件名没有问题"""
    txt_list_path = []
    dir = '../data/input_data'
    for second_dir_name in os.listdir(dir):
        for file_name in os.listdir(dir + '/' + second_dir_name):
            if file_name.endswith('.txt'):
                txt_list_path.append(dir + '/' + second_dir_name + '/' + file_name)
    for each in txt_list_path:
        if len(re.findall('[\u4e00-\u9fa5]', each)) != 1:
            print(each)


def find_missing_y_by_frequent(data):
    """发现 工手微甘年爪凶兄六弓巴拜 miss，此时应该是空，即y空可以丢弃"""
    res = data['y'].groupby(by='y').count()
    for each in res:
        continue


def check_test():
    """观察交集判断是否有误"""
    from knn import KNearestNeighbourClf
    map_df = CommanTools.read_chinese_parts()
    for each in map_df.columns:
        map_df[each] = pd.to_numeric(map_df[each])

    x_standard = pd.DataFrame([1, 0.5, 0.271875, 0.56875, 0.04375]).T
    x_standard.columns = ['code', 'x', 'y', 'w', 'h']

    x = pd.DataFrame([1, 78, 67, 94, 252]).T
    x.columns = [0, 1, 2, 3, 4]
    for each in x.columns:
        x[each] = pd.to_numeric(x[each])

    map_df = CommanTools.read_chinese_parts()
    for each in map_df.columns:
        map_df[each] = pd.to_numeric(map_df[each])
    code_columns = [i for i in range(0, 5, 5)]
    for each in code_columns:
        # code 转换
        x[each] = pd.to_numeric(x[each])
        merge_result = x[[each]].merge(map_df, how='left', left_on=each, right_on='mode_code')['standard_code']
        x[each] = merge_result
        # x y w h 转换
        temp_x = (x[each + 2] + x[each + 4]) / 640
        temp_y = (x[each + 1] + x[each + 3]) / 640
        temp_w = abs(x[each + 2] - x[each + 4]) / 320
        temp_h = abs(x[each + 1] - x[each + 3]) / 320
        x.loc[:, each + 1] = temp_x
        x.loc[:, each + 2] = temp_y
        x.loc[:, each + 3] = temp_w
        x.loc[:, each + 4] = temp_h

        x.loc[:, each + 1] = x.loc[:, each + 1].apply(lambda x0: -1 if x0 < 0 else x0)
        x.loc[:, each + 2] = x.loc[:, each + 2].apply(lambda x0: -1 if x0 < 0 else x0)
        x.loc[:, each + 3] = x.loc[:, each + 3].apply(lambda x0: -1 if x0 == 0 else x0)
        x.loc[:, each + 4] = x.loc[:, each + 4].apply(lambda x0: -1 if x0 == 0 else x0)

    x = list(x.iloc[0, :])
    x_standard = list(x_standard.iloc[0, :])

    min_x1 = x[1] - x[4] / 2
    max_x1 = x[1] + x[4] / 2
    min_x2 = x_standard[1] - x_standard[4] / 2
    max_x2 = x_standard[1] + x_standard[4] / 2
    min_y1 = x[2] - x[3] / 2
    max_y1 = x[2] + x[3] / 2
    min_y2 = x_standard[2] - x_standard[3] / 2
    max_y2 = x_standard[2] + x_standard[3] / 2
    minx = max(min_x1, min_x2)
    miny = max(min_y1, min_y2)
    maxx = min(max_x1, max_x2)
    maxy = min(max_y1, max_y2)
    if minx > maxx or miny > maxy:
        # 无交集
        print("无交集")
        return False
    else:
        print("交集")

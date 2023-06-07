import os

import numpy as np
import pandas as pd
import re
from sklearn.neighbors import KNeighborsClassifier

from mytools import CommanTools


class KNearestNeighbourClf(object):
    @staticmethod
    def _load_data_by_reg(dir_path=r'data/input_data/new_label', pattern=''):
        columns = [i for i in range(0, 50)]
        x = pd.DataFrame(columns=columns)
        y = []
        for each_file in os.listdir(dir_path):
            if not each_file.endswith('.txt'):
                continue
            with open(dir_path + '/' + each_file, 'r', encoding='utf-8') as f:
                y.append(re.findall(pattern="([\u4e00-\u9fa5]).txt", string=each_file)[0])
                new_row = pd.DataFrame([-1 for i in range(50)]).T
                new_row.columns = columns
                x = pd.concat([x, new_row], axis=0)
                word_str = f.read()
                word_list = word_str.split('\n')
                word_list = [i for i in word_list if i != '']
                for i in range(len(word_list)):
                    found_data = re.findall(pattern, word_list[i])
                    found_data = found_data[0]
                    x.iloc[len(x) - 1, 5 * i:5 * i + 5] = list(found_data)
        x.reset_index(inplace=True, drop=True)
        return x, y

    @staticmethod
    def load_standard_data(dir_path=r'data/input_data/new_label'):
        return KNearestNeighbourClf._load_data_by_reg(dir_path, "(.*?) (.*?) (.*?) (.*?) (.*)")

    @staticmethod
    def load_model_data(dir_path=r'data/input_data/new_label'):
        return KNearestNeighbourClf._load_data_by_reg(dir_path, r"\(b'(.*?) .*?', (.*?), (.*?), (.*?), (.*?)\)")

    @staticmethod
    def model_data_convert(x):
        """
        1.对x的编码转换（依据chinese_parts文件）
        2.转换成x,y,w,h形式
        :return: x
        """
        map_df = CommanTools.read_chinese_parts()
        for each in map_df.columns:
            map_df[each] = pd.to_numeric(map_df[each])
        for each in x.columns:
            x[each] = pd.to_numeric(x[each])
        code_columns = [i for i in range(0, 50, 5)]
        for each in code_columns:
            x[each] = pd.to_numeric(x[each])
            merge_result = x[[each]].merge(map_df, how='left', left_on=each, right_on='mode_code')['standard_code']
            x[each] = merge_result

            temp_x = (x[each + 2] + x[each + 4]) / 640
            temp_y = (x[each + 1] + x[each + 3]) / 640
            temp_w = abs(x[each + 2] - x[each + 4]) / 320
            temp_h = abs(x[each + 1] - x[each + 3]) / 320

            x.loc[:, each + 1] = temp_x.apply(lambda x0: -1 if x0 < 0 else x0)
            x.loc[:, each + 2] = temp_y.apply(lambda x0: -1 if x0 < 0 else x0)
            x.loc[:, each + 3] = temp_w.apply(lambda x0: -1 if x0 == 0 else x0)
            x.loc[:, each + 4] = temp_h.apply(lambda x0: -1 if x0 == 0 else x0)

        return x

    @staticmethod
    def distance_func(x_input, x_standard):
        num1, num2 = KNearestNeighbourClf._get_part_num(x_input), KNearestNeighbourClf._get_part_num(x_standard)
        if num1 > num2:
            bigger_word, smaller_word = x_input, x_standard
            bigger_len, smaller_len = num1, num2
            standard_is_shorter = True
        else:
            bigger_word, smaller_word = x_standard, x_input
            bigger_len, smaller_len = num2, num1
            standard_is_shorter = False

        has_used = [False for i in range(bigger_len)]
        distance = 0
        for i in range(smaller_len):
            distance_matrix = []
            for j in range(bigger_len):
                if not has_used[j]:
                    dis = KNearestNeighbourClf._calculate_part_distance(smaller_word[i * 5:i * 5 + 5],
                                                                        bigger_word[j * 5:j * 5 + 5],
                                                                        standard_is_shorter)
                    distance_matrix.append(dis)
                else:
                    distance_matrix.append(np.inf)

            distance_this_part = min(distance_matrix)
            j = distance_matrix.index(distance_this_part)
            has_used[j] = False
            distance += distance_this_part

        distance += 2 * (bigger_len - smaller_len)
        return distance

    @staticmethod
    def _get_part_num(x):
        for i in range(0, 50, 5):
            if x[i] == -1:
                return int(i / 5)
        return 0

    @staticmethod
    def _calculate_part_distance(x_smaller, x_bigger, first_para_is_standard):
        """
        :return:[0，1]
        """
        if x_smaller[0] == x_bigger[0]:
            part_dis = 0
        else:
            part_dis = 1

        bottom_x = min(x_smaller[1] + x_smaller[4] / 2, x_bigger[1] + x_bigger[4] / 2)
        top_x = max(x_smaller[1] - x_smaller[4] / 2, x_bigger[1] - x_bigger[4] / 2)
        dx_mixed = bottom_x - top_x
        right_y = min(x_smaller[2] + x_smaller[3] / 2, x_bigger[2] + x_bigger[3] / 2)
        left_y = max(x_smaller[2] - x_smaller[3] / 2, x_bigger[2] - x_bigger[3] / 2)
        dy_mixed = right_y - left_y
        area_mixed = max(dx_mixed, 0) * max(dy_mixed, 0)

        area1 = x_smaller[3] * x_smaller[4]
        area2 = x_bigger[3] * x_bigger[4]
        if first_para_is_standard:
            area_dis = 1 - (area_mixed / area1)
        else:
            area_dis = 1 - (area_mixed / area2)
        return 2 * part_dis + area_dis

    def __init__(self, distance_func):
        self.clf = KNeighborsClassifier(n_neighbors=1, algorithm='auto', metric=distance_func)

    def fit(self, x, y):
        """记住数据集"""
        self.clf.fit(x, y)

    def predict(self, x):
        """计算与每个字的差距"""
        return self.clf.predict(x)

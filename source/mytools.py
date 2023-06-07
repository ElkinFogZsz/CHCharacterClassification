"""
Remove stupid smells
"""

import os
import re
import random
from functools import wraps

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import seaborn as sns


class CommanTools(object):
    """
    一个通过工具箱，内置各种静态函数，包括数据读取、数据集划分、数据预处理、预测结果等等建模通用的函数
    """

    @staticmethod
    def union_load_train_data(pathlist):
        """将训练集中的数据合并为一个dataframe，最多7个部首，以padding方式补全"""
        columns_name = []
        for i in range(10):
            columns_name.extend('code' + i, 'proba' + i, 'lf_x' + i, 'lf_y' + i, 'rb_x' + i, 'rb_y' + i)
        x = pd.DataFrame(columns=columns_name)
        for each_path in pathlist:
            one_word_df = CommanTools._load_one_file_model_res(each_path)
            for i in range(len(one_word_df)):
                x = pd.concat(pd.DataFrame([0] * 60, columns=columns_name))
                x.iloc[len(x) - 1, i * 6:i * 6 + 5] = one_word_df.iloc[i, :]
        return x

    @staticmethod
    def type_change(data, type_str=""):
        """
        TODO:加入类型参数以及check装饰器
        数据类型转化 + 测试集划分
        :param data:
        :param model:
        :return: X_train, X_test, y_train, y_test
        """
        # Step1. 数据类型转换
        data['word'] = LabelEncoder().fit_transform(data['word']).astype('int')
        for each in data.columns:
            if each != 'y':
                data[each] = pd.to_numeric(data[each], errors='coerce').astype('int')
            else:
                data[each] = data[each].astype('int')
        # Step2. 数据集分割 ， 选用相同的划分方式
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1],
                                                            data.iloc[:, -1],
                                                            test_size=0.25,
                                                            random_state=9527)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def multi_classification_PR_performance(y_test, y_predict_clf):
        """

        :param y_test:
        :param y_predict_clf:分类的预测结果
        :return:
        """
        micro_precision = precision_score(y_test, y_predict_clf, average='micro')
        macro_precision = precision_score(y_test, y_predict_clf, average='macro')
        print('micro-precision:  ', micro_precision)
        print('macro-precision:  ', macro_precision)

        micro_recall = recall_score(y_test, y_predict_clf, average='micro')
        macro_recall = recall_score(y_test, y_predict_clf, average='macro')
        print('micro-recall:  ', micro_recall)
        print('macro-recall:  ', macro_recall)

    @staticmethod
    def multi_classification_AUC_performance_auc(y_test, y_predict_proba):
        """
        get auc score
        :param y_test:
        :param y_predict_proba:
        :return:
        """
        macro_auc = roc_auc_score(y_test, y_predict_proba, average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_test, y_predict_proba, average='micro', multi_class='ovr')
        print('分类计算macro-AUC:{}   全局计算micro-AUC:{}', macro_auc, micro_auc)

    @staticmethod
    def read_train_test_data(train_path, text_path):
        """读取train,test文件列表，返回train数据集"""
        x_train, y_train = CommanTools.load_fileList_to_dataframe_model_res(train_path)

        return x_train, y_train, text_path

    @staticmethod
    def load_fileList_to_dataframe_model_res(path_list, columns=['code', 'proba', 'lf_x', 'lf_y', 'rb_x', 'rb_y', 'number_part']):
        """
        将多个文件合并成一个dataframe
        :param path_list:
        :return:x,y
        """
        train = pd.DataFrame(columns=columns)
        test = []
        for each in path_list:
            # 读取训练集中的所有数据
            data_finded = CommanTools._load_one_file_model_res(each, columns)
            train = pd.concat([train, data_finded], axis=0)
            test.extend([re.findall(r"/([\u4e00-\u9fa5]).txt", each)[0]] * len(data_finded))
        return train, test

    @staticmethod
    def load_fileList_to_dataframe_standard_res(path_list, columns=['code', 'x', 'y', 'w', 'h']):
        """
        将多个文件合并成一个dataframe
        :param path_list:
        :return:x,y
        """
        train = pd.DataFrame(columns=columns)
        test = []
        for each in path_list:
            # 读取训练集中的所有数据
            data_finded = CommanTools._load_one_file_standard_res(each, columns)
            train = pd.concat([train, data_finded], axis=0)
            test.extend([re.findall(r"/([\u4e00-\u9fa5]).txt", each)[0]] * len(data_finded))
        return train, test

    @staticmethod
    def _load_one_file_model_res(path, columns=['code', 'proba', 'lf_x', 'lf_y', 'rb_x', 'rb_y']):
        """
        返回一个汉字文件对应的datafarme
        :param path:
        :return:
        """
        with open(path, 'r') as f:
            content = f.read()
            data_finded = re.findall(r"\(b'(.*?) (.*?)', (.*?), (.*?), (.*?), (.*?)\)", content)
            number_part = len(data_finded)
            data_finded = pd.DataFrame(data_finded, columns=columns)
            data_finded['number_part'] = [number_part] * number_part
        return data_finded

    @staticmethod
    def _load_one_file_standard_res(path, columns=['code', 'x', 'y', 'w', 'h']):
        with open(path, 'r') as f:
            content = f.read()
            data_finded = re.findall("(.*?) (.*?) (.*?) (.*?) (.*?)\n", content)
            number_part = len(data_finded)
            data_finded = pd.DataFrame(data_finded, columns=columns)
            data_finded['number_part'] = [number_part] * number_part
        return data_finded

    @staticmethod
    def pre_process_data_model_res(dataframe, continuous_columns=[], exponent_list=[]):
        """
        第一次调用（训练集）会进行boxcox转化并返回λ数组，第二次调用时（测试集）传入λ数组，根据该数组进行转化。
        :param dataframe:
        :param continuous_columns:
        :param exponent_list:
        :return:对于训练集，会返回dataframe,expoent_list,对于测试集会返回dataframe
        """
        for each_col in dataframe.columns:
            if each_col == 'proba':
                dataframe[each_col] = [1 for i in dataframe[each_col]]
            dataframe[each_col] = dataframe[each_col].astype(int)
        dataframe = dataframe.drop(columns=['proba'])
        # 离散特征构造
        # # 中点特征
        # dataframe['mid_x'] = (dataframe['lf_x'] + dataframe['rb_x'])/2
        # dataframe['mid_y'] = (dataframe['lf_y'] + dataframe['rb_y']) / 2
        # dataframe['mid_x'] = pd.cut(dataframe['mid_x'], bins=[0, 100, 150, 200, np.inf], labels=[1, 2, 3, 4])
        # dataframe['mid_y'] = pd.cut(dataframe['mid_y'], bins=[0, 100, 150, 200, np.inf], labels=[1, 2, 3, 4])
        # # 面积特征
        # dataframe['area'] = abs((dataframe['lf_y'] - dataframe['rb_y']) * (dataframe['lf_x'] - dataframe['rb_x']))
        # dataframe['area'] = pd.cut(dataframe['area'], bins=[0, 5000, 15000, np.inf], labels=[1, 2, 3])
        # # 根据eda观察的结果，对其进行如下规则的分箱
        # dataframe['lf_x'] = pd.cut(dataframe['lf_x'], bins=[0, 50, 80, 120, 200, np.inf], labels=[1, 2, 3, 4, 5])
        # dataframe['lf_y'] = pd.cut(dataframe['lf_y'], bins=[0, 50, 100, 200, np.inf], labels=[1, 2, 3, 4])
        # dataframe['rb_x'] = pd.cut(dataframe['rb_x'], bins=[0, 100, 200, 270, np.inf], labels=[1, 2, 3, 4])
        # dataframe['rb_y'] = pd.cut(dataframe['rb_y'], bins=[0, 100, 175, 270, np.inf], labels=[1, 2, 3, 4])
        # 如果是在predict时调用，则进行数据转化
        if len(exponent_list) != 0:
            for i in range(len(exponent_list)):
                _lambda = exponent_list[i]
                # box-cox公式
                dataframe[continuous_columns[i]] = [(pow(y, _lambda) - 1) / _lambda for y in
                                                    dataframe[continuous_columns[i]]]
            return dataframe
        elif len(continuous_columns) != 0:
            exponent_list = CommanTools.continues_normal(dataframe, continuous_columns)
            return dataframe, exponent_list

    @staticmethod
    def encode_part_from_model_to_res(dataframe, map_df):
        """将模型结果转化成标准结果"""
        result = dataframe.merge(map_df, how='left', left_on='code', right_on='mode_code')
        return result['standard_code']

    @staticmethod
    def read_chinese_parts():
        """
        读取chines_parts文件
        :return: 返回两列数据的dataframe,第一列为mode_code,第二列为standard_code
        """
        with open(r'data/input_data/chinese_parts.txt') as f:
            _encode_str = f.read()
        _encode_list = _encode_str.split('\n')
        map_df = pd.DataFrame(columns=['mode_code', 'standard_code'])
        for each in _encode_list:
            temp = re.findall('  (.*?): (.*)', each)
            if len(temp) == 0:
                continue
            model_code = temp[0][0]
            standard_code = temp[0][1]
            new_row = pd.DataFrame({'mode_code': [model_code], 'standard_code': [standard_code]})
            map_df = pd.concat([map_df, new_row], axis=0)
        return map_df

    @staticmethod
    def pre_process_model_data(dataframe, continuous_columns=[], exponent_list=[]):
        """
        将预测结果与标准数据对齐，code,x,y,w,h并正态化
        :param dataframe:
        :param continuous_columns:
        :param exponent_list:
        :param map_df:code转化的
        :return:对于训练集，会返回dataframe,expoent_list,对于测试集会返回dataframe
        """
        for each_col in ['lf_x', 'lf_y', 'rb_x', 'rb_y']:
            dataframe[each_col] = pd.to_numeric(dataframe[each_col], 'coerce')
        # 模型的结果和给出的结果的xy顺序是不同的，根据‘两’字宋体的结果对出来的
        dataframe['x'] = (dataframe['lf_y'] + dataframe['rb_y']) / 2
        dataframe['y'] = (dataframe['lf_x'] + dataframe['rb_x']) / 2
        dataframe['w'] = abs(dataframe['lf_y'] - dataframe['rb_y'])
        dataframe['h'] = abs(dataframe['lf_x'] - dataframe['rb_x'])
        dataframe = dataframe.drop(columns=['proba', 'lf_x', 'lf_y', 'rb_x', 'rb_y'])
        map_df = CommanTools.read_chinese_parts()
        dataframe['code'] = CommanTools.encode_part_from_model_to_res(dataframe[['code']], map_df)
        for i in range(len(exponent_list)):
            _lambda = exponent_list[i]
            # box-cox公式
            dataframe[continuous_columns[i]] = [(pow(y, _lambda) - 1) / _lambda for y in
                                                dataframe[continuous_columns[i]]]
        return dataframe

    @staticmethod
    def pre_process_data_standard_res(dataframe, continuous_columns=[], exponent_list=[]):
        """
        训练集调用的方式，对连续数据*320，并进行box-cox正态化。
        :param dataframe:
        :param continuous_columns:
        :param exponent_list:
        :return:对于训练集，会返回dataframe,expoent_list,对于测试集会返回dataframe
        """
        for each_col in dataframe.columns:
            if each_col in continuous_columns:
                dataframe[each_col] = pd.to_numeric(dataframe[each_col], 'coerce')
                dataframe[each_col] = dataframe[each_col] * 320
                exponent_list = CommanTools.continues_normal(dataframe, continuous_columns)
        return dataframe, exponent_list

    @staticmethod
    def continues_normal(dataframe, continuous_columns):
        """
        λ list
        :param dataframe:
        :param continuous_columns:
        :return:exponent_list
        """
        exponent_list = []
        if len(continuous_columns) != 0:
            for each in continuous_columns:
                x = np.array(dataframe[each])
                # Laplace平滑
                x = [i + 1 for i in x]
                dataframe[each], exponent = stats.boxcox(x)
                exponent_list.append(exponent)
        return exponent_list

    @staticmethod
    def get_all_second_son_file_path(dir):
        """获取二级文档数据"""
        path_list = []
        for second_dir_name in os.listdir(dir):
            for file_name in os.listdir(dir + '/' + second_dir_name):
                if file_name.endswith('.txt'):
                    path_list.append(dir + '/' + second_dir_name + '/' + file_name)
        return path_list

    @staticmethod
    def get_all_first_son_file_path(dir):
        """获取下级文档数据"""
        path_list = []
        for file_name in os.listdir(dir):
            if file_name.endswith('.txt'):
                path_list.append(dir + '/' + file_name)
        return path_list

    @staticmethod
    def train_test_split_according_file(dir, text_ratio=0.25):
        """
        进行训练集和测试集的划分， 每一个字随机划分以3:1的比例进行划分
        :param text_ratio: 划分比例
        :param path_file_list: 文件列表
        :return: train_path,text_path 对应的文件路径
        todo 后续增加字体，应该实现text_ratio的逻辑
        """
        txt_list_path = []
        random.seed(9527)
        word_style_number = len(os.listdir(dir))
        txt_list_path = CommanTools.get_all_second_file_path(dir)
        # 获得全部的汉字
        word_dict = {}
        for each in txt_list_path:
            word = re.findall(pattern="/([\u4e00-\u9fa5]).txt", string=each)[0]
            if not isinstance(word_dict.get(word), list):
                word_dict[word] = [each]
            else:
                word_dict[word].append(each)
        # 对于每一个汉字，随机选择一个作为text
        train_path, text_path = [], []
        for each_word in word_dict.keys():
            text_number = random.randint(0, 3)
            text_path.append(word_dict[each_word][text_number])
            train_path.extend([word_dict[each_word][i] for i in range(word_style_number) if i != text_number])

        return train_path, text_path

    @staticmethod
    def plot_corr_matrix_hot_map(dataframe):
        """画出df的相关性热力图"""
        cor = dataframe.corr()
        sns.heatmap(cor,
                    annot=True,  # 显示相关系数的数据
                    center=0.5,  # 居中
                    fmt='.2f',  # 只显示两位小数
                    linewidth=0.5,  # 设置每个单元格的距离
                    linecolor='blue',  # 设置间距线的颜色
                    vmin=0, vmax=1,  # 设置数值最小值和最大值
                    xticklabels=True, yticklabels=True,  # 显示x轴和y轴
                    square=True,  # 每个方格都是正方形
                    cbar=True,  # 绘制颜色条
                    cmap='coolwarm_r',  # 设置热力图颜色
                    )

    @staticmethod
    def ana_error_predict_word(y_predict, y_test, type=''):
        # 观察分错的样本
        error_clf_word_list = []
        for i in range(0, len(y_predict)):
            if y_predict[i] != y_test[i]:
                error_clf_word_list.append([y_test[i], y_predict[i]])
        pd.DataFrame(error_clf_word_list, columns=['real', 'predict']).to_csv(
            r'data/mid_data/error_word_ana_' + type + '.csv',
            encoding='utf_8_sig')

    @staticmethod
    def load_filename_by_dir(dir):
        """
        获取dir目录下的所有文件
        :param dir:
        :return:
        """
        txt_list_path = []
        for second_dir_name in os.listdir(dir):
            for file_name in os.listdir(dir + '/' + second_dir_name):
                if file_name.endswith('.txt'):
                    txt_list_path.append(dir + '/' + second_dir_name + '/' + file_name)
        data = []
        for each in txt_list_path:
            with open(each, 'r') as f:
                x = f.read()
                x += re.findall('input_data/.*?/(.*?).txt', each)[0]
                data.append(x)
        return data


import datetime
import os
import random
import re
from functools import wraps

import pandas as pd
import numpy as np
import sklearn.naive_bayes as nb
from naive_bayes import NBClassifier
import matplotlib

from mytools import CommanTools
from knn import KNearestNeighbourClf
from ensemble_clf import EnsembleClassifier,EnsembleClassifierTools


def lgb_one_hot_experiment(data):
    """
    将数据转化为结构化数据 ,其中一个：
    (b'5 1.00', 101, 74, 132, 233)\n(b'235 1.00', 117, 72, 228, 235)\n乏

    micro-precision:  {} 0.001973684210526316
    macro-precision:  {} 0.0013277100905966885
    """

    def code_x(_avg_x):
        if 125 <= _avg_x <= 150:
            return 9
        elif _avg_x > 150:
            return 18
        else:
            return 0

    def code_y(_avg_y):
        if 125 <= _avg_y <= 150:
            return 3
        elif _avg_y > 150:
            return 6
        else:
            return 0

    def code_ares(_part_area):
        if 5000 <= _part_area <= 15000:
            return 1
        elif _part_area > 15000:
            return 2
        else:
            return 0

    df = pd.DataFrame(columns=[list(range(12))])
    for each_word in data:
        _unit_result = re.findall("\(b'(.*?) (.*?)', (.*?), (.*?), (.*?), (.*?)\)", each_word)
        row = pd.DataFrame([0] * 443 * 27).T
        row.columns = [list(range(443 * 27))]
        for each_samll_part in _unit_result:
            _part_num = int(each_samll_part[0])
            _part_prob = float(each_samll_part[1])
            _avg_x = (int(each_samll_part[2]) + int(each_samll_part[4])) / 2
            _avg_y = (int(each_samll_part[3]) + int(each_samll_part[5])) / 2
            _part_area = abs((int(each_samll_part[2]) - int(each_samll_part[4])) *
                             (int(each_samll_part[3]) - int(each_samll_part[5])))
            coding_res = code_x(_avg_x) + code_y(_avg_y) + code_ares(_part_area)
            _pos = _part_num + 443 * coding_res
            row.iloc[0, _pos] += _part_prob
        try:
            row['y'] = re.findall("\n([\u4e00-\u9fa5])", each_word)[0]
            df = pd.concat([df, row], axis=0)
        except Exception as e:
            print(each_word + '解析失败')
    return df


def nb_model_input_experiment():
    """ fit时以部首为单位，predict时以word为单位"""
    # 数据划分
    train_path, text_path = CommanTools.train_test_split_according_file(r'data/input_data')
    x_train, y_train, text_path = CommanTools.read_train_test_data(train_path, text_path)
    # 数据预处理
    x_train, exponent_list = CommanTools.pre_process_model_data(x_train, ['lf_x', 'rb_y'])
    # 初始化模型
    matplotlib.use('TkAgg')
    CommanTools.plot_corr_matrix_hot_map(x_train)
    number_of_class = len(set(y_train))
    class_prior = [1 / number_of_class for i in range(len(set(y_train)))]
    nbc = NBClassifier(clf=nb.CategoricalNB(class_prior=class_prior),
                       gnb=nb.GaussianNB(priors=class_prior),
                       # continuous_columns=['lf_x', 'lf_y', 'rb_x', 'rb_y']
                       continuous_columns=['lf_x', 'rb_y']
                       )
    # 获取条件概率
    nbc.fit(x_train, y_train)
    # 对于每一个汉字，预测所有部首的概率，对乘取最大即为分类。
    y_predict = nbc.predict(text_path, class_prior, exponent_list=exponent_list)
    y_test = [re.findall(r"/([\u4e00-\u9fa5]).txt", each)[0] for each in text_path]
    CommanTools.multi_classification_PR_performance(y_test, y_predict)
    # 观察分错的样本
    error_clf_word_list = []
    for i in range(0, len(y_predict)):
        if y_predict[i] != y_test[i]:
            error_clf_word_list.append([y_test[i], y_predict[i]])
    pd.DataFrame(error_clf_word_list, columns=['real', 'predict']).to_csv(r'data/mid_data/error_word_ana.csv',
                                                                          encoding='utf_8_sig')


def nb_standard_input_experiment():
    """fit时以部首为单位，predict时以word为单位"""
    # 获取训练集
    train_path = CommanTools.get_all_first_son_file_path(r'data/input_data/new_label')
    x_train, y_train = CommanTools.load_fileList_to_dataframe_standard_res(train_path,
                                                                           columns=['code', 'x', 'y', 'w', 'h'])
    # 训练集数据预处理
    x_train, exponent_list = CommanTools.pre_process_data_standard_res(x_train, ['x', 'y', 'w', 'h'])
    # 初始化模型
    # CommanTools.plot_corr_matrix_hot_map(x_train)
    number_of_class = len(set(y_train))
    class_prior = [1 / number_of_class for i in range(len(set(y_train)))]
    nbc = NBClassifier(clf=nb.CategoricalNB(class_prior=class_prior), gnb=nb.GaussianNB(priors=class_prior),
                       continuous_columns=['x', 'y', 'w', 'h'])
    nbc.fit(x_train, y_train)
    # 分字体预测
    for each_dir in os.listdir(r'data/input_data'):
        if each_dir not in ['img_song']:
            continue
        # 获取全部txt文件
        file_list = [r'data/input_data/' + each_dir + '/' + each for each in os.listdir(r'data/input_data/' + each_dir)]
        file_list = [each for each in file_list if each.endswith('.txt')]
        # 以文件为单位进行预测
        y_predict = nbc.predict(file_list, class_prior, exponent_list=exponent_list)
        y_test = [re.findall(r"/([\u4e00-\u9fa5]).txt", each)[0] for each in file_list]
        print(each_dir, 'predict result')
        CommanTools.multi_classification_PR_performance(y_test, y_predict)
        CommanTools.ana_error_predict_word(y_predict, y_test)


def knn_experiment():
    x_train, y_train = KNearestNeighbourClf.load_standard_data()
    for each_dir in os.listdir(r'data/input_data'):
        if each_dir in ['img_song', 'img_li', 'img_ki', 'img_hei']:
            print(datetime.datetime.now(), re.findall('.*_(.*)', each_dir)[0], '体的识别开始')
            x_test, y_test = KNearestNeighbourClf.load_model_data(r'data/input_data' + '/' + each_dir)
            x_test = KNearestNeighbourClf.model_data_convert(x_test)
            knn_clf = KNearestNeighbourClf(KNearestNeighbourClf.distance_func)
            knn_clf.fit(x_train, y_train)
            y_predict = knn_clf.predict(x_test)
            x_test, y_test = KNearestNeighbourClf.load_model_data(r'data/input_data' + '/' + each_dir)
            CommanTools.multi_classification_PR_performance(y_test, y_predict)
            CommanTools.ana_error_predict_word(y_predict, y_test, each_dir)
        else:
            continue


def ensemble_experiment():
    clf = EnsembleClassifier()
    x, y = EnsembleClassifierTools.load_standard_data(r'../data/input_data/new_label')
    clf.fit(x, y)


if __name__ == '__main__':
    knn_experiment()


import numpy as np
import pandas as pd

from mytools import CommanTools


# class NaiveBayesClf(object):
#     """
#     Todo
#     This is a useless wheel.
#     """
#
#     def __init__(self):
#         """生成用于存储概率信息的数据结构,其中conditional_probability_matrix"""
#         self.conditional_probability_matrix = np.matrix()
#         self.feature_probaility_matrix
#
#     def anlysis_x(self, x1, x2):
#         """
#         如果识别为上侧返回0，识别为中间返回9，下侧返回18
#         :param x1:
#         :param x2:
#         :return:
#         """
#         # 容易判断的情况:如果最低位置都高于中点，则为上侧
#         if x2 <= 150:
#             return 0
#         elif x1 >= 150:
#             return 18
#         else:
#             # 如果一个部首位于中线上下的位置差不多，就可以认为他是中间部首
#             if max(150 - x1, x2 - 150) / min(150 - x1, x2 - 150) < self.x_code_rate:
#                 return 9
#             # 上侧占的多，返回0
#             elif 150 - x1 > x2 - 150:
#                 return 0
#             else:
#                 return 18
#
#     def anlysis_area(self, area):
#         """
#         返回部首的大小判断，根据数据分析制定分箱的标准是5000-15000-np.inf
#         :param area:部首的大小
#         :return:
#         """
#         if area <= 5000:
#             return 0
#         elif area >= 15000:
#             return 2
#         else:
#             return 1
#
#     def convert(self, x):
#         """
#         对每个部首进行映射，映射规则如下：
#             前1/9的区域上侧部首，其中前1/3的区域为左侧部首，其中3个区域又代表面积大中小。
#             计算偏移量s，映射后的编号= 映射前的编号 + 430 * (s-1)
#         :param x:
#         :return:一个n*2维的Dataframe，第一维是转化后的编码，第二维是汉字
#         """
#         _middle_data = pd.DataFrame(columns=['new_code', 'word'])
#         for row in x.iterrows():
#             shifting = 0
#             shifting += self.anlysis_x(row[2], row[4])
#             # y与x的解析逻辑完全相同只是返回值缩小3倍
#             shifting += (self.anlysis_x(row[3], row[5]) / 3)
#             area = (row[2] - row[4]) * (row[3] - row[5])
#             shifting += self.anlysis_area(abs(area))
#             new_code = (shifting - 1) * 443 + row[0]
#             row = pd.DataFrame([new_code, row[6]], columns=['new_code', 'word'])
#             _middle_data = pd.concat([_middle_data, row])
#         return _middle_data
#
#     def fit(self, df, x_code_rate=2):
#         """
#         计算并存储有关概率信息
#         :param df: Dataframe(n*m),其中一行如：235,1.00, 117, 72, 228, 235，乏
#         :param x_code_rate:当一个部首相对中线占据更多的一侧/较少的一侧<x_code_rate时，将其识别为中间
#         :return: none
#         """
#         # 编码
#         _middle_data = self.convert(df.iloc[:, :-1])
#         # 计算概率
#
#     def predict(self):
#         pass
#
#     def predict_proba(self):
#         pass

class NBClassifier(object):
    """
    This is a smart use of sklearn.naive_bayes .
    """

    def __init__(self, clf, gnb, continuous_columns=[]):
        """NBClassifier内部由一个clf及一个gnb构成，在Fit和Predcit时，会根据特征的数据类型调用"""
        self.clf = clf
        self.gnb = gnb
        self.continuous_columns = continuous_columns

    def fit(self, x, y):
        self.categorical_columns = set(x.columns).difference(self.continuous_columns)
        self.clf.fit(x[self.categorical_columns], y)
        self.gnb.fit(x[self.continuous_columns], y)

    def predict_proba(self, text_path, class_prior, exponent_list):
        """
        按照文件计算，每个文件返回一个预测结果
        :param text_path: y的文件列表
        :return:与text_path一致顺序的预测结果
        """
        total_proba_list = []
        for each_text in text_path:
            if not each_text.endswith('.txt'):
                continue
            this_word_df = CommanTools._load_one_file_model_res(each_text)
            if len(this_word_df) == 0:
                print(each_text, '空文件不进行预测,直接设为先验概率')
                total_proba_list.append(class_prior)
                continue
            this_word_df = CommanTools.pre_process_model_data(this_word_df, self.continuous_columns, exponent_list)
            try:
                each_part_proba_clf = self.clf.predict_proba(this_word_df[self.categorical_columns])
                each_part_proba_gnb = self.gnb.predict_proba(this_word_df[self.continuous_columns])
            except Exception as e:
                # 如果遇到训练集中没有的部首
                print(e.args, '测试集中出现训练集中没有的数据，该样本设为先验概率', each_text)
                total_proba_list.append(class_prior)
                continue
            total_proba = [1] * len(each_part_proba_clf[0])
            for i in range(len(each_part_proba_clf)):
                # 采用下列计算方式，降低计算误差
                total_proba *= (each_part_proba_clf[i] * each_part_proba_gnb[i] / class_prior)
            total_proba *= class_prior
            total_proba_list.append(total_proba)
        return total_proba_list

    def predict(self, text_path, class_prior, exponent_list=[]):
        """
        返回分类结果
        :param text_path: 测试文件路径列表
        :param class_prior: 先验概率
        :return: 预测出的分类结果
        """
        y_proba = self.predict_proba(text_path, class_prior, exponent_list)
        class_list = list(self.clf.classes_)
        word_result_list = []
        for each_y_proba in y_proba:
            word_result = class_list[list(each_y_proba).index(max(each_y_proba))]
            word_result_list.append(word_result)
        return word_result_list



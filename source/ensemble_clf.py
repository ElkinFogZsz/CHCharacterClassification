import os
import re

import pandas as pd


class EnsembleClassifierTools(object):
    """static method about ensembleClassifier"""

    @staticmethod
    def _load_data_by_reg(dir_path=r'data/input_data/new_label', pattern=''):
        """
        根据正则加载一个文件夹下的所有文件
        :param dir_path: 目录
        :param pattern: x,y
        :return:
        """
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
                    finded_data = re.findall(pattern, word_list[i])
                    finded_data = finded_data[0]
                    x.iloc[len(x) - 1, 5 * i:5 * i + 5] = list(finded_data)
        x.reset_index(inplace=True, drop=True)
        return x, y

    @staticmethod
    def load_standard_data(dir_path=r'data/input_data/new_label'):
        return EnsembleClassifierTools._load_data_by_reg(dir_path, "(.*?) (.*?) (.*?) (.*?) (.*)")

    @staticmethod
    def load_model_data(dir_path=r'data/input_data/new_label'):
        return EnsembleClassifierTools._load_data_by_reg(dir_path, r"\(b'(.*?) .*?', (.*?), (.*?), (.*?), (.*?)\)")

    @staticmethod
    def consume_one_hot_feature(x):
        """
        从标准标注数据中加载one_hot特征
        :param x: 拼接而成的标准输入
        :return: 生成的one-hot数据
        """
        one_hot_x = pd.DataFrame(columns=[i + 1 for i in range(525)])
        for row in x.iterrows():
            one_hot_row = pd.DataFrame([0] * 525).T
            one_hot_row.columns = [i + 1 for i in range(525)]
            for i in range(0, 50, 5):
                if row[1][i] != -1:
                    code = int(row[1][i])
                    one_hot_row.iloc[:, code-1] += 1
            one_hot_x = pd.concat([one_hot_x, one_hot_row], ignore_index=True)
        return one_hot_x

    @staticmethod
    def consume_grid_feature(x):
        """
        从标准标注数据中加载gird_location特征，分成8*8个网格，统计每个网格内部首的个数
        :param x: 拼接而成的标准输入
        :return: gird_location_df,64列,每列取值为0-n,代表这个格子内有部首的个数,行优先。
        """
        grid_location_df = pd.DataFrame(columns=[i for i in range(64)])
        for row in x.iterrows():
            new_row = pd.DataFrame([0] * 64).T
            new_row.columns = [i for i in range(64)]
            for i in range(0, 50, 5):
                if row[1][i] == -1:
                    break
                # 获取一个部首所占据的区域
                for y in range(0, 320, 40):
                    for x in range(0, 320, 40):
                        if EnsembleClassifierTools.has_mixed(
                                [x + 20, y + 20, 20, 20],
                                [row[1][i + 1], row[1][i + 2], row[1][i + 3], row[1][i + 4]]):
                            index = int(x / 40) * 8 + int(y / 40)
                            new_row[index] += 1
            grid_location_df = pd.concat([grid_location_df, new_row], axis=0)
        return grid_location_df

    @staticmethod
    def has_mixed(rectangle1=[], rectangle2=[]):
        """判断两个矩形是否有交集,输入格式是中心点x，y，w，h"""
        x1, x2 = rectangle1[0], float(rectangle2[0])*320
        y1, y2 = rectangle1[1], float(rectangle2[1])*320
        w1, w2 = rectangle1[2], float(rectangle2[2])*320
        h1, h2 = rectangle1[3], float(rectangle2[3])*320
        if abs(x1 - x2) < 0.5 * (h1 + h2) and abs(y1 - y2) < 0.5 * (w1 + w2):
            return True
        else:
            return False


class Controller(object):
    """第一个弱分类器, ensembleClassifier has a controller"""

    @staticmethod
    def choose_clf_number_during_fit(x):
        """
        :param x: dataframe(n*50)，其中一行为 code,x,y,w,h * 10，无则为-1
        :return: list(n*1),如[1,1,2,3……]，每项代表该字的部首数。
        """
        part_num = []
        for row in x.iterrows():
            part_number = 0
            for i in range(0, 50, 5):
                if row[1][i] == -1:
                    part_number = i / 5
                    break
            part_num.append(part_number)
        return part_num


class EnsembleClassifier(object):
    """
    multi model - multi feature classifier.
    """

    def __init__(self):
        """ EnsClf分为10个分类器，第一个分类器负责识别部首信息并调用，剩余九个分类器分别负责部首数1-9的汉字分类 """
        self.part_clf = Controller()
        self.memory = []
        self.clfs = []
        for i in range(9):
            struct_df = pd.DataFrame(columns=[i for i in range(525)])
            grid_location_df = pd.DataFrame(columns=[i for i in range(64)])
            self.clfs.append([struct_df, grid_location_df])

    def fit(self, x, y):
        """
        :param x 结构化信息，5个一组的 code ,x,y,w,h
        :param y 汉字
        :return: 无
        """
        # part_clf 获取该汉字的部首数
        part_number_list = self.part_clf.choose_clf_number_during_fit(x)
        # 生成每个汉字的one_hot特征及位置特征
        one_hot_struct_x = EnsembleClassifierTools.consume_one_hot_feature(x)
        gird_location_x = EnsembleClassifierTools.consume_grid_feature(x)

    def predict(self, x, predict_num=None):
        """

        :param x:输入数据
        :param predict_num: 只启用prdict_num编号的分类器
        :return:
        """
        if predict_num is None:
            predict_num = [1, 2, 3, 4, 5, 6, 7, 8, 9]


if __name__ == '__main__':
    clf = EnsembleClassifier()
    x, y = EnsembleClassifierTools.load_standard_data(r'../data/input_data/new_label')
    clf.fit(x, y)

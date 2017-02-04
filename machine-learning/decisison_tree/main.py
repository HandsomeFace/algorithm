from numpy import *
from math import log
import operator


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calculate_shannon_entropy(data_set):
    '''计算香农熵

    :param data_set: 输入数据集
    :return: 香农熵的值
    '''
    num_entry = len(data_set)
    label_count = {}
    for feather_vec in data_set:
        current_label = feather_vec[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1

    shannon_entropy = 0.0
    for k, v in label_count.items():
        prob = float(v)/num_entry
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def split_data_set(data_set, axis, value):
    '''抽取出数据集中第axis维中值等于value的行，并且该行中去除掉第axis维

    :param data_set: 输入数据集
    :param axis: 维度
    :param value: 数值
    :return:
    '''
    ret_data_set = []
    for feather_vector in data_set:
        if feather_vector[axis] == value:
            reduced_vector = feather_vector[:axis]
            reduced_vector.extend(feather_vector[axis + 1:])
            ret_data_set.append(reduced_vector)
    return ret_data_set


def choose_best_feature_to_split(dataSet):
    '''选择最好的特征

    :param dataSet:
    :return: 最好的特征
    '''
    num_feature = len(dataSet[0]) - 1
#    base_entropy = calculate_shannon_entropy(dataSet)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        #求某一列的所有值
        feature_list = [example[i] for example in dataSet]
        unique_value = set(feature_list)
        new_entropy = 0.0
        for value in unique_value:
            sub_data_set = split_data_set(dataSet, i, value)
            prob = len(sub_data_set) / float(len(dataSet))
            new_entropy += prob * calculate_shannon_entropy(sub_data_set)
#        info_gain = base_entropy - new_entropy
        info_gain = -new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(classList):
    class_count = {}
    for vote in classList:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataSet, labels):
    class_list = [line[-1] for line in dataSet]
    #类别完全相同时停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    #遍历完所有特征时返回出现次数做多的
    if len(dataSet[0]) == 1:
        return majority_cnt(class_list)

    #正常循环
    best_feature = choose_best_feature_to_split(dataSet)
    best_label = labels[best_feature]
    decision_tree = {best_label: {}}
    del(labels[best_feature])

    feature_values = [sample[best_feature] for sample in dataSet]
    unique_feature_value = set(feature_values)
    for feature_value in unique_feature_value:
        sub_labels = labels[:]
        decision_tree[best_label][feature_value] = \
            create_tree(split_data_set(dataSet, best_feature, feature_value), sub_labels)
    return decision_tree


if __name__ == '__main__':
    dataSet, labels = create_data_set()
    print(choose_best_feature_to_split(dataSet))
    print(create_tree(dataSet, labels))

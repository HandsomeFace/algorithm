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


def calculate_gini(data_set):
    ''' 计算dataSet的基尼系数

    :param data_set:
    :return:
    '''
    numEntries = len(data_set)
    labelCounts = {}

    for row in data_set:
        currentLabel = row[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    gini = 1.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key])/numEntries
        gini -= prob * prob
    return gini


def split_discrete_data_set(data_set, axis, value):
    '''对离散变量划分数据集，抽取出数据集中第axis维中值等于value的行，并且该行中去除掉第axis维

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


def split_continuous_data_set(data_set, axis, value, direction):
    '''对连续变量划分数据集

    :param data_set:
    :param axis:
    :param value:
    :param direction: 0，表示大于value；其他表示小于等于value
    :return:
    '''
    ret_data_set = []
    for feather_vector in data_set:
        if direction == 0:
            if feather_vector[axis] > value:
                reduced_vector = feather_vector[:axis]
                reduced_vector.extend(feather_vector[axis + 1:])
                ret_data_set.append(reduced_vector)
        else:
            if feather_vector[axis] <= value:
                reduced_vector = feather_vector[:axis]
                reduced_vector.extend(feather_vector[axis + 1:])
                ret_data_set.append(reduced_vector)
    return ret_data_set


def choose_best_feature_to_split(dataSet):
    ''' ID3选择最好的特征

    :param dataSet:
    :return: 最好的特征
    '''
    num_feature = len(dataSet[0]) - 1
    base_entropy = calculate_shannon_entropy(dataSet)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        #求某一列的所有值
        feature_list = [example[i] for example in dataSet]
        unique_value = set(feature_list)
        new_entropy = 0.0
        for value in unique_value:
            sub_data_set = split_discrete_data_set(dataSet, i, value)
            prob = len(sub_data_set) / float(len(dataSet))
            new_entropy += prob * calculate_shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def choose_best_feature_to_split_C45(dataSet):
    ''' C4.5选择最好的特征

    :param dataSet:
    :return: 最好的特征
    '''
    num_feature = len(dataSet[0]) - 1
    base_entropy = calculate_shannon_entropy(dataSet)
    best_gain_rate = 0.0
    best_feature = -1
    for i in range(num_feature):
        #求某一列的所有值
        feature_list = [example[i] for example in dataSet]
        unique_value = set(feature_list)
        new_entropy = 0.0
        split_info = 1.0    # 防止split_info可能为0，这里采用预先加1的方法
        for value in unique_value:
            sub_data_set = split_discrete_data_set(dataSet, i, value)
            prob = len(sub_data_set) / float(len(dataSet))
            split_info -= prob * log(prob, 2)
            new_entropy += prob * calculate_shannon_entropy(sub_data_set)
        gain_rate = (base_entropy - new_entropy) / split_info
        if gain_rate > best_gain_rate:
            best_gain_rate = gain_rate
            best_feature = i
    return best_feature


def choose_best_feature_to_split_gini(dataSet):
    ''' gini选择最好的特征

    :param dataSet:
    :return: 最好的特征
    '''
    num_feature = len(dataSet[0]) - 1
    base_gini = calculate_gini(dataSet)
    best_gini_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        #求某一列的所有值
        feature_list = [example[i] for example in dataSet]
        unique_value = set(feature_list)
        newGiniIndex = 0.0
        for value in unique_value:
            subDataSet = split_discrete_data_set(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newGiniIndex += prob * calculate_gini(subDataSet)
        gini_gain = base_gini - newGiniIndex
        if gini_gain > best_gini_gain:
            best_gini_gain = gini_gain
            best_feature = i
    return best_feature


def choose_best_feature_to_split_withContinuousFeature(dataSet, labels):
    ''' gini选择最好的特征，会特殊处理连续值的属性

    :param dataSet:
    :param labels:
    :return:
    '''
    num_feature = len(dataSet[0]) - 1
    best_feature = -1
    best_gini_index = inf
    bestSplitDict = {}

    for i in range(num_feature):
        feature_list = [example[i] for example in dataSet]
        unique_value = set(feature_list)
        # 对连续型特征进行处理
        if (type(feature_list[0]).__name__ == "int") or (type(feature_list[0]).__name__ == "float"):
            # 生成n-1个候选划分点
            sortfeatList = sorted(unique_value)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j]+sortfeatList[j+1])/2.0)

            best_split_gini = inf
            bestSplit = -1
            for j in range(len(splitList)):
                value = splitList[j]
                newGiniIndex = 0.0
                subDataSet0 = split_continuous_data_set(dataSet, i, value, 0)
                subDataSet1 = split_continuous_data_set(dataSet, i, value, 1)
                prob0 = len(subDataSet0)/float(len(dataSet))
                newGiniIndex += prob0 * calculate_gini(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newGiniIndex += prob1 * calculate_gini(subDataSet1)
                if newGiniIndex < best_split_gini:
                    best_split_gini = newGiniIndex
                    bestSplit = j

            bestSplitDict[labels[i]] = splitList[bestSplit]
            GiniIndex = best_split_gini
        # 对离散特征处理
        else:
            newGiniIndex = 0.0
            for value in unique_value:
                subDataSet = split_discrete_data_set(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newGiniIndex += prob * calculate_gini(subDataSet)
            GiniIndex = newGiniIndex

        if GiniIndex < best_gini_index:
            best_gini_index = GiniIndex
            best_feature = i

    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理，即是否小于等于bestSplitValue
    if (type(dataSet[0][best_feature]).__name__ == "int") or (type(dataSet[0][best_feature]).__name__ == "float"):
        bestSplitValue = bestSplitDict[labels[best_feature]]
        labels[best_feature] = labels[best_feature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][best_feature] <= bestSplitValue:
                dataSet[i][best_feature] = 1
            else:
                dataSet[i][best_feature] = 0

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
    ''' 递归创建决策树，使用map保存

    :param dataSet: 输入数据集
    :param labels: 各个属性名，（'no surfacing', 'flippers'）表示属性列为是否浮出水面和是否有璞
    :return: 决策树
    '''
    class_list = [line[-1] for line in dataSet]
    #类别完全相同时停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    #遍历完所有特征时返回出现次数做多的
    if len(dataSet[0]) == 1:
        return majority_cnt(class_list)

    #正常循环
    best_feature = choose_best_feature_to_split_withContinuousFeature(dataSet, labels)
    best_label = labels[best_feature]
    decision_tree = {best_label: {}}
    del(labels[best_feature])

    feature_values = [sample[best_feature] for sample in dataSet]
    unique_feature_value = set(feature_values)
    for feature_value in unique_feature_value:
        sub_labels = labels[:]
        decision_tree[best_label][feature_value] = create_tree(split_discrete_data_set(dataSet, best_feature, feature_value), sub_labels)
    return decision_tree


def classify(inputTree, featureLabels, testVec):
    ''' 使用决策树进行分类

    :param inputTree:决策树
    :param featureLabels:
    :param testVec:
    :return:
    '''
    first_lable = list(inputTree.keys())[0]
    second_dict = inputTree[first_lable]
    feature_index = featureLabels.index(first_lable)
    for key in second_dict.keys():
        if testVec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], featureLabels, testVec)
            else:
                class_label = second_dict[key]
    return class_label


if __name__ == '__main__':
    dataSet, labels = create_data_set()
    # print(choose_best_feature_to_split(dataSet))
    print(create_tree(dataSet, labels))

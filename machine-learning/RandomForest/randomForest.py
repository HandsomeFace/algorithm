from numpy import *
from math import log
import operator


def loadDataSet(fileName):
    dataMat = []
    labels = "Cap-shape, Cap-surface, Cap-color, Bruises, Odor, Gill-attachment, Gill-spacing, Gill-size, Gill-color, Stalk-shape, Stalk-root, Stalk-surface-above-ring, Stalk-surface-below-ring, Stalk-color-above-ring, Stalk-color-below-ring, Veil-type, Veil-color, Ring-number, Ring-type, Spore-print-color, Population, Habitat"
    labelMat = labels.strip().split(",")
    totalLines = open(fileName).readlines()
    for line in totalLines:
        lineArr = line.strip().split(",")
        dataMat.append(lineArr)
    return dataMat, labelMat


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
            sub_data_set = split_data_set(dataSet, i, value)
            prob = len(sub_data_set) / float(len(dataSet))
            new_entropy += prob * calculate_shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
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
    best_feature = choose_best_feature_to_split(dataSet)
    best_label = labels[best_feature]
    decision_tree = {best_label: {}}
    del(labels[best_feature])

    feature_values = [sample[best_feature] for sample in dataSet]
    unique_feature_value = set(feature_values)
    for feature_value in unique_feature_value:
        sub_labels = labels[:]
        decision_tree[best_label][feature_value] = create_tree(split_data_set(dataSet, best_feature, feature_value), sub_labels)
    return decision_tree


def createForest(dataSet, featureLabels, numTress):
    ''' 创建随机森林

    :param dataSet: 数据集
    :param featureLabels: 每个特征名称的列表
    :param numTress: 随机森林中决策树的个数
    :return:
    '''
    row, col = shape(dataSet)
    forest = []
    for i in range(numTress):
        # 随机抽取一半的数据
        samples = unique(random.randint(0, row, size=(row+1)//2))
        sampleDatas = array(dataSet)[samples.tolist()]

        # 从抽取的数据中随机抽取一半的特征
        features = unique(random.randint(0, col-1, col//2))
        features_list = features.tolist()
        feature_labels = array(featureLabels)[features_list]
        features_list.append(-1)
        sample_feature_data = sampleDatas[:, features_list]

        # 根据抽样的数据构建决策树
        tree = create_tree(sample_feature_data.tolist(), feature_labels.tolist())
        forest.append(tree)
    return forest


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
    feature_value = testVec[feature_index]

    #特殊处理，如果树中找不到该属性的值，则从树中随机选取一个,或者直接返回错误
    value = second_dict.get(feature_value, "false")
    if value == "false":
        return "cannot"
        # values = second_dict.values();
        # r = random.randint(0, len(values))
        # value = values[r]

    if type(value).__name__ == 'dict':
        class_label = classify(value, featureLabels, testVec)
    else:
        class_label = value
    return class_label


def predict(forest, featureLabels, test_data):
    ''' 使用随机森林forest对测试数据test_data进行预测

    :param forest:
    :param featureLabels:
    :param test_data:
    :return:
    '''
    predict_labels = []
    # 对每一个样本
    for row in test_data:
        dic = {}
        # 使用所有的决策树做预测，返回预测最多的作为结果
        for tree in forest:
            predict_label = classify(tree, featureLabels, row)
            if predict_label == "cannot":
                continue
            dic[predict_label] = dic.get(predict_label, 0) + 1
        l = sorted(dic.items(), key=lambda ele:ele[1], reverse=True)
        if len(l) == 0:
            label = row[-1]
            predict_labels.append(label)
        else:
            predict_labels.append(l[0][0])

    return array(predict_labels)


def accuracy(test_data, predictedLables):
    row = shape(test_data)[0]
    numCorrect = 0
    for i in range(row):
        if predictedLables[i] == array(test_data)[i,-1]:
            numCorrect += 1
    print(numCorrect * 1.0 / row)


if __name__ == '__main__':
    dataSet, labels = loadDataSet('mushroom.dat')
    trainData = dataSet[:4000]
    testData = dataSet[4000:]
    forest = createForest(trainData, labels, 10)
    predictedlabels = predict(forest, labels, testData)
    accuracy(testData, predictedlabels)


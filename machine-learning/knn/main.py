from numpy import *
import operator
from os import listdir


def create_data_set():
    """创建数据集和lable

    :rtype: object
    """
    groups = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels


def classify(input, dataSet, labels, k):
    """knn分类算法

    input：要预测的数据向量
    dataSet：样本数据集
    labels：样本数据集对应的label
    k：取最近的k个数据来判断待预测的数据的类别
    :rtype: int，待预测的数据向量的类别
    """

    dataSetSize = dataSet.shape[0]
    diffMat = tile(input, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    distanceIndicies = distances.argsort()

    classCount = {}

    for i in range(k):
        voteLabel = labels[distanceIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    sample = open(filename)
    allLines = sample.readlines()
    numberOfLines = len(allLines)
    retMatrix = zeros((numberOfLines, 3))
    retLables = []

    index = 0
    for line in allLines:
        line = line.strip()
        listFromLine = line.split('\t')
        retMatrix[index,:] = listFromLine[0:3]
        retLables.append(int(listFromLine[-1]))
        index += 1

    return retMatrix,retLables


def nomalization(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normalizeDatsSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalizeDatsSet = dataSet - tile(minVals, (m, 1))
    normalizeDatsSet = normalizeDatsSet / tile(ranges, (m, 1))
    return normalizeDatsSet, ranges, minVals


def datingClassTest():
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = nomalization(datingDataMat)
    m = normMat.shape[0]
    testNum = int(m * 0.1)
    errorCount = 0.0

    for i in range(testNum):
        classfyResult = classify(normMat[i, :], normMat[testNum:m, :], datingLabels[testNum:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classfyResult, datingLabels[i]))
        if(classfyResult != datingLabels[i]):
            errorCount += 1.0

    print("the total error rate is: %f" % (errorCount / float(testNum)))
    print(errorCount)


def classify_person():
    resultList = ['not at all', 'in small doses', 'in largs dosed']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = nomalization(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classfyResult = classify((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this perpon: ", resultList[classfyResult - 1])


def image2vector(filename):
    file = open(filename)
    return_vector = zeros((1, 1024))

    for i in range(32):
        line = file.readline().strip()
        for j in range(32):
            return_vector[0, 32*i+j] = int(line[j])

    return return_vector


def handwritingClassTest(dirname):
    trainingLabels = []
    trainingFileList = listdir(dirname)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        file_name = trainingFileList[i]
        fileStr = file_name.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        trainingLabels.append(classNumStr)
        trainingMat[i, :] = image2vector(dirname + '/' + file_name)
        print(i)



if __name__ == '__main__':
    handwritingClassTest('trainingDigits')


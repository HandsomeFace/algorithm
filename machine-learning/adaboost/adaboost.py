from numpy import *


def loadSimpleData():
    dataMat = matrix([
        [1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshold, inequation):
    """单层决策树（决策树桩）分类函数，在所有可能得树桩值上进行迭代来得到具有最小加权错误率的单层决策树

    :param dataMatrix: m X n维数据矩阵
    :param dimen: 要检验的维度，1,2...n
    :param threshold: 阈值
    :param inequation: 比较操作 ”lt“ or ”gt“
    :return: 列向量，dataMatrix中第dimen维度的数据，满足a inequation threshold的设为-1，不满足的为1
            即树的左分支为-1，右分支为1
    """
    retArray = ones([shape(dataMatrix)[0], 1])
    if inequation == 'lt':
        retArray[dataMatrix[:, dimen] <= threshold] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshold] = -1.0
    return retArray     #列向量


def buildStump(dataArr, classLabels, D):
    """构建一层决策树桩

    :param dataArr: m X n维数据矩阵
    :param classLabels: dataArr对应的类别
    :param D:权重，m维列向量
    :return:决策树桩
    """
    dataMat = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMat)
    numSteps = 10.0; bestStump = {}; bestClassErrorList = mat(zeros((m, 1)))
    minError = inf

    for i in range(n):
        rangeMin = dataMat[:,i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            threshod = rangeMin + float(j) * stepSize
            for ineq in ['lt', 'gt']:
                predictVal = stumpClassify(dataMat, i, threshod, ineq)
                errArr = mat(ones((m,1)))
                errArr[predictVal == labelMat] = 0
                weightedError = D.T * errArr
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshod, ineq, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassErrorList = predictVal.copy()
                    bestStump['dim'] = i
                    bestStump['threshod'] = threshod
                    bestStump['ineq'] = ineq

    return bestStump,minError,bestClassErrorList


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    """adaboosting训练决策树桩

    :param dataArr: m X n维数据矩阵
    :param classLabels: dataArr对应的类别
    :param numIt: 最多迭代次数
    :return: weakClassArr-决策树桩弱分类器数组；aggClassEst-弱分类器聚合的分类错误
    """
    labelMat = mat(classLabels).T
    weakClassArr = []
    m,n = shape(dataArr)
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, minError, bestClassEst = buildStump(dataArr, classLabels, D)
        #print("D:", D.T)
        #print("classEst: ", bestClassEst.T)
        alpha = float(0.5*log((1-minError)/max(minError,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = multiply(-1 * alpha * labelMat, bestClassEst)
        D = multiply(D, exp(expon))
        D = D/sum(D)
        aggClassEst += alpha * bestClassEst
        #print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != labelMat, ones((m,1)))
        #print("aggErrors error: ", aggErrors)
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break

    return weakClassArr,aggClassEst


def adaClassfy(dataToClassfy, classifierArr):
    dataMat = mat(dataToClassfy)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat,classifierArr[i]['dim'],classifierArr[i]['threshod'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)


if __name__ == '__main__':
    # dataMat, classLabels = loadSimpleData()
    # D = mat(ones((5,1))/5)
    # bestStump, minError, bestClassEst = buildStump(dataMat,classLabels,D)
    # classifierArray, _ = adaBoostTrainDS(dataMat,classLabels,9)
    # adaClassfy([0,0], classifierArray)
    dataMat, classLabels = loadDataSet('horseColicTraining2.txt')
    classifierArray, _ = adaBoostTrainDS(dataMat, classLabels,10)
    testDataMat, testClassLabels = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassfy(testDataMat, classifierArray)
    errArr = ones((67,1))
    print(errArr[prediction10 != mat(testClassLabels).T].sum())

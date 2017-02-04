from numpy import *



def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return mat(dataMat)


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1


def regLeaf(dataSet):
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * X.T * Y
    return ws,X,Y


def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


def chooseBestSplit(dataSet, leafType, errType, cond=(1,4)):
    tolS = cond[0]
    tolN = cond[1]

    # dataSet中都属于同一类别，直接返回
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)

    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A[0]):
            mat0,mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue;
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)

    mat0,mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)

    return bestIndex, bestValue


def createTree(dataSet, leafType, errType, cond=(1,4)):
    feature, value = chooseBestSplit(dataSet, leafType, errType, cond)
    if feature == None:
        return value
    retTree = {}
    retTree['spInd'] = feature
    retTree['spVal'] = value
    lSet,rSet = binSplitDataSet(dataSet, feature, value)
    retTree['left'] = createTree(lSet, leafType, errType, cond)
    retTree['right'] = createTree(rSet, leafType, errType, cond)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0


def prune(tree,testData):
    if shape(testData)[0] == 0:
        return getMean(tree)

    if (isTree(tree['right'])) or (isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

    if (isTree(tree['left'])):
        tree['left'] = prune(tree['left'], lSet)
    if (isTree(tree['right'])):
        tree['right'] = prune(tree['right'], rSet)

    if (not isTree(tree['right'])) and (not isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inData):
    return float(model)


def modelTreeEval(model, inData):
    n = shape(inData)[1]
    X = mat(ones((1, n+1)))
    X[:,1:n+1] = inData
    return float(X * model)


def treeForecast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForecast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForecast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForecast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForecast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    # dataMat = loadDataSet('ex2.txt')
    # tree = createTree(dataMat, regLeaf, regErr, cond=(0,1))
    # print(tree)
    # testData = loadDataSet('ex2test.txt')
    # prune(tree, testData)
    # dataMat = loadDataSet('exp2.txt')
    # tree = createTree(dataMat, modelLeaf, modelErr, (1, 10))
    # print(tree)
    trainMat = loadDataSet('bikeSpeedVsIq_train.txt')
    testMat = loadDataSet('bikeSpeedVsIq_test.txt')
    myTree = createTree(trainMat, regLeaf, regErr, (1,20))
    yHat = createForecast(myTree, testMat[:,0])
    print(corrcoef(testMat[:,1], yHat, rowvar=0)[0,1])
    myTree1 = createTree(trainMat, modelLeaf, modelErr, (1, 20))
    yHat1 = createForecast(myTree1, testMat[:, 0], modelTreeEval)
    print(corrcoef(testMat[:, 1], yHat1, rowvar=0)[0, 1])


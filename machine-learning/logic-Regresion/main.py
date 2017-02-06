from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    totalLines = open('testSet.txt').readlines()
    for line in totalLines:
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn,labelMatIn):
    dataMat = matrix(dataMatIn)
    labelMat = matrix(labelMatIn).transpose()
    m,n = shape(dataMat)
    alpha = 0.001;maxIters = 500
    weights = ones((n,1))
    for k in range(maxIters):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha*dataMat.transpose()*error
    return weights


def plotBestFit(wei):
    #weights = wei.getA()
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);
            ycord1.append(dataArr[i,2]);
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2]);

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red',marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatIn,labelMatIn):
    m,n = shape(dataMatIn)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatIn[i] * weights))
        error = labelMatIn[i] - h
        for j in range(len(weights)):
            weights[j] = weights[j] + dataMatIn[i][j] * alpha * error
    return weights


def stocGradAscent1(dataMatIn,labelMatIn,numIter=150):
    m,n = shape(dataMatIn)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatIn[randIndex] * weights))
            error = labelMatIn[randIndex] - h
            for k in range(n):
                weights[k] = weights[k] + dataMatIn[randIndex][k] * alpha * error
            del(dataIndex[randIndex])
    return weights


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscent1(dataMat,labelMat)
    plotBestFit(weights)



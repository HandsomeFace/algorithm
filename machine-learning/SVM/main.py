from numpy import *
from time import sleep


def loadDataSet(fileName):
    dataMat = [];labelMat = []
    fileHandle = open(fileName)
    for line in fileHandle.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMat,labelMat


def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj >  H:
        aj = H
    if aj < L:
        aj = L
    return aj


class optStruct:
    def __init__(self, dataMatIn, lableIn, c, toler):
        self.X = dataMatIn
        self.labelMat = lableIn
        self.C = c
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = matrix(zeros((self.m, 1)))
        self.b = 0
        self.eCache = matrix(zeros((self.m, 2)))


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1;maxDeltaE = 0;Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerLoop(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy();alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if L == H:
            print("L==H")
            return 0

        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:
            print("eta >= 0")
            return 0

        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)

        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0

        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)

        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (oS.alphas[i] > 0 and oS.alphas[i] < oS.C): oS.b = b1
        elif (oS.alphas[j] > 0 and oS.alphas[j] < oS.C): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def smoProcess(dataMatIn,labelMatIn,C,toler,maxIter,kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(labelMatIn).transpose(), C, toler)
    iter = 0
    entirSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entirSet)):
        alphaPairsChanged = 0
        if entirSet:
            for i in range(oS.m):
                alphaPairsChanged += innerLoop(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLoop(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        if entirSet:
            entirSet = False
        elif (alphaPairsChanged == 0):
            entirSet = True
        print("iteration number: %d"%iter)

    return oS.b,oS.alphas


def calcWs(alphas, dataArr,classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i,:].T)
    return w


def plotBestFit(dataMat,labelMat,weights,b):
    import matplotlib.pyplot as plt
#    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    x = array(arange(-2.0, 10.0, 0.1))
    y = ((-weights[0] * x - b) / weights[1]).transpose()
    ax.plot(x, y)
    y1 = ((1 - weights[0] * x - b) / weights[1]).transpose()
    ax.plot(x, y1)
    y2 = ((-1 - weights[0] * x - b) / weights[1]).transpose()
    ax.plot(x, y2)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


if __name__ == '__main__':
    dataArr,labelArr = loadDataSet('testSet.txt')
    b,alphas = smoProcess(dataArr,labelArr,0.6,0.0001,40)
    ws = calcWs(alphas,dataArr,labelArr)
    plotBestFit(dataArr, labelArr,ws,b)
    print(mat(dataArr)[0]*mat(ws) + b)

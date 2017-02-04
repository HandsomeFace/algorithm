from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataList = []; labelList = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataList.append(lineArr)
        labelList.append(float(curLine[-1]))
    return dataList,labelList


def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * xMat.T * yMat
    return ws


def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * weights * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * xMat.T * weights * yMat
    return testPoint * ws


def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()


def ridgeRegres(xMat,yMat,lamda=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1])*lamda
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * xMat.T * yMat
    return ws


def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    numTests = 30
    wsMat = zeros((numTests,shape(xMat)[1]))
    for i in range(numTests):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wsMat[i,:] = ws.T
    return wsMat


def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for symbol in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*symbol
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T

    return returnMat


def dotPresent(xArr,yArr,ws):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat[:,0].flatten().A[0])
    # ax.plot(xMat[:,1], xMat * ws)
    strInd = xMat.copy()[:,1].argsort(0)
    xSort = xMat[strInd][:,0,:]
    ax.plot(xSort[:, 1], lwlrTest(xArr, xArr, yArr, 0.01)[strInd])
    plt.show()


from time import sleep
import json
import urllib.request
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(1)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


if __name__ == '__main__':
    # xArr, yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr,yArr)
    # dotPresent(xArr,yArr,ws)
    # yPre = lwlrTest(xArr, xArr, yArr, 0.003)
    # print(corrcoef(yArr,yPre[:,0].flatten().A[0]))
    # print(yArr[0])
    # print(lwlr(xArr[0], xArr, yArr, 0.0001))
    # xArr, yArr = loadDataSet('abalone.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ridgeWeights = ridgeTest(xArr,yArr)
    # stageWiseWeights = stageWise(xArr, yArr, 0.005, 1000)
    # ax.plot(stageWiseWeights)
    # plt.show()
    lgX = []; lgY = []
    setDataCollect(lgX, lgY)
    print(lgX)




from numpy import *
from collections import defaultdict
from sklearn.cluster import DBSCAN


def loadDataSet(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return dataSet


def distance_euclid(vecA, vecB):
    return sqrt(sum(power((vecA - vecB),2)))


class cluster:
    def __init__(self, id1, points1):
        self.id = id1
        self.points = points1


def mydbscan(dataMat, eps, minPts):
    m,n = shape(dataMat)

    #建立邻近队列
    surroundPoints = defaultdict(list)
    for i in range(m):
        total = 0
        for j in range(i+1,m):
            dist = distance_euclid(dataMat[i], dataMat[j])
            if dist <= eps:
                surroundPoints[i].append(j)
                surroundPoints[j].append(i)

    clusters = []
    visitLabel = zeros(m)
    noises = []

    for i in range(m):
        if visitLabel[i] == 1:
            continue
        visitLabel[i] = 1
        neighborPts = surroundPoints[i]
        if len(neighborPts) < minPts:
            noises.append(i)
        else:
            nextCluster = cluster(len(clusters), [])
            clusters.append(nextCluster)
            nextCluster.points.append(i)
            for k in neighborPts:
                if visitLabel[k] == 0:
                    visitLabel[k] = 1
                    neighborPts1 = surroundPoints[k]
                    if len(neighborPts1) >= minPts:
                        neighborPts = list(set(neighborPts).union(set(neighborPts1)))
                inOtherCluster = False
                for j in range(len(clusters)):
                    if k in clusters[j].points:
                        inOtherCluster = True
                        break
                if not inOtherCluster:
                    nextCluster.points.append(k)

    return clusters,noises


def plotPartition(dataMat, clusters, nois):
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    m = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    for i in nois:
        xcord1.append(dataArr[i, 0])
        ycord1.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='yellow', marker='s')

    colors = ['blue','red','green','black']

    for i in range(len(clusters)):
        pointsIdx = clusters[i].points
        xcord2 = []; ycord2 = []
        for j in pointsIdx:
            xcord2.append(dataMat[j,0])
            ycord2.append(dataMat[j,1])
        ax.scatter(xcord2, ycord2, s=30, c=colors[i%len(colors)])


    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


def plotPartition1(dataMat, labelMat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(xcord1, ycord1, s=30, c='yellow', marker='s')

    colors = ['blue','red','green','black']
    color_noise = 'yellow'

    i = 0
    for lable,points in labelMat.items():
        xcord2 = []; ycord2 = []
        if lable == -1:
            color = color_noise
        else:
            color = colors[i%len(colors)]
        i += 1
        for j in points:
            xcord2.append(dataMat[j,0])
            ycord2.append(dataMat[j,1])
        ax.scatter(xcord2, ycord2, s=30, c=color)


    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


def descan_sklearn(dataMat):
    clusters = DBSCAN(eps=1.0, min_samples=8).fit(dataMat)
    labels = clusters.labels_
    retMat = defaultdict(list)
    for i in range(len(labels)):
        retMat[labels[i]].append(i)
    plotPartition1(dataMat, retMat)


if __name__ == '__main__':
    dataSet = loadDataSet('testSet.txt')
    dataMat = mat(dataSet)
    clusters,noises = mydbscan(dataMat, 1, 8)
    print(len(clusters))
    plotPartition(dataMat,clusters,noises)
    # descan_sklearn(dataMat)


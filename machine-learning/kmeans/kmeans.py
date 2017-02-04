from numpy import *


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


def distance_pearson(vecA, vecB):
    ''' 计算两个向量之间的皮尔逊距离

    :param vecA:
    :param vecB:
    :return:
    '''
    # vecA和vecB行向量都是矩阵

    if isinstance(vecA, matrix):
        vecA1 = vecA.tolist()[0]
        vecB1 = vecB.tolist()[0]
    else:
        vecA1 = vecA
        vecB1 = vecB
    if len(vecA1) != len(vecB1):
        return None
    N = len(vecA1)
    sumA = sum(vecA1)
    sumB = sum(vecB1)

    sumAB = sum([vecA1[i]*vecB1[i] for i in range(N)])
    sumASquare = sum([power(vecA1[i], 2) for i in range(N)])
    sumBSquare = sum([power(vecB1[i], 2) for i in range(N)])

    numerator = sumAB - (sumA * sumB)/float(N)
    denominator = sqrt((sumASquare-power(sumA,2)/float(N))*(sumBSquare-power(sumB,2)/float(N)))

    if denominator == 0:
        return None

    return numerator/denominator


def randCentroid(dataMat, k):
    n = shape(dataMat)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataMat[:,j])
        maxJ = max(dataMat[:,j])
        rangeJ = maxJ - minJ
        centroids[:,j] = mat(minJ + random.rand(k,1) * rangeJ)
    return centroids


def kMeans(dataMat, k, distance=distance_euclid, createCent=randCentroid):
    '''

    :param dataMat:
    :param k:簇的数量
    :param distance:计算距离的函数
    :param createCent:
    :return: centroids：k个质心点
             clusterAssement：每个点到其质心点的euclid距离
    '''
    m = shape(dataMat)[0]
    centroids = createCent(dataMat, k)
    clusterAssement = mat(zeros((m,2)))

    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distance(centroids[j,:], dataMat[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssement[i,0] != minIndex:
                clusterChanged = True
            clusterAssement[i,:] = minIndex,minDist**2
        # print(centroids)

        #k-means中位点选取采用簇中所有点的平均值
        for cent in range(k):
            pointsInCluster = dataMat[nonzero(clusterAssement[:,0] == cent)[0]]
            centroids[cent,:] = mean(pointsInCluster, axis=0)
    return centroids,clusterAssement


def kMedoids(dataMat, k, distance=distance_euclid, createCent=randCentroid):
    '''

    :param dataMat:
    :param k:簇的数量
    :param distance:计算距离的函数
    :param createCent:
    :return: centroids：k个质心点
             clusterAssement：每个点到其质心点的euclid距离
    '''
    m = shape(dataMat)[0]
    centroids = createCent(dataMat, k)
    clusterAssement = mat(zeros((m,2)))

    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distance(centroids[j,:], dataMat[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssement[i,0] != minIndex:
                clusterChanged = True
            clusterAssement[i,:] = minIndex,minDist**2
        print(centroids)

        #k-medoids中位点选取采用簇中到其他所有点距离最小的点
        for cent in range(k):
            pointsInCluster = dataMat[nonzero(clusterAssement[:,0] == cent)[0]]     #取出当前簇中的所有点
            innerm = shape(pointsInCluster)[0]
            bestPointIndex = -1; lowestSSE = inf;
            for inneri in range(innerm):    #依次选择各个点
                sumInner = 0.0
                sumInner = sum(distance(pointsInCluster[innerj],pointsInCluster[inneri]) for innerj in range(innerm))
                if sumInner < lowestSSE:
                    lowestSSE = sumInner;
                    bestPointIndex = inneri
            centroids[cent,:] = pointsInCluster[bestPointIndex,:]
    return centroids,clusterAssement


def biKmeans(dataMat,k,distance=distance_euclid):
    m = shape(dataMat)[0]
    clusterAssement = mat(zeros((m,2)))
    centroid0 = mean(dataMat, axis = 0).tolist()[0]
    centroids = [centroid0]
    #初始簇
    for j in range(m):
        clusterAssement[j,1] = distance(dataMat[j], mat(centroid0)) ** 2

    while len(centroids) < k:
        lowestSSE = inf
        for i in range(len(centroids)):
            pointsInCurrCluster = dataMat[nonzero(clusterAssement[:,0] == i)[0]]        #取出类别为i的所有数据点
            centroidMat,splitClustAssement = kMeans(pointsInCurrCluster,2,distance)     #在数据点上做2means划分
            sseSplit = sum(splitClustAssement[:,1])
            sseNotSplit = sum(clusterAssement[nonzero(clusterAssement[:,0] != i)[0],1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if ((sseSplit + sseNotSplit) < lowestSSE):
                bestCentroidToSplit = i
                bestNewCentroids = centroidMat
                bestClusterAssement = splitClustAssement.copy()
                lowestSSE = sseSplit + sseNotSplit

        bestClusterAssement[nonzero(bestClusterAssement[:,0] == 1)[0], 0] = len(centroids)
        bestClusterAssement[nonzero(bestClusterAssement[:,0] == 0)[0], 0] = bestCentroidToSplit
        print('the bestCentToSplit is: ', bestCentroidToSplit)
        print('the len of bestClustAss is: ', len(bestClusterAssement))
        centroids[bestCentroidToSplit] = bestNewCentroids[0:,].tolist()[0]
        centroids.append(bestNewCentroids[1:,].tolist()[0])
        clusterAssement[nonzero(clusterAssement[:,0] == bestCentroidToSplit)[0],:] = bestClusterAssement
    return mat(centroids),clusterAssement


def plotPartition(dataMat, centroids):
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    m = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(m):
        xcord1.append(dataArr[i, 0])
        ycord1.append(dataArr[i, 1])
    for i in range(shape(centroids)[0]):
        xcord2.append(centroids[i, 0])
        ycord2.append(centroids[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='black')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


########################层次聚类#############################
class ClusterNode:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id


def hierachyCluster(dataMat, k, distanceEvalue=distance_pearson):
    clusterNodes = [ClusterNode(dataMat[i], id=i) for i in range(shape(dataMat)[0])]
    # 缓存距离
    distanceCache = {}
    flag = None
    currentCluster = -1

    while (len(clusterNodes) > k):
        minDistance = inf
        clusterLen = len(clusterNodes)
        for i in range(clusterLen - 1):
            for j in range(i+1, clusterLen):
                if distanceCache.get((clusterNodes[i].id, clusterNodes[j].id)) == None:
                    distanceCache[(clusterNodes[i].id, clusterNodes[j].id)] = distanceEvalue(clusterNodes[i].vec, clusterNodes[j].vec)
                d = distanceCache[(clusterNodes[i].id, clusterNodes[j].id)]
                if d < minDistance:
                    minDistance = d
                    flag = (i,j)

        index1,index2 = flag
        # 使用均值距离，每个合并的簇

        newVec = [(clusterNodes[index1].vec[i]+clusterNodes[index2].vec[i])/2.0 for i in range(len(clusterNodes[index1].vec))]
        newNode = ClusterNode(newVec[0], left=clusterNodes[index1], right=clusterNodes[index2], distance=minDistance, id=currentCluster)
        currentCluster -= 1

        del(clusterNodes[index2])
        del(clusterNodes[index1])
        clusterNodes.append(newNode)
    return clusterNodes
    ########################层次聚类#############################




if __name__ == '__main__':
    # dataSet = loadDataSet('testSet.txt')
    # dataMat = mat(dataSet)
    # centroids,b = kMedoids(dataMat, 4)
    # plotPartition(dataMat, centroids)
    # dataSet = loadDataSet('testSet2.txt')
    # dataMat = mat(dataSet)
    # centroids,_ = biKmeans(dataMat, 3)
    # plotPartition(dataMat, centroids)
    dataSet = loadDataSet('testSet.txt')
    dataMat = mat(dataSet)
    trees = hierachyCluster(dataMat, 4, distanceEvalue=distance_euclid)
    centroids=zeros((4,2))
    centroids[0, :] = trees[0].vec
    centroids[1, :] = trees[1].vec
    centroids[2, :] = trees[2].vec
    centroids[3, :] = trees[3].vec
    plotPartition(dataMat, centroids)
    print(trees)
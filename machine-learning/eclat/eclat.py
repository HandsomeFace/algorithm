

def loadSimpDat():
    simpDat = {}
    simpDat['T1'] = ['r', 'z', 'h', 'j', 'p']
    simpDat['T2'] = ['z', 'y', 'x', 'w', 'v', 'u', 't', 's']
    simpDat['T3'] = ['z']
    simpDat['T4'] = ['r', 'x', 'n', 'o', 's']
    simpDat['T5'] = ['y', 'r', 'x', 'z', 'q', 't', 'p']
    simpDat['T6'] = ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    return simpDat


def createVerticalSet(inDataSet):
    ''' 创建倒排的数据集，即item-TID集。

    :param inDataSet:
    :return: item-TID数据集
    '''
    verticalSet = {}
    for tid,trans in inDataSet.items():
        for item in trans:
            if not item in verticalSet.keys():
                verticalSet[item] = set([])
            verticalSet[item].add(tid)

    return verticalSet


def eclatInner(prefix, verticalSet, freqItems, minSupport=1):
    ''' 真正的eclat算法实现

    :param prefix: 函数内部使用，用于递归
    :param verticalSet: 倒排的数据集
    :param freqItems: 用于保存频繁项集的列表
    :param minSupport: 最小支持度
    :return:
    '''
    #先过滤掉不满足最小支持度的项
    innerVeticalSet = [item for item in verticalSet.items() if len(item[1]) >= minSupport]
    while innerVeticalSet:
        item,tidSet = innerVeticalSet.pop()
        freqItems.append(prefix+[item])

        newKSet = {}
        for i,tidSetI in innerVeticalSet:
            joinSet = tidSet & tidSetI
            if len(joinSet) >= minSupport:
                newKSet[i] = joinSet
        if len(newKSet) > 0:
            eclatInner(prefix+[item], newKSet, freqItems, minSupport)


def eclat(dataSet, minSupport=1):
    ''' 堆外的eclat算法接口

    :param dataSet: 倒排的数据集
    :param minSupport: 最小支持度
    :return: 频繁项集
    '''
    freqItems = []
    prefix = []
    eclatInner(prefix, dataSet, freqItems, minSupport)
    return freqItems


if __name__ == '__main__':
    dataSet = loadSimpDat()
    verticalSet = createVerticalSet(dataSet)
    print(verticalSet)
    print(eclat(verticalSet, 3))

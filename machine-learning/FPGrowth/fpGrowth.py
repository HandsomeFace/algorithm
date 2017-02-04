from collections import OrderedDict


class treeNode:
    """ FP-tree节点

    """
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}


    def inc(self, numOccur):
        self.count += numOccur


    def display(self, ind=1):
        print('  '*ind, self.name, '  ', self.count)
        for child in self.children.values():
            child.display(ind+1)


def createTree(dataSet, minSuppport=1):
    ''' 构造fp-tree

    :param dataSet:
    :param minSuppport: 最小支持度--某个频繁项集在所有事务中出现的频数的最小值，这里是频数不是频率
    :return: fp-tree，和头链表
    '''
    ###################构造头链表和1维频繁集########################
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):
        if headerTable[k] < minSuppport:    #频数小于最小支持度的删除
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())   #1维频繁项集
    if (len(freqItemSet) == 0):
        return None,None
    for k in headerTable.keys():
        headerTable[k] = [headerTable[k], None]
    ###################构造头链表和1维频繁集########################

    root = treeNode('Null', 1, None)

    #从每条记录中抽取出现在1维频繁项集中的元素，然后按照频数从大到小排序，最后更新fp-tree
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            updateTree(orderedItems, root, headerTable, count)
    print('headerTable: ', headerTable)
    return root,headerTable


def updateTree(items, inTree, headerTable, count):
    ''' 递归更新fp树和头链表

    :param items:
    :param inTree:
    :param headerTable:
    :param count:
    :return: 无
    '''
    if items[0] in inTree.children.keys():
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePattern, treeNode):
    conditionPatterns = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            conditionPatterns[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return conditionPatterns


def mineTree(inTree, headerTable, minSupport, prefix, freqItemList):
    ''' 从fp-tree中挖掘频繁项集

    :param inTree: fp-tree
    :param headerTable: 头链表
    :param minSupport:
    :param prefix: 用于递归调用时产生频繁项
    :param freqItemList: 保存结果list
    :return:
    '''
    #首先，headerTable中从频数最小的开始，排序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    #遍历头链表的每个节点元素
    for basePat in bigL:
        newFreqSet = prefix.copy()
        newFreqSet.append(basePat)
        # newFreqSet.add(basePat)
        freqItemList.insert(0, newFreqSet)
        CPB = findPrefixPath(basePat, headerTable[basePat][1])      #条件模式基
        conditionTree, conditionHeadTab = createTree(CPB, minSupport)
        if conditionHeadTab != None:
            print('conditional tree for: ', newFreqSet)
            conditionTree.display(1)
            mineTree(conditionTree, conditionHeadTab, minSupport, newFreqSet, freqItemList)


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


if __name__ == '__main__':
    simpleData = loadSimpDat()
    simpleData = createInitSet(simpleData)
    print(simpleData)
    fpTree, headerTable = createTree(simpleData, 3)
    fpTree.display()
    print(headerTable)
    freqItems = []
    mineTree(fpTree, headerTable, 3, list([]), freqItems)
    print(freqItems)
    # parseData = [line.split() for line in open('kosarak.dat').readlines()]
    # parseData = createInitSet(parseData)
    # fpTree, headerTable = createTree(parseData, 100000)
    # freqItems = []
    # mineTree(fpTree, headerTable, 100000, list([]), freqItems)
    # print(freqItems)

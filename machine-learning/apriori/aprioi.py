from numpy import *


def loadDataSet():
    ''' 生成测试数据集

    '''
    return [
        [1,3,4],[2,3,5],[1,2,3,5],[2,5]
    ]


def createC1(dataSet):
    ''' 获取1-item候选集

    :param dataSet:
    :return: 1-item候选集
    '''
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(dataSet, Ck, minSupport):
    ''' 获取k-item频繁项集

    :param dataSet:
    :param Ck: 候选集
    :param minSupport:
    :return: 频繁项集
    '''
    ssCnt = {}
    for tid in dataSet:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = float(len(dataSet))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList,supportData


def genCk(Lk, K):
    '''从k-item频繁项集中生成k+1-item候选集，原则：保证前K-2项相同，

    :param Lk: 频繁项集
    :param K:
    :return: 候选集
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:K-2]
            L2 = list(Lk[j])[:K-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                Lkij = Lk[i] | Lk[j]
                if has_infrequent_subset(Lkij, Lk):
                    retList.append(Lkij)
    return retList


def has_infrequent_subset(candidate, itemsets):
    ''' 假设candidate是k候选项，判断candidate的k-1子集中是否有不存在k-1候选项中的项

    :param candidate:
    :param itemsets:
    :return:
    '''
    tmpSet = set(candidate)
    for item in tmpSet:
        cc = tmpSet.copy()
        cc.remove(item)
        if not frozenset(cc) in itemsets:
            return False
    return True


def apriori(dataSet, minSupport):
    '''寻找频繁项集

    :param dataSet:
    :param minSupport: 最小支持度--某个频繁项集在所有事务中出现的频率的最小值
    :return:所有维度的频繁项集，以及支持度
    '''
    C1 = createC1(dataSet)
    L1,supportData = scanD(dataSet, C1, minSupport)

    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = genCk(L[k-2], k)
        Lk, supK = scanD(dataSet, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData


def generateRules(L, supportData, minConfidence=0.7):
    bigRuleList = []
    for i in range(1, len(L)):  #only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                ruleFromConsequence(freqSet, H1, supportData, bigRuleList, minConfidence)
            else:
                calcConfidence(freqSet, H1, supportData, bigRuleList, minConfidence)


def calcConfidence(freqSet, H, supportData, brl, minConfidence):
    prunedH = []
    for conseq in H:
        confidence = supportData[freqSet]/supportData[freqSet-conseq]
        if confidence > minConfidence:
            print(freqSet-conseq,'--->',conseq,'confidence:',confidence)
            brl.append((freqSet-conseq, conseq, confidence))
            prunedH.append(conseq)
    return prunedH


def ruleFromConsequence(freqSet, H, supportData, brl, minConfidence):
    m = len(H[0])
    if (len(freqSet) > (m+1)):
        calcConfidence(freqSet, H, supportData, brl, minConfidence)
        Hmp1 = genCk(H, m+1)
        Hmp1 = calcConfidence(freqSet, Hmp1, supportData, brl, minConfidence)
        if (len(Hmp1) > 1):
            ruleFromConsequence(freqSet, Hmp1, supportData, brl, minConfidence)


if __name__ == '__main__':
    dataArr = loadDataSet()
    C1 = createC1(dataArr)
    L1,sup = scanD(dataArr, C1, 0.5)
    C2 = genCk(L1, 2)
    L, supportData = apriori(dataArr, 0.5)
    print(L)
    print(supportData)
    generateRules(L, supportData, 0.5)



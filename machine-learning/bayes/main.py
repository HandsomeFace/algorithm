from numpy import *


def loadDataSet():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0,1,0,1,0,1]
    return posting_list,classVec


def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    ''' 根据词汇表vocabList，把inputSet向量化转换

    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!", word)
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!", word)
    return returnVec


def trainNB1(trainMatrix, trainCategory):
    ''' 训练naive bayes模型。

    :param trainMatrix:
    :param trainCategory:
    :return:
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #p(1)的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    #计算p(w|ci),eg.：p(w0|c0).....p(wn|c0),p(w0|c1).....p(wn|c1)
    # p0Num = zeros(numWords); p1Num = zeros(numWords)
    # p0Denom = 0.0; p1Denom = 0.0
    # 由于要计算所有概率p(wj|ci)的乘积，如果其中一项为0，则结果为0,这里做修正
    # 将所有单词的出现次数初始化为1，分母初始化为2
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = p1Num/p1Denom
    p0Vec = p0Num/p0Denom

    # 由于概率都比较小，为了防止乘积下溢，改为log
    p1Vec = log(p1Num / p1Denom)
    p0Vec = log(p0Num / p0Denom)
    # 返回概率：p(w|0),p(w|1),p(1)
    return p0Vec,p1Vec,pAbusive


def classfyNB(vec2Classify, p0Vec, p1Vec, pAbusive):
    p1 = sum(vec2Classify * p1Vec) + log(pAbusive)
    p0 = sum(vec2Classify * p0Vec) + log(1-pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0


def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,23):
        print(i)
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = list(range(44)); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB1
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB1(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classfyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))


if __name__ == '__main__':
    # textSet, classes = loadDataSet()
    # vocabs = createVocabList(textSet)
    # dataSet = []
    # for row in textSet:
    #     vectorRow = bagOfWords2Vec(vocabs, row)
    #     dataSet.append(vectorRow)
    # p0Vec, p1Vec, pAbusive = trainNB1(dataSet, classes)
    # testEntry = ['love', 'my', 'dalmation']
    # thisDoc = array(bagOfWords2Vec(vocabs, testEntry))
    # print(classfyNB(thisDoc, p0Vec, p1Vec, pAbusive))
    # testEntry = ['stupid', 'garbage']
    # thisDoc = array(bagOfWords2Vec(vocabs, testEntry))
    # print(classfyNB(thisDoc, p0Vec, p1Vec, pAbusive))
    spamTest()



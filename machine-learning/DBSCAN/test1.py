import pylab as pl
from collections import defaultdict,Counter
from numpy import *
points=[]
fr = open('testSet.txt')
for line in fr.readlines():
    curLine = line.strip().split('\t')
    fltLine = list(map(float, curLine))
    points.append(fltLine)
# 计算每个数据点相邻的数据点，邻域定义为以该点为中心以边长为2*EPs的网格
Eps = 1
surroundPoints = defaultdict(list)
for idx1,point1 in enumerate(points):
  for idx2,point2 in enumerate(points):
    if (idx1 < idx2):
      if(sqrt(sum(power((mat(point1) - mat(point2)),2))))<=Eps:
        surroundPoints[idx1].append(idx2)
        surroundPoints[idx2].append(idx1)
# 定义邻域内相邻的数据点的个数大于4的为核心点
MinPts = 8
corePointIdx = [pointIdx for pointIdx,surPointIdxs in surroundPoints.items() if len(surPointIdxs)>=MinPts]
# 边界点：该点不是核心点，其邻域内包含至少一个核心点
borderPointIdx = []
for pointIdx,surPointIdxs in surroundPoints.items():
  if (pointIdx not in corePointIdx):
    for onesurPointIdx in surPointIdxs:
      if onesurPointIdx in corePointIdx:
        borderPointIdx.append(pointIdx)
        break
# 噪音点既不是边界点也不是核心点
noisePointIdx = [pointIdx for pointIdx in range(len(points)) if pointIdx not in corePointIdx and pointIdx not in borderPointIdx]
corePoint = [points[pointIdx] for pointIdx in corePointIdx]
borderPoint = [points[pointIdx] for pointIdx in borderPointIdx]
noisePoint = [points[pointIdx] for pointIdx in noisePointIdx]

groups = [idx for idx in range(len(points))]
# 各个核心点与其邻域内的所有核心点放在同一个簇中
for pointidx in corePointIdx:
    for oneSurroundIdx in surroundPoints[pointidx]:
        if oneSurroundIdx in corePointIdx:
            groups[oneSurroundIdx] = groups[pointidx]
# 边界点跟其邻域内的某个核心点放在同一个簇中
for pointidx in borderPointIdx:
    for oneSurroundIdx in surroundPoints[pointidx]:
        if oneSurroundIdx in corePointIdx:
            groups[pointidx] = groups[oneSurroundIdx]
            break
# 取簇规模最大的5个簇
wantGroupNum = 4
finalGroup = Counter(groups).most_common(4)
finalGroup = [onecount[0] for onecount in finalGroup]
group1 = [points[idx] for idx in range(len(points)) if groups[idx]==finalGroup[0]]
group2 = [points[idx] for idx in range(len(points)) if groups[idx]==finalGroup[1]]
group3 = [points[idx] for idx in range(len(points)) if groups[idx]==finalGroup[2]]
group4 = [points[idx] for idx in range(len(points)) if groups[idx]==finalGroup[3]]
pl.plot([eachpoint[0] for eachpoint in group1], [eachpoint[1] for eachpoint in group1], 'or')
pl.plot([eachpoint[0] for eachpoint in group2], [eachpoint[1] for eachpoint in group2], 'oy')
pl.plot([eachpoint[0] for eachpoint in group3], [eachpoint[1] for eachpoint in group3], 'og')
pl.plot([eachpoint[0] for eachpoint in group4], [eachpoint[1] for eachpoint in group4], 'or')
# 打印噪音点，黑色
pl.plot([eachpoint[0] for eachpoint in noisePoint], [eachpoint[1] for eachpoint in noisePoint], 'ok')
pl.show()
from matplotlib.pyplot import *
from collections import defaultdict
import random

#function to calculate distance
def dist(p1, p2):
  return ((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)**(0.5)

#randomly generate around 100 cartesian coordinates
all_points=[]

fr = open('testSet.txt')
for line in fr.readlines():
    curLine = line.strip().split('\t')
    fltLine = list(map(float, curLine))
    all_points.append(fltLine)


#take radius = 8 and min. points = 8
E = 1
minPts = 8

#find out the core points
other_points =[]
core_points=[]
plotted_points=[]
for point in all_points:
  point.append(0) # assign initial level 0
  total = 0
  for otherPoint in all_points:
    distance = dist(otherPoint,point)
    if distance<=E:
      total+=1

  if total > minPts:
    core_points.append(point)
    plotted_points.append(point)
  else:
    other_points.append(point)

#find border points
border_points=[]
for core in core_points:
  for other in other_points:
    if dist(core,other)<=E:
      border_points.append(other)
      plotted_points.append(other)


#implement the algorithm
cluster_label=0

for point in core_points:
  if point[2]==0:
    cluster_label+=1
    point[2]=cluster_label

  for point2 in plotted_points:
    distance = dist(point2,point)
    if point2[2] ==0 and distance<=E:
      print (point, point2)
      point2[2] =point[2]


#after the points are asssigned correnponding labels, we group them
cluster_list = defaultdict(lambda: [[],[]])
for point in plotted_points:
  cluster_list[point[2]][0].append(point[0])
  cluster_list[point[2]][1].append(point[1])

markers = ['+','*','.','d','^','v','>','<','p']

#plotting the clusters
i=0
print (cluster_list)
for value in cluster_list:
  cluster= cluster_list[value]
  plot(cluster[0], cluster[1],markers[i])
  i = i%10+1

#plot the noise points as well
noise_points=[]
for point in all_points:
  if not point in core_points and not point in border_points:
    noise_points.append(point)
noisex=[]
noisey=[]
for point in noise_points:
  noisex.append(point[0])
  noisey.append(point[1])
plot(noisex, noisey, "x")

title(str(len(cluster_list))+" clusters created with E ="+str(E)+" Min Points="+str(minPts)+" total points="+str(len(all_points))+" noise Points = "+ str(len(noise_points)))
axis((-6,6,-6,6))
show()
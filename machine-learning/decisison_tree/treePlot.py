import matplotlib.pyplot as plt


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def get_leaf_numbers(tree):
    num = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num += get_leaf_numbers(second_dict[key])
        else:
            num += 1

    return num


def get_tree_deep(tree):
    max_deep = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_deep = 1 + get_tree_deep(second_dict[key])
        else:
            this_deep = 1
        if this_deep > max_deep:
            max_deep = this_deep
    return max_deep


def plot_middle_text(curPoint, parentPoint, txt):
    xMid = (curPoint[0] + parentPoint[0]) / 2.0
    yMid = (curPoint[1] + parentPoint[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txt)


def plot_tree(tree, parentPoint, txt):
    num_leafs = get_leaf_numbers(tree)

    first_str = list(tree.keys())[0]
    curPoint = (plot_tree.xOff + (1.0 + float(num_leafs))/2.0/plot_tree.totalW, plot_tree.yOff)
    plot_middle_text(curPoint, parentPoint, txt)
    plotNode(first_str, curPoint, parentPoint, decisionNode)

    second_dict = tree[first_str]
    plot_tree.yOff -= 1.0/plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], curPoint, str(key))
        else:
            plot_tree.xOff += 1.0/plot_tree.totalW
            plotNode(second_dict[key], (plot_tree.xOff, plot_tree.yOff), curPoint, leafNode)
            plot_middle_text((plot_tree.xOff, plot_tree.yOff), curPoint, str(key))
    plot_tree.yOff += 1.0/plot_tree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=True, **axprops)
    plot_tree.totalW = float(get_leaf_numbers(inTree))
    plot_tree.totalD = float(get_tree_deep(inTree))
    plot_tree.xOff = -0.5/plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(inTree, (0.5, 1.0), '')
    plt.show()


def classify(inputTree, featureLabels, testVec):
    first_str = list(inputTree.keys())[0]
    second_dict = inputTree[first_str]
    feature_index = featureLabels.index(first_str)
    for key in second_dict.keys():
        if testVec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], featureLabels, testVec)
            else:
                class_label = second_dict[key]
    return class_label


def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


if __name__ == '__main__':
    tree = retrieveTree(0)
    createPlot(tree)

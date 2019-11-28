from Tree.ChooseBestFeatureToSplit import chooseBestFeatureToSplit
from Tree.MajorityCnt import majorityCnt
from Tree.SplitDataSet import splitDataSet
def createTree(dataSet, lables):
    classList = [example[-1] for example in  dataSet]
    if classList.count(classList[0] == len(classList)):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat  = chooseBestFeatureToSplit(dataSet)
    print(bestFeat)
    bestFeatLabel = lables[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(lables[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = lables[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
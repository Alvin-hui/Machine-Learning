from math import log
from Tree.CalcShannonEnt import calcshannonent
from Tree.CreateDataSet import creatDataSet
from Tree.SplitDataSet import splitDataSet
from Tree.ChooseBestFeatureToSplit import chooseBestFeatureToSplit
from Tree.CreateTree import createTree
myDat, lables = creatDataSet()


print(myDat)
classlist = [example[-1] for example in myDat]
print(classlist)
#print(lables)
#print(calcshannonent(myDat))
#print(splitDataSet(myDat, 0, 1))

#print(chooseBestFeatureToSplit(myDat))
#myTree = createTree(myDat, lables)
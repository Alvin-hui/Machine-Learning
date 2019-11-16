from math import log
from Tree.CalcShannonEnt import calcshannonent
from Tree.CreateDataSet import creatDataSet
from Tree.SplitDataSet import splitDataSet
from Tree.ChooseBestFeatureToSplit import chooseBestFeatureToSplit

myDat, lables = creatDataSet()


print(myDat)
print(calcshannonent(myDat))
print(splitDataSet(myDat, 0, 1))

print(chooseBestFeatureToSplit(myDat))
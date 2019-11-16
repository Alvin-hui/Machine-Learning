def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVac = featVec[:axis]
            reduceFeatVac.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVac)
    return retDataSet

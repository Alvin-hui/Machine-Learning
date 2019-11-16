from math import log
def calcshannonent(dataSet):
    numEntries = len(dataSet)
    lableCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in lableCounts.keys():
            lableCounts[currentLabel] = 0
        lableCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in lableCounts:
        prob = float(lableCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt
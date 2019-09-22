from KNN import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
group, lables = kNN.createDataSet()

print(kNN.classify0([10, 10], group, lables, 3))
print(group)
print(lables)

datingDataMat, datingLabels = kNN.file2matrix('/Users/albert_king/Desktop/Machine-Learning/KNN/datingTestSet2.txt')

print(datingDataMat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()


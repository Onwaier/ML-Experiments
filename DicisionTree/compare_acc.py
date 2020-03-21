import numpy as np
import matplotlib.pyplot as plt
from math import log
import operator
import random
import operator
import copy
import re
import DicisionTree
from sklearnTree import dicisionTree


plt.rcParams['font.family']=['STFangsong']



x1 = [i for i in range(1, 11)]
y1 = []
x2 = [i for i in range(1, 11)]
y2 = []
x3 = [i for i in range(1, 11)]
y3 = []
x4 = [i for i in range(1, 11)]
y4 = []
x5 = [i for i in range(1, 11)]
y5 = []
x6 = [i for i in range(1, 11)]
y6 = []
x7 = [i for i in range(1, 11)]
y7 = []
x8 = [i for i in range(1, 11)]
y8 = []

x9 = [i for i in range(1, 11)]
y9 = []
x10 = [i for i in range(1, 11)]
y10 = []
x11 = [i for i in range(1, 11)]
y11 = []
x12 = [i for i in range(1, 11)]
y12 = []
x = np.arange(0, 12)

for i in range(10):
    data, labels = DicisionTree.getdataset('./Titanic_dataset.txt')
    random.shuffle(data)
    traindata = data[:425]
    testdata = data[425:]
    myTree = DicisionTree.createTree(traindata, copy.deepcopy(labels), 4)
    y1.append(DicisionTree.test(copy.deepcopy(myTree), testdata, copy.deepcopy(labels)))

    NewTree = DicisionTree.Pruning2(copy.deepcopy(myTree), testdata, copy.deepcopy(labels))
    y2.append(DicisionTree.test(NewTree, testdata, copy.deepcopy(labels)))

    NewTree2 = DicisionTree.PostPruningTree(copy.deepcopy(myTree), traindata, testdata, copy.deepcopy(labels))
    y3.append(DicisionTree.test(NewTree2, testdata, copy.deepcopy(labels)))

    testacc, trainacc, acc = dicisionTree(data, traindata, testdata, labels)
    y4.append(testacc)
    y5.append(DicisionTree.test(copy.deepcopy(myTree), traindata, copy.deepcopy(labels)))
    y6.append(DicisionTree.test(NewTree, traindata, copy.deepcopy(labels)))
    y7.append(DicisionTree.test(NewTree2, traindata, copy.deepcopy(labels)))
    y8.append(trainacc)
    y9.append(DicisionTree.test(copy.deepcopy(myTree), data, copy.deepcopy(labels)))
    y10.append(DicisionTree.test(NewTree, data, copy.deepcopy(labels)))
    y11.append(DicisionTree.test(NewTree2, data, copy.deepcopy(labels)))
    y12.append(acc)



l1 = plt.plot(x1, y1, 'r--', label='未剪枝')
l2 = plt.plot(x2, y2, 'g--', label='后剪枝1(悲观剪枝)')
l3 = plt.plot(x3, y3, 'b--', label='后剪枝2')
l4 = plt.plot(x4, y4, 'm--', label='sklearn')
plt.plot(x1, y1, 'ro-', x2, y2, 'g+-', x3, y3, 'b^-', x4, y4, 'mx-')
# plt.plot(x1, y1, 'ro-', x3, y3, 'b^-')
plt.title('剪枝前后在测试集上的表现')
plt.xlabel('num')
plt.ylabel('acc')
plt.legend()
plt.show()




l1 = plt.plot(x5, y5, 'r--', label='未剪枝')
l2 = plt.plot(x6, y6, 'g--', label='后剪枝1(悲观剪枝)')
l3 = plt.plot(x7, y7, 'b--', label='后剪枝2')
l4 = plt.plot(x8, y8, 'm--', label='sklearn')
plt.plot(x5, y5, 'ro-', x6, y6, 'g+-', x7, y7, 'b^-', x8, y8, 'mx-')
# plt.plot(x1, y1, 'ro-', x3, y3, 'b^-')
plt.title('剪枝前后在训练集上的表现')
plt.xlabel('num')
plt.ylabel('acc')
plt.legend()
plt.show()



l1 = plt.plot(x9, y9, 'r--', label='未剪枝')
l2 = plt.plot(x10, y10, 'g--', label='后剪枝1(悲观剪枝)')
l3 = plt.plot(x11, y11, 'b--', label='后剪枝2')
l4 = plt.plot(x12, y12, 'm--', label='sklearn')
plt.plot(x9, y9, 'ro-', x10, y10, 'g+-', x11, y11, 'b^-', x12, y12, 'mx-')
# plt.plot(x1, y1, 'ro-', x3, y3, 'b^-')
plt.title('剪枝前后在整个数据集上的表现')
plt.xlabel('num')
plt.ylabel('acc')
plt.legend()
plt.show()
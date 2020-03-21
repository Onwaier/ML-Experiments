from math import log
import operator
import random
import operator
import copy
import re
from plot_tree import createPlot
from sklearnTree import dicisionTree

def chooseLeft(data, labels):
    featurenum = len(data[0]) - 1
    baseEntropy = computeShannonEnt(data)
    maxGain = 0.0
    bestFeature = -1
    for i in range(featurenum):
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in data]
        # 创建set集合{}，元素不可重复
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(data, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(data))
            # 根据公式计算经验条件熵
            newEntropy += prob * computeShannonEnt((subDataSet))
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 打印每个特征的信息增益

        # print("特征%s的增益为%.3f" % (labels[i], infoGain))
        # 计算信息增益
        if (infoGain > maxGain):
            # 更新信息增益，找到最大的信息增益
            maxGain = infoGain
            # 记录信息增益最大的特征的索引值
            bestFeature = i
            # 返回信息增益最大特征的索引值
    # print("最大增益特征：", labels[bestFeature])
    return bestFeature


# get the data and label
def getdataset(path):
    f = open(path)
    data = []
    for line in f.readlines():
        lines = line.split(',')
        p = lines[1]
        s = lines[2]
        a = float(lines[3])
        Em = lines[4]
        Sex = lines[5]
        if p == '"1st"':
            pclass = 0
        elif p == '"2nd"':
            pclass = 1
        else:
            pclass = 2
        if s == '0':
            Survived = 0
        else:
            Survived = 1
        if a < 6:
            age = 0
        elif a < 18:
            age = 1
        elif a < 50:
            age = 2
        else:
            age = 3
        if Em == '"Southampton"':
            Embarked = 0
        elif Em == '"Cherbourg"':
            Embarked = 1
        else:
            Embarked = 2
        if Sex == '"female"\n':
            sex = 0
        else:
            sex = 1
        data.append([pclass, age, Embarked, sex, Survived])
    label = ["pclass", "age", "Embarked", "sex"]
    return data, label


# compute ShannonEnt

def computeShannonEnt(dataset):
    # ShannonEnt = 0.0
    datanum = len(dataset)
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataset:
        currentLabel = featVec[-1]  # 提取标签信息
        if currentLabel not in labelCounts.keys():  # 如果标签没有放入统计次数的字典，添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # label计数

    shannonEnt = 0.0  # 经验熵
    # 计算经验熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / datanum  # 选择该标签的概率
        shannonEnt -= prob * log(prob, 2)  # 利用公式计算
    return shannonEnt  # 返回经验熵

# 划分出所有dataSet[k][axis] = value的数据
# 并更新dataSet,删除第axis个特征
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 求classList出现次数最多的类别（0或者1）
def majorityCnt(classList):
    classCount = {}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        # 根据字典的值降序排列
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet, labels, deepth):
    # 取分类标签（是否存活：yes or no）
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同，则停止继续划分
    if deepth == 0:
        return max(set(classList), key=classList.count)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:
        # print(majorityCnt(classList))
        return majorityCnt(classList)
    # 选择最优特征
    bestFeat = chooseLeft(dataSet, labels)
    # 最优特征的标签

    bestFeatLabel = labels[bestFeat]

    # 根据最优特征的标签生成树
    myTree = {bestFeatLabel: {}}
    # 删除已经使用的特征标签
    # del (labels[bestFeat])
    # 得到训练集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
    # 去掉重复的属性值
    uniqueVls = set(featValues)
    # 遍历特征，创建决策树
    feaLabel = []
    for label in labels:
        if label!=bestFeatLabel:
            feaLabel.append(label)
    for value in uniqueVls:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  feaLabel, deepth-1)
    return myTree


# 对一条数据进行分类
def classify(tree, people, labels):
    classLabel = 0
    # 获取决策树节点
    firstStr = next(iter(tree))
    # 下一个字典
    secondDict = tree[firstStr]
    for k in tree.keys():
        featIndex = labels.index(k)
        for key in secondDict.keys():
            if people[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], people,labels)
                else:
                    classLabel = secondDict[key]
        return classLabel

# 决策树在测试数据集上的准度
def test(tree, dataset,labels):
    rightnum = 0
    # print(dataset)
    # print(tree)
    for people in dataset:
        label = people[-1]
        out = classify(tree, people, labels)
        if out == label:
            rightnum += 1
    ap = rightnum / len(dataset)
    return ap

# 测试投票节点正确率
def testingMajor(major, data_test):
    right = 0.0
    for i in range(len(data_test)):
        if major == data_test[i][-1]:
            right += 1
    # print 'major %d' %error
    return float(right) / len(data_test)


# 悲观剪枝


def Pruning2(PreTree, dataSet,labels):
    firstStr = next(iter(PreTree))
    # 下一个字典
    secondDict = PreTree[firstStr]
    # 遍历字典
    # copy一份PreTree，因为需要临时修改
    newTree = copy.deepcopy(PreTree)
    tmp_labels = copy.deepcopy(labels)
    bestFeat = labels.index(firstStr) # 返回下标
    for key in secondDict.keys():
        # 非叶子节点且测试集不为空
        if (PreTree[firstStr][key] != 0) & (PreTree[firstStr][key] != 1) & (len(dataSet) != 0):

            ap = test(newTree, dataSet, tmp_labels)

            # 分别计算节点替换成0或1的准度
            newTree[firstStr][key] = 0 #替换成节点0



            newap0 = test(newTree, dataSet, tmp_labels)

            newTree[firstStr][key] = 1 #替换成节点1
            newap1 = test(newTree, dataSet, tmp_labels)

            # print(newap0, newap1, ap)
            tmp_list = [ap, newap0, newap1]
            idx = tmp_list.index(max(tmp_list))
            # 准度提高，替换成叶子节点，否则递归子树
            if idx == 1:
                newTree[firstStr][key] = 0
            elif idx == 2:
                newTree[firstStr][key] = 1
            else:
                new_labels = copy.deepcopy(labels)
                del new_labels[bestFeat]
                newTree[firstStr][key] = Pruning2(PreTree[firstStr][key], splitDataSet(dataSet, bestFeat, key), copy.deepcopy(new_labels))

    return newTree

# 降低误差剪枝
def PostPruningTree(PreTree, traindataSet, testdataSet, labels):
    firstStr = list(PreTree.keys())[0]
    # 下一个字典
    secondDict = PreTree[firstStr]
    # 遍历字典
    bestFeat = labels.index(firstStr)  # 返回下标

    # 获取训练数据集的所有标签
    classList = [example[-1] for example in traindataSet]
    subTraindataSet = []
    subTestdataSet = []
    temp_labels = copy.deepcopy(labels)
    del labels[bestFeat]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 如果不是叶子节点
            # 求按bestFeat ==  key划分的训练数据集与测试数据集
            subTraindataSet = splitDataSet(traindataSet, bestFeat, key)
            subTestdataSet = splitDataSet(testdataSet, bestFeat, key)
            # 测试数据集不为空，递归子树
            if len(subTestdataSet) > 0:
                PreTree[firstStr][key] = PostPruningTree(secondDict[key], subTraindataSet,
                                                     subTestdataSet, copy.deepcopy(labels))
    # 将子树替换叶子节点（majority），若精度提高则替换
    if len(testdataSet) == 0 or test(PreTree, testdataSet, temp_labels) >= testingMajor(majorityCnt(classList), testdataSet):
        return PreTree
    else:
        return majorityCnt(classList)

if __name__ == '__main__':
    data, labels = getdataset('./Titanic_dataset.txt')
    random.shuffle(data)
    traindata = data[:425]
    testdata = data[425:]
    myTree = createTree(traindata, copy.deepcopy(labels), 4)
    print('preTree', myTree)
    preap = test(myTree, testdata, copy.deepcopy(labels))
    print('ap_before_Pruning:', preap)

    # NewTree = Pruning(myTree, traindata, copy.deepcopy(labels))
    # NewTree = PostPruningTree(myTree, traindata, testdata, copy.deepcopy(labels))
    # createPlot(NewTree)
    # print('afterTree', NewTree)
    # afterap = test(NewTree, testdata, copy.deepcopy(labels))

    # print("ap_after_Pruning0:", afterap)

    NewTree = Pruning2(copy.deepcopy(myTree), testdata, copy.deepcopy(labels))
    # NewTree = PostPruningTree(myTree, traindata, testdata, copy.deepcopy(labels))

    print('afterTree', NewTree)
    afterap = test(NewTree, testdata, copy.deepcopy(labels))

    print("ap_after_Pruning1:", afterap)

    # NewTree = Pruning2(myTree, traindata, copy.deepcopy(labels))
    NewTree2 = PostPruningTree(copy.deepcopy(myTree), traindata, testdata, copy.deepcopy(labels))

    print('afterTree', NewTree2)
    afterap = test(NewTree2, testdata, copy.deepcopy(labels))

    print("ap_after_Pruning2:", afterap)

    #dicisionTree(data, label)
    createPlot(myTree)
    createPlot(NewTree)
    createPlot(NewTree2)

    dicisionTree(traindata, testdata, labels)
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import random

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

def dicisionTree(data, traindata,testdata,label):
    dataX = np.array(data)[..., :4]
    datay = np.array(data)[..., 4]
    trainX = np.array(traindata)[..., :4]
    trainy = np.array(traindata)[..., 4]
    testX = np.array(testdata)[..., :4]
    testy = np.array(testdata)[..., 4]
    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')
    #拟合模型
    clf.fit(trainX, trainy)
    score = clf.score(testX, testy)
    with open("ID3.dot", 'w') as f:
        f = tree.export_graphviz(clf, feature_names=label, class_names=['survival','death'],out_file=f)
    # print('test', score)
    # print('train', clf.score(trainX, trainy))
    return clf.score(testX, testy), clf.score(trainX, trainy), clf.score(dataX, datay)

if __name__ == '__main__':
    data,label = getdataset('./Titanic_dataset.txt')
    random.shuffle(data)
    traindata = data[:425]
    testdata = data[425:]
    dicisionTree(data, traindata,testdata, label)
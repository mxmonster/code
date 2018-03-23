# coding=utf-8
'''

时间：2016/03/22

作者：moverzp

功能：ID3决策树的简单实例，根据create_data_set中创建的dataSet数据集创建决策树

'''

from numpy import *  # 一个用python实现的科学计算包

from math import log

import operator


def create_data_set():
    dataSet = [[1, 1, 'yes'],  # 是list

               [1, 1, 'yes'],

               [1, 0, 'no'],

               [0, 1, 'no'],

               [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']

    return dataSet, labels


def cal_Ent(dataSet):  # 根据给定数据集计算熵

    num = len(dataSet)

    labels = {}

    for row in dataSet:  # 统计所有标签的个数

        label = row[-1]

        if label not in labels.keys():
            labels[label] = 0

        labels[label] += 1

    Ent = 0.0

    for key in labels:  # 计算熵

        prob = float(labels[key]) / num

        Ent -= prob * log(prob, 2)

    return Ent


def split_data_set(dataSet, axis, value):  # 按照给定特征划分数据集，返回符合条件的数据

    retDataSet = []

    for row in dataSet:

        if (row[axis]) == value:
            reducedRow = row[:axis]

            reducedRow.extend(row[axis + 1:])

            retDataSet.append(reducedRow)

    return retDataSet


def choose_best_feature(dataSet):  # 选择最佳决策特征

    num = len(dataSet[0]) - 1  # 特征数

    baseEnt = cal_Ent(dataSet)

    bestInfoGain = 0.0

    bestFeature = -1

    for i in range(num):

        featlist = [example[i] for example in dataSet]  # 按列遍历数据集，选取一个特征的所有值

        uniqueVals = set(featlist)  # 一个特征可以取的值

        newEnt = 0.0

        for value in uniqueVals:
            subDataSet = split_data_set(dataSet, i, value)

            prob = len(subDataSet) / float(len(dataSet))

            newEnt += prob * cal_Ent(subDataSet)

        infoGain = baseEnt - newEnt

        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain

            bestFeature = i

    return bestFeature


def majorityCnt(classList):  # 多数表决法则

    classCount = {}

    for vote in classList:  # 统计数目

        if vote not in classCount.keys(): classCount[vote] = 0

        classCount += 1

    sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return classCount[0][0]


def create_tree(dataSet, labels):
    labelsCloned = labels[:]

    classList = [example[-1] for example in dataSet]  # [yes,yes,no,no,no]

    if classList.count(classList[0]) == len(classList):  # 只有一种类别，则停止划分

        return classList[0]

    if len(dataSet[0]) == 1:  # 没有特征，则停止划分

        return majorityCnt(classList)

    bestFeat = choose_best_feature(dataSet)

    bestFeatLabel = labelsCloned[bestFeat]  # 最佳特征的名字

    myTree = {bestFeatLabel: {}}

    del (labelsCloned[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]  # 获取最佳特征的所有属性

    uniqueVals = set(featValues)

    for value in uniqueVals:  # 建立子树

        subLabels = labelsCloned[:]  # 深拷贝，不能改变原始列表的内容，因为每一个子树都要使用

        myTree[bestFeatLabel][value] = create_tree(split_data_set(dataSet, bestFeat, value), subLabels)

    return myTree


def classify(tree, featLabels, testVec):
    firstJudge = tree.keys()[0]

    secondDict = tree[firstJudge]

    featIndex = featLabels.index(firstJudge)

    for key in secondDict:

        if key == testVec[featIndex]:

            if type(secondDict[key]).__name__ == 'dict':

                classLabel = classify(secondDict[key], featLabels, testVec)

            else:

                classLabel = secondDict[key]

    return classLabel


def store_tree(tree, fileName):  # 保存树

    import pickle

    fw = open(fileName, 'w')

    pickle.dump(tree, fw)

    fw.close()


def grab_tree(fileName):  # 读取树

    import pickle

    fr = open(fileName)

    return pickle.load(fr)


dataSet, labels = create_data_set()

tree = create_tree(dataSet, labels)

print "decision tree:\n%s" % tree

print "classify result:\n%s" % classify(tree, labels, [1, 1])

store_tree(tree, "ID3_ex_tree.txt")

import treePlotter

treePlotter.createPlot(tree)






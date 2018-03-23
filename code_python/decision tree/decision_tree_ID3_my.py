# coding=utf-8

from math import log  # 加入log
'''
import datetime是引入整个datetime包，如果使用datetime包中的datetime类,需要加上模块名的限定。
import datetime 
print datetime.datetime.now()
    如果不加模块名限定会出现错误
from datetime import datetime是只引入datetime包里的datetime类,在使用时无需添加模块名的限定。
'''
import operator

import treePlotter


'''
隐形眼镜数据集包含患者的眼睛状况以及医生推荐的隐形眼镜类型，
患者信息有4维，分别表示年龄，视力类型，是否散光，眼睛状况，
隐形眼镜类型有3种，分别是软材质，硬材质和不适合带隐形眼镜。
'''


def file2matrix():  #获取数据集
    file = open("lenses.data.txt")  # 打开数据集
    allLines = file.readlines()  # 获取所有数据
    row = len(allLines)  # 获取数据集行数
    dataSet = []  # lsit 列表
    for line in allLines:
        line = line.strip()   # 移除多余字符
        listFromLine = line.split()  # split()：拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
        dataSet.append(listFromLine) # append加入的依旧是列表,如果要直接添加数据，使用extend
    labels = ['age', 'prescription', 'astigmatic', 'tear rate']  # 年龄，视力类型，是否散光，眼睛状况
    return dataSet, labels


def cal_Ent(dataSet):   # 计算熵
    num = len(dataSet)  #获取数据集有多少条记录
    labels = {}  # 字典 dict 使用键-值（key-value）存储，具有极快的查找速度。
    for row in dataSet:   # 统计隐形眼镜类型不同标签的个数
        label = row[-1]  # 获取隐形眼镜种类类型
        if label not in labels.keys():
            labels[label] = 0  # 如果不在关键字中，加入到dict中
        labels[label] += 1   # 统计不同隐形眼镜的个数
    Ent = 0.0
    for key in labels:
        #  计算熵
        prob = float(labels[key])/num
        Ent -= prob * log(prob, 2)
    return Ent


def split_data_set(dataSet, axis, value):   # 按照给定特征划分数据集，返回第axis个特征的值为value的所有数据
    reDataSet = []
    for row in dataSet:
        if(row[axis]) == value:
            reducedRow = row[:axis]
            reducedRow.extend(row[axis+1:])   # list.extend(sequence) 把一个序列seq的内容添加到列表中
            reDataSet.append(reducedRow)
    return reDataSet  # 返回第axis个特征值为value的数据集


def choose_best_feature(dataSet):  # 选择最佳决策特征
    num = len(dataSet[0])-1  # 获取特征数
    baseEnt = cal_Ent(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(num):
        featlist = [example[i] for example in dataSet]  # 按列遍历数据集，选取一个特征的所有值
        uniqueVals = set(featlist)  # 一个特征可以取的值  set中如果参数有重复，会自动忽略
        newEnt = 0.0
        for value in uniqueVals:
            subDataSet = split_data_set(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEnt += prob * cal_Ent(subDataSet)
        infoGain = baseEnt - newEnt   # 信息增益  ID3算法
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
'''
即使用此子集中出现次数最多的类别作为此节点类别，然后将此节点作为叶子节点。
'''

def majorityCnt(classList):  #多数表决法则 vote
    print classList
    classcount = {}  # dict
    for vote in classList:  # 统计数目
        if vote not in classcount.keys():
            classcount[vote] = 0
        classcount += 1  # sorted排序函数，reverse排序规则 true降序 false升序(默认)
    sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)    # iteritems()将dict返回为迭代器对象类似于items()
    return classcount[0][0]  # 取最多的


'''
决策树一般使用递归的方法生成。
编写递归函数有一个好习惯，就是先考虑结束条件。
生成决策树结束的条件有两个：其一是划分的数据都属于一个类，其二是所有的特征都已经使用了。
在第二种结束情况中，划分的数据有可能不全属于一个类，这个时候需要根据多数表决准则确定这个子数据集的分类。
在非结束的条件下，首先选择出信息增益最大的特征，然后根据其分类。
分类开始时，记录分类的特征到决策树中，然后在特征标签集中删除该特征，表示已经使用过该特征。
根据选中的特征将数据集分为若干个子数据集，然后将子数据集作为参数递归创建决策树，
最终生成一棵完整的决策树。
'''


def create_tree(dataSet, labels):   # 生成决策树
    labelsCloned = labels[:]
    classList = [example[-1] for example in dataSet]  # [yes,yes,no,no,no]获取隐形眼镜类型列
    if classList.count(classList[0]) == len(classList):  # 只有一种类别，则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 没有特征，则停止划分  所有的特征都使用了
        return majorityCnt(classList)
    bestFeat = choose_best_feature(dataSet)  # 最佳特征的序号
    bestFeatLabel = labelsCloned[bestFeat]  # 最佳特征的名字
    myTree = {bestFeatLabel: {}}
    del (labelsCloned[bestFeat])  # 删除最佳特征
    featValues = [example[bestFeat] for example in dataSet]  # 获取最佳特征的所有属性 list
    uniqueVals = set(featValues)  #去重复属性
    for value in uniqueVals:  # 建立子树
        subLabels = labelsCloned[:]  # 深拷贝，不能改变原始列表的内容，因为每一个子树都要使用
        myTree[bestFeatLabel][value] = create_tree(split_data_set(dataSet, bestFeat, value), subLabels)
    return myTree


'''
使用决策树对输入进行分类的函数也是一个递归函数。
分类函数需要三个参数：决策树，特征列表，待分类数据。
特征列表是联系决策树和待分类数据的桥梁，决策树的特征通过特征列表获得其索引，
再通过索引访问待分类数据中该特征的值。
'''


def classify(tree, featLabels, testVec):
    firstJudge = tree.keys()[0]
    secondDict = tree[firstJudge]
    featIndex = featLabels.index(firstJudge) #获得特征索引
    for key in secondDict: #进入对应的分类集合
        if key == testVec[featIndex]: #按特征分类
            if type(secondDict[key]).__name__ == 'dict': #如果分类结果是一个字典，则说明还要继续分类
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: #分类结果不是字典，则分类结束
                classLabel = secondDict[key]
    return classLabel


'''
在机器学习中，我们常常需要把训练好的模型存储起来，这样在进行决策时直接将模型读出，
而不需要重新训练模型，这样就大大节约了时间。Python提供的pickle模块就很好地解决了这个问题，
它可以序列化对象并保存到磁盘中，并在需要的时候读取出来，任何对象都可以执行序列化操作。
'''


def store_tree(tree, fileName):   # 保存树存入到文件中
    import pickle
    fw = open(fileName, 'w')
    pickle.dump(tree, fw)  # j将对象存入已经打开的file中
    fw.close()


def grab_tree(fileName): # 读取树
    import pickle
    fr = open(fileName)
    return pickle.load(fr)


dataSet, labels = file2matrix()    # 加载原数据
print str(dataSet)
tree = create_tree(dataSet, labels)
print "decision tree:\n%s" % tree

store_tree(tree,'ID3_lenses_tree.txt')
grab_tree('ID3_lenses_tree.txt')
treePlotter.createPlot(tree)  # 决策树的可视化










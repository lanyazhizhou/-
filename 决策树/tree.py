# -*- coding: UTF-8 -*-
"""
作者：蓝亚之舟
时间：2018.6.29
博客地址：https://blog.csdn.net/yuangan1529/article/details/80872141
"""
import operator
from math import log  #计算信息熵要用到log函数

"""
函数0：创造数据集
好吧，这个函数又是最简单的一个数据集，下面还有一个数据集 
第二个数据集还好看一些
"""
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def createDataSet1():    # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发','声音']  #两个特征
    return dataSet,labels

"""
函数1：计算信息熵
一个样本信息熵越小表示样本纯度越高

"""
def calcShannonEnt(dataSet):   #这里是指香农熵，其实就是我们常说的信息熵

    #第一步：准备工作
    numEntries = len(dataSet)  #求数据集里面的总样本数
    labelCounts = {}           #建立词典用来盛放样本标签，结构为：(标签：数量)

    #第二步：将数据集导入字典中
    for featVec in dataSet:
        currentLabel = featVec[-1]    #这里的-1表示列表中最后一个元素，也就是标签
        if currentLabel not in labelCounts.keys():  #将标签添加到标签列表中
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1   #标签数量加1


    #第三步：计算数据集的信息熵，主要是公式
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt       #最后返回的是一个值



"""
函数3：分割数据集
就是将数据集dataSet按照axis这一特征来划分，如果样本的axis特征值为value
就将这个样本的属性除去axis，然后添加到返回样本中
dataSet相当于所有西瓜样本，axis相当于西瓜色泽、敲声、根蒂三个特征中的一个
比如axis是色泽，则value可以是青绿、黄绿、青色中的一个
splitDataSet（西瓜，色泽，青绿），返回的是当前样本中色泽是青绿色的所有西瓜
"""
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:   #对所有样本进行选择，如果axis属性为value就选中
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            #上面去掉了axis属性，因为划分得到的样本都是该属性

            retDataSet.append(reducedFeatVec)
    return retDataSet  #返回的样本：去掉了axis属性，同时其axis全部为value


"""
函数4：选择最优划分属性
这也就是我们一开始说的决策树难点：如何选择划分属性
这里采用的是信息增益，信息增强越大则代表用这个属性划分越好
详细请参考信息增益公式，下面就是参照信息增益公式来写的
注意：最后返回的是一个数字，这个数字表示第几个数据特征
"""
def chooseBestFeatureToSplit(dataSet):

    #第一步：准备工作
    numFeatures = len(dataSet[0]) - 1  #数据特征数量，-1是除去标签那一列
    baseEntropy = calcShannonEnt(dataSet)  #计算数据集总的信息熵
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):   #针对每个数据特征都要进行操作

        # 第二步：计算信息增益=总信息熵-权重*每个属性信息熵
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy

        #第三步：找出所有信息增益中最大的，其对应特征即为所求
        if(infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature  #返回最大信息增益的数据特征（这是一个数字）

"""
函数5：多票表决叶子节点的类别标签
当所有属性特征都是用完了以后，这个节点就是叶子节点了，但是并不是说这个节点是叶子节点
它包含的样本就都属于同一个类别，相反有时候会有很多类别，这时候我们要决定选择哪个类别作为标签
结论当然是类别数量最多的了
参数classList：叶子节点的所有样本的类别标签（注意有重复的）
"""
def majorityCnt(classList):
    classCount = {}    #用来存储类别和其对应的数量，格式为：{类别：数量}
    for vote in classList:
        if vote not in classCount.keys():  #如果字典中没有这个类，就增加，并且对应数量为0
            classCount[vote] = 0
        classCount[vote] +=1  #对应类别数量加1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #上面这个参数是字典按照其value值进行排序，是降序排列，所以第一个元素就是数量最多的类，也就是返回值

    return sortedClassCount[0][0]   #返回类数量最多的标签


"""
函数6：构建决策树
这个相当于是主函数，将上面所有函数都统筹起来，进行构建
当然，其中数据集和标签集可能要用到函数1（这里要另行调用了）
"""
def createTree(dataSet,labels,featLabels):

    #第一步：决定递归进程的结束和返回
    classList = [example[-1] for example in dataSet]
    #上面这句话要解释一下，for-in语句是将dataSet一行行赋值给example
    #然后又将example最后一项，也就是标签赋值给classList列表
    #说白了，就是将数据集所有标签制作成列表，然后赋值给classList

    #停止迭代1:classList中所有label相同，直接返回该label
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    #停止迭代2:用完了所有特征仍然不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  #根据类数量最多的决定该节点的类标签


    #第二步：决定最优的分叉节点
    bestFeat = chooseBestFeatureToSplit(dataSet)  #这个就是算法第八行
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)

    #第三步：根据上面的分叉节点，进行分组和分叉
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])   #删除分叉节点的属性特征
    featValues = [example[bestFeat] for example in dataSet]  #将最优属性的特征值制成列表以此来分叉
    uniqueVals = set(featValues)   #列表转换为set集合
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels,featLabels)
    return myTree


"""
函数7：通过我们构建的决策树来测试一个新的样本
可以说这个函数是功能函数了，调用上面的决策树开始预测新的样本
参数inputTree：就是上面函数6生成的决策树
参数featLabels:就是函数0生成的labels标签数据集（无重复）
参数testVec：这是一个新的样本特征数据，通过这个，可以预测其标签
"""
def classify(inputTree,featLabels,testVec):
    firstStr = next(iter(inputTree))  #得到当前树的第一个划分属性值
    secondDict = inputTree[firstStr]  #根据上面的划分属性值，得到其对应的划分子树
    featIndex = featLabels.index(firstStr)  #得到划分属性对应的索引
    classLabel = '无'
    for key in secondDict.keys():   #遍历所有划分属性值
        if testVec[featIndex] == key:  #查看测试数据集划分属性对应的值
            if type(secondDict[key]).__name__ == 'dict':  #
                classLabel = classify(secondDict[key],featLabels,testVec)  #递归
            else:
                classLabel = secondDict[key]
    return classLabel


myDat,labels = createDataSet1()
featLabels = []   #因为下面了labels会在调用createTree函数之后，发生改变，所以这里创建一个新的用来盛放
myTree = createTree(myDat,labels,featLabels)
print(myTree)
print(classify(myTree,featLabels,['粗','长']))  #注意这里的粗和长的顺序是一定的，这和决策树有关系
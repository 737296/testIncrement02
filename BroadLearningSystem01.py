# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:09:38 2018

@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time


def show_accuracy(predictLabel, Label): 
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label
    predlabel = predictLabel
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))


def tansig(x):
    return (2/(1+np.exp(-2*x)))-1


def sigmoid(data):
    return 1.0/(1+np.exp(-data))
    

def linear(data):
    return data
    

def tanh(data):
    return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    

def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z

#稀疏自编码器
def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)   
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


"""
基本BLS
参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
"""
def BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3):
    L = 0
    train_x = preprocessing.scale(train_x, axis=1)
    # sklearn.preprocessing.scale(X, axis=0) preprocessing用于更改原始特征向量表示形式以适应后续评估量。
    # X 以此数据为中心缩放
    # axis沿着计算均值和标准差的轴。如果是0，独立的标准化每个特征，如果是1则标准化每个样本（即行）
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    # np.hstack(tup，axis):在水平方向上平铺   axis=0时为垂直拼接；axis=1时为水平拼接
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])
    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    # 对数据集进行增广，在训练集最后一列加1，是为了生成特征节点时，可以直接通过矩阵运算增加偏置项
    train_acc_all = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    for i in range(N2):# 生成映射特征窗口下的映射特征节点
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        # 对每个样本特征进行一次权重随机的卷积和偏置得到新的特征
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        #  preprocessing.MinMaxScaler将相同的缩放应用到每个窗口的特征节点中
        #  1、StandardScaler().fit(X)  scaler里面存的有计算出来的均值和方差，此时X即为FeatureOfEachWindow
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        # scaler1.transform进行转换，只是把训练数据转换成标准的正态分布
        # 2、用scaler中的均值和方差来转换X，使X标准化
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        #  稀疏自编码器，对FeatureOfEachWindow进行稀疏表示，找到一个稀疏矩阵W使得H*W=A，H训练集，A特征FeatureOfEachWindowAfterPreprocess
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        # 生成一个窗口的特征节点
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
#        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        # 对每一个窗口特征节点进行归一化
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow
        print('特征映射层输出形状：',OutputOfFeatureMappingLayer.shape)
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 

    np.save('distOf.npy', distOfMaxAndMin)
    np.save('minOfEac.npy', minOfEachWindow)
    np.save('Beta1O.npy', Beta1OfEachWindow)
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3))-1  #.orth矩阵正交规范化
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T

    np.save('weightOfEn.npy', weightOfEnhanceLayer)
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
#    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)#缩放尺度
    # 对增强节点进行激活
    np.save('outpara.npy', parameterOfShrink)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入——整个网络的输入和权重就训练完成了
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    print('最终输入形状：',InputOfOutputLayer.shape)
    pinvOfInput = pinv(InputOfOutputLayer,c)#pinv求矩阵的伪逆矩阵
    OutputWeight = np.dot(pinvOfInput,train_y)
    print("伪逆后的输入的形状，训练标签形状")
    print(pinvOfInput.shape, train_y.shape)
    OutputWeight = np.transpose(OutputWeight)
    print('输出权重形状：',OutputWeight.shape)
    np.save('outw.npy', OutputWeight)

    time_end=time.time() 
    trainTime = time_end - time_start
    
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    print('训练集的输出：',OutputOfTrain)
    print('训练标签：',train_y)
    print('训练集输出个数')
    print(OutputOfTrain.shape[0])
    for i  in range(OutputOfTrain.shape[0]):
        print('训练集输出的第i行第一列数据：')
        print(OutputOfTrain[i,0])
        OutputOfTrain[i,0]=round(OutputOfTrain[i,0])# 把输出的浮点数转成整数，好与label进行比较相等
        print('训练集输出的浮点数转为整数：')
        print(OutputOfTrain[i,0])
    # OutputOfTrain=OutputOfTrain.astype(int)
    # OutputOfTrain =math.trunc(OutputOfTrain)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc_all[0][0] = trainAcc
    train_time[0][0] = trainTime

    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] =(ymax-ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]-ymin

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])

    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    for i  in range(OutputOfTest.shape[0]):
        # print(OutputOfTest[i,0])
        OutputOfTest[i,0]=round(OutputOfTest[i,0])
        # print(OutputOfTest[i,0])
    time_end=time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accuracy is' ,testAcc * 100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    return test_acc,test_time,train_acc_all,train_time

#%%%%%%%%%%%%%%%%%%%%%%%%    
'''
增加增强节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
l------步数：增量增加强化节点的步数控制
M------步长
'''
def BLS_AddEnhanceNodes(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,L,M):
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0

    train_x = preprocessing.scale(train_x,axis = 1) #处理数据 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])

    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    Beta1OfEachWindow = []
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) 
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append( np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis =0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis =0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
 
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,c)
    OutputWeight = pinvOfInput.dot(train_y)
    OutputWeight=np.transpose(OutputWeight) # transpose转置，把权重矩阵转一下，不然维度对不上
    time_end=time.time() 
    trainTime = time_end - time_start
    
    
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    for i  in range(OutputOfTrain.shape[0]):
        # print(OutputOfTest[i,0])
        OutputOfTrain[i,0]=round(OutputOfTrain[i,0])

    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accuracy is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    # 测试过程
    test_x = preprocessing.scale(test_x, axis=1) 
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
 
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    for i  in range(OutputOfTest.shape[0]):
        # print(OutputOfTest[i,0])
        OutputOfTest[i,0]=round(OutputOfTest[i,0])# 把输出的浮点数转成整数，好与label进行比较相等
    time_end=time.time() #训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc*100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增量增加强化节点
    '''
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start=time.time()
        if N1*N2>= M : 
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M)-1)
        else :
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M).T-1).T
        
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s/np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd*parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer,OutputOfEnhanceLayerAdd])
        
        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T,D)).I.dot(np.dot(D.T,pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        OutputWeightEnd = np.transpose(OutputWeightEnd) # 转置
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e+1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)

        for i in range(OutputOfTrain1.shape[0]):
            # print(OutputOfTest[i,0])
            OutputOfTrain1[i, 0] = round(OutputOfTrain1[i, 0])
        TrainingAccuracy = show_accuracy(OutputOfTrain1,train_y)
        train_acc[0][e+1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        

        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest=np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)

        for i in range(OutputOfTest1.shape[0]):
            # print(OutputOfTest[i,0])
            OutputOfTest1[i, 0] = round(OutputOfTest1[i, 0])
        TestingAcc = show_accuracy(OutputOfTest1,test_y)
        
        Test_time = time.time() - time_start
        test_time[0][e+1] = Test_time
        test_acc[0][e+1] = TestingAcc
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %' )
        
    return test_acc,test_time,train_acc,train_time


'''
增加 映射特征层节点和增强层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
L------步数

M1-----增加映射节点数
M2-----与增加映射节点对应的强化节点数
M3-----新增加的强化节点
'''

lste=[] # 节点增加情况
lstacc=[]
#%%%%%%%%%%%%%%%%
def BLS_AddFeatureEnhanceNodes(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,L,M1,M2,M3):
    
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0

    train_x = preprocessing.scale(train_x,axis = 1) 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])

    Beta1OfEachWindow = list()
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) 
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis = 0) - np.min(outputOfEachWindow,axis = 0))
        minOfEachWindow.append(np.mean(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    #生成强化层
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayerTrain = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayerTrain,c)
    OutputWeight =pinvOfInput.dot(train_y) #全局伪逆
    OutputWeight=np.transpose(OutputWeight) #转置
    time_end=time.time() #训练完成
    trainTime = time_end - time_start
    
    OutputOfTrain = np.dot(InputOfOutputLayerTrain,OutputWeight)
    for i in range(OutputOfTrain.shape[0]):
        # print(OutputOfTest[i,0])
        OutputOfTrain[i, 0] = round(OutputOfTrain[i, 0])
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    test_x = preprocessing.scale(test_x,axis = 1) 
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i] 

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
  
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    for i in range(OutputOfTest.shape[0]):
        # print(OutputOfTest[i,0])
        OutputOfTest[i, 0] = round(OutputOfTest[i, 0])
    time_end=time.time() 
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc*100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增加Mapping 和 强化节点
    '''
    WeightOfNewFeature2 = list()
    WeightOfNewFeature3 = list()
    for e in list(range(L)):
        time_start = time.time()
        random.seed(e+N2+u)
        weightOfNewMapping = 2 * random.random([train_x.shape[1]+1,M1]) - 1
        NewMappingOutput = FeatureOfInputDataWithBias.dot(weightOfNewMapping)

        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(NewMappingOutput)
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(NewMappingOutput)
        betaOfNewWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfNewWindow)
   
        TempOfFeatureOutput = FeatureOfInputDataWithBias.dot(betaOfNewWindow)
        distOfMaxAndMin.append( np.max(TempOfFeatureOutput,axis = 0) - np.min(TempOfFeatureOutput,axis = 0))
        minOfEachWindow.append(np.mean(TempOfFeatureOutput,axis = 0))
        outputOfNewWindow = (TempOfFeatureOutput-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e]

        OutputOfFeatureMappingLayer = np.hstack([OutputOfFeatureMappingLayer,outputOfNewWindow])

        NewInputOfEnhanceLayerWithBias = np.hstack([outputOfNewWindow, 0.1 * np.ones((outputOfNewWindow.shape[0],1))])

        if M1 >= M2:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2])-1)
        else:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2]).T-1).T  
        WeightOfNewFeature2.append(RelateEnhanceWeightOfNewFeatureNodes)
        
        tempOfNewFeatureEhanceNodes = NewInputOfEnhanceLayerWithBias.dot(RelateEnhanceWeightOfNewFeatureNodes)
        
        parameter1 = s/np.max(tempOfNewFeatureEhanceNodes)

        outputOfNewFeatureEhanceNodes = tansig(tempOfNewFeatureEhanceNodes * parameter1)

        if N2*N1+e*M1>=M3:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3) - 1)
        else:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3).T-1).T
        WeightOfNewFeature3.append(weightOfNewEnhanceNodes)

        InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

        tempOfNewEnhanceNodes = InputOfEnhanceLayerWithBias.dot(weightOfNewEnhanceNodes)
        parameter2 = s/np.max(tempOfNewEnhanceNodes)
        OutputOfNewEnhanceNodes = tansig(tempOfNewEnhanceNodes * parameter2)
        OutputOfTotalNewAddNodes = np.hstack([outputOfNewWindow,outputOfNewFeatureEhanceNodes,OutputOfNewEnhanceNodes])
        tempOfInputOfLastLayes = np.hstack([InputOfOutputLayerTrain,OutputOfTotalNewAddNodes])
        D = pinvOfInput.dot(OutputOfTotalNewAddNodes)
        C = OutputOfTotalNewAddNodes - InputOfOutputLayerTrain.dot(D)
        
        if C.all() == 0:
            w = D.shape[1]
            B = (np.eye(w)- D.T.dot(D)).I.dot(D.T.dot(pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeight = pinvOfInput.dot(train_y)
        OutputWeight=np.transpose(OutputWeight)# 转置
        InputOfOutputLayerTrain = tempOfInputOfLastLayes
        
        time_end = time.time()
        Train_time = time_end - time_start
        train_time[0][e+1] = Train_time
        predictLabel = InputOfOutputLayerTrain.dot(OutputWeight)
        for i in range(predictLabel.shape[0]):
            # print(OutputOfTest[i,0])
            predictLabel[i, 0] = round(predictLabel[i, 0])
        TrainingAccuracy = show_accuracy(predictLabel,train_y)
        train_acc[0][e+1] = TrainingAccuracy
        # print('总步长：',L,'目前步长：',e)
        lste.append(e)
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        lstacc.append(TrainingAccuracy)
        # 测试过程
        #先生成新映射窗口输出
        time_start = time.time() 
        WeightOfNewMapping =  Beta1OfEachWindow[N2+e]

        outputOfNewWindowTest = FeatureOfInputDataWithBiasTest.dot(WeightOfNewMapping )
        
        outputOfNewWindowTest = (outputOfNewWindowTest-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e] 
        
        OutputOfFeatureMappingLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,outputOfNewWindowTest])
        
        InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest,0.1*np.ones([OutputOfFeatureMappingLayerTest.shape[0],1])])
        
        NewInputOfEnhanceLayerWithBiasTest = np.hstack([outputOfNewWindowTest,0.1*np.ones([outputOfNewWindowTest.shape[0],1])])

        weightOfRelateNewEnhanceNodes = WeightOfNewFeature2[e]
        
        OutputOfRelateEnhanceNodes = tansig(NewInputOfEnhanceLayerWithBiasTest.dot(weightOfRelateNewEnhanceNodes) * parameter1)
        
        weightOfNewEnhanceNodes = WeightOfNewFeature3[e]
        
        OutputOfNewEnhanceNodes = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfNewEnhanceNodes)*parameter2)
        
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest,outputOfNewWindowTest,OutputOfRelateEnhanceNodes,OutputOfNewEnhanceNodes])
    
        predictLabel = InputOfOutputLayerTest.dot(OutputWeight)

        TestingAccuracy = show_accuracy(predictLabel,test_y)
        time_end = time.time()
        Testing_time= time_end - time_start
        test_time[0][e+1] = Testing_time
        test_acc[0][e+1]=TestingAccuracy
        print('Testing Accuracy is : ', TestingAccuracy * 100, ' %' )
        # print(L, e, e, e, e, e, e)
        # print('总步长：', L, '目前步长：', e)

    x = np.array(lste)
    y=np.array(lstacc)
    plt.title('tilte')
    plt.plot(x, y)
    plt.show()
    return test_acc,test_time,train_acc,train_time
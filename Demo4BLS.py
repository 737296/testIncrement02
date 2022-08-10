# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:35:24 2018


@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.
"""

'''

Installation has been tested with Python 3.5.
Since the package is written in python 3.5, 
python 3.5 with the pip tool must be installed first. 
It uses the following dependencies: numpy(1.16.3), scipy(1.2.1), keras(2.2.0), sklearn(0.20.3)  
You can install these packages first, by the following commands:

pip install numpy
pip install scipy
pip install keras (if use keras data_load())
pip install scikit-learn
'''

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import scipy.io as scio
from BroadLearningSystem01 import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes


''' For Keras dataset_load()'''
import glob
import cv2
import  os

#TRAIN_IMG_PATH表示文件夹的路径，glob.glob匹配所有的符合条件的文件，并将其以list的形式返回
# TRAIN_IMG_PATH1=r'E:\Downloads\newPIC\03increase\normal'
TRAIN_IMG_PATH2=r'C:\Users\mlamp\Documents\03increase\crack'
TRAIN_IMG_PATH1=r'C:\Users\mlamp\Documents\03increase\damage'
# Test_IMG_PATH1=r'E:\Downloads\newPIC\04testIncrease\normal'
Test_IMG_PATH2=r'C:\Users\mlamp\Documents\01basis\basicCrack'
Test_IMG_PATH1=r'C:\Users\mlamp\Documents\01basis\basicDamage'
all_img_paths1 = glob.glob(os.path.join(TRAIN_IMG_PATH1, "*.png"))
all_img_paths2 = glob.glob(os.path.join(TRAIN_IMG_PATH2, "*.png"))
# all_img_paths3 = glob.glob(os.path.join(TRAIN_IMG_PATH3, "*.png"))
all_img_paths01 = glob.glob(os.path.join(Test_IMG_PATH1, "*.png"))
all_img_paths02 = glob.glob(os.path.join(Test_IMG_PATH2, "*.png"))
# all_img_paths03 = glob.glob(os.path.join(Test_IMG_PATH3, "*.png"))

# zeros(shape, dtype=float, order='C')
# 返回：返回来一个给定形状和类型的用0填充的数组；
# 参数：shape:形状
X01=np.zeros(65536) # X01、X02、X03存储图像转为numpy数据时的数组 256*256
X02=np.zeros(65536)
# X03=np.zeros(65536)
X=np.zeros(65536)
print(X01.shape)
imgs1=[]
imgs2=[]
# imgs3=[]
# data_dir=all_img_paths1

#     imgs1.append(img)
target_size = (256, 256)
#将图片缩放到目标大小，不改变原数据，需将得到结果赋值给新变量

for img_path in all_img_paths1:
    img1=Image.open(img_path)
    print("img1:",img1)
    print("img1:",type(img1))
    new_image = img1.resize(target_size)# resize 重新缩放
    # imgs1.append(img1)
    # img_gray = color.rgb2gray(new_image)
    X1 = np.array(new_image) # 创建数组
    X1=X1[:,:,0:1] # 取图片数据的其中一个维度
    X1=X1.reshape(-1) # 无论该数组形状是什么样的，统一变为一维,顺序默认为先行后列
    print(X1.shape,X01.shape)
    X01 = np.vstack((X01, X1)) #  将每个同一标签下的图片（转为一维之后的向量）向量纵向堆叠

Y1= np.ones((X01.shape[0]))
Y1=np.transpose(Y1)
print('imgs1:')
print(imgs1)

for img_path in all_img_paths2:
    img2=Image.open(img_path)
    print("img2:",img2)
    print("img2:",type(img2))
    new_image = img2.resize(target_size)
    # imgs1.append(img1)
    # img_gray = color.rgb2gray(new_image)
    X2 = np.array(new_image)
    X2=X2[:,:,0:1]
    X2=X2.reshape(-1)
    print(X2.shape,X02.shape)
    X02 = np.vstack((X02, X2))

Y2= np.ones((X02.shape[0]))*2
Y2=np.transpose(Y2)
print('裂纹标签Y2:')
print(Y2)
"""
for img_path in all_img_paths3:
    img3=Image.open(img_path)
    print("img3:",img3)
    print("img3:",type(img3))
    new_image = img3.resize(target_size)
    # imgs1.append(img1)
    # img_gray = color.rgb2gray(new_image)
    X3 = np.array(new_image)
    X3=X3[:,:,0:1]
    X3=X3.reshape(-1)
    print(X3.shape,X03.shape)
    X03 = np.vstack((X03, X3))
Y3= np.ones((X03.shape[0]))*3
Y3=np.transpose(Y3)
"""
print('最终X01、X02形状')
print(X01.shape,X02.shape)

X=np.vstack((X01,X02))
#X=np.vstack((X,X03))
print('X01、X02纵向堆积后X形状:')
print(X.shape)

# print(Y1.shape,Y2.shape,Y3.shape)
Y=np.hstack((Y1,Y2))
# Y=np.hstack((Y,Y3))
Y=np.transpose(Y)
print('Y:')
print(Y)
# print('三个训练集Y的形状各:')
# print(Y1.shape,Y2.shape,Y3.shape)
print(Y.shape)

imgs12=[]
imgs22=[]
# imgs32=[]
X012=np.zeros(65536)
X022=np.zeros(65536)
# X032=np.zeros(65536)
X2=np.zeros(65536)
for img_path in all_img_paths01:
    img12=Image.open(img_path)
    print("测试集img1:",img12)
    print("测试集img1类型:",type(img12))
    new_image = img12.resize(target_size)
    # imgs1.append(img1)
    # img_gray = color.rgb2gray(new_image)
    X12 = np.array(new_image)
    X12=X12[:,:,0:1]
    X12=X12.reshape(-1)
    print(X12.shape,X01.shape)
    X012 = np.vstack((X012, X12))


print(X.shape)
Y12= np.ones((X012.shape[0]))
Y12=np.transpose(Y12)
print("测试集img2:")
print(imgs12)

for img_path in all_img_paths02:
    img22=Image.open(img_path)
    print("测试集img22:",img22)
    print("测试集img22:",type(img22))
    new_image = img22.resize(target_size)
    # imgs1.append(img1)
    # img_gray = color.rgb2gray(new_image)
    X22 = np.array(new_image)
    X22=X22[:,:,0:1]
    X22=X22.reshape(-1)
    print(X22.shape,X022.shape)
    X022 = np.vstack((X022, X22))
Y22= np.ones((X022.shape[0]))*2
Y22=np.transpose(Y22)
print("测试集标签Y2:")
print(Y22)
"""
for img_path in all_img_paths03:
    img32=Image.open(img_path)
    print("测试集破损img32:",img32)
    print("测试集破损img32类型:",type(img32))
    new_image = img32.resize(target_size)
    # imgs1.append(img1)
    # img_gray = color.rgb2gray(new_image)
    X32 = np.array(new_image)
    X32=X32[:,:,0:1]
    X32=X32.reshape(-1)
    print(X32.shape,X032.shape,X032.shape,X032.shape,X032.shape,X032.shape,X032.shape,X032.shape,X032.shape)
    X032 = np.vstack((X032, X32))
"""
X2=np.vstack((X012,X022))
# X2=np.vstack((X2,X032))
print(X2.shape,X2.shape,X2.shape,X2.shape,X2.shape,X2.shape,X2.shape,X2.shape)
# Y32= np.ones((X032.shape[0]))*2
# Y32=np.transpose(Y32)
Y2=np.hstack((Y12,Y22))
# Y2=np.hstack((Y2,Y32))
Y2=np.transpose(Y2)
print(Y2.shape,Y2.shape,Y2.shape,Y2.shape,Y2.shape,Y2.shape,Y2.shape,Y2.shape,Y2.shape,Y2.shape,Y2.shape)

# from sklearn import preprocessing
# from numpy import random
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# iris=load_iris()
# X = iris.data
# Y = iris.target
# traindata,testdata,trainlabel,testlabel = train_test_split(X,Y,test_size=0.2,random_state = 2018)

traindata = X/255
trainlabel = Y
testdata = X2/255
testlabel = Y2
N1 = 10  #  # of nodes belong to each window
N2 = 10  #  # of windows -------Feature mapping layer
N3 = 500 #  # of enhancement nodes -----Enhance layer
L = 2    #  # of incremental steps
M1 = 50  #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient

print('-------------------BLS_BASE---------------------------')
BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
print('-------------------BLS_ENHANCE------------------------')
BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
print('-------------------BLS_FEATURE&ENHANCE----------------')
M2 = 50  #  # of adding feature mapping nodes
M3 = 50  #  # of adding enhance nodes
BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)

# -*- coding: utf-8 -*-
# 文件名通配符
import glob
# os操作系统的功能接口函数
import os
# numpy用于矩阵处理
import numpy as np
# PIL 图像处理包
from PIL import Image
# 下两行代码作用：遇到截断的JPEG时，程序就会跳过去，读取另一张图片
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from BroadLearningSystem01 import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes



# TRAIN_IMG_PATH表示文件夹的路径，glob.glob匹配所有的符合条件的文件，并将其以list的形式返回
TRAIN_IMG_PATH2=r'C:\Users\mlamp\Documents\03increase\crack'
TRAIN_IMG_PATH1=r'C:\Users\mlamp\Documents\03increase\damage'
Test_IMG_PATH2=r'C:\Users\mlamp\Documents\01basis\basicCrack'
Test_IMG_PATH1=r'C:\Users\mlamp\Documents\01basis\basicDamage'

all_img_paths1 = glob.glob(os.path.join(TRAIN_IMG_PATH1, "*.png"))
all_img_paths2 = glob.glob(os.path.join(TRAIN_IMG_PATH2, "*.png"))
all_img_paths01 = glob.glob(os.path.join(Test_IMG_PATH1, "*.png"))
all_img_paths02 = glob.glob(os.path.join(Test_IMG_PATH2, "*.png"))


# zeros(shape, dtype=float, order='C')
# 返回：返回来一个给定形状和类型的用0填充的数组；
# 参数：shape
# X01、X02、X03存储图像转为numpy数据时的数组 256*256=65536

X01 = np.zeros(65536)
X02 = np.zeros(65536)
X = np.zeros(65536)
# print(X01.shape)
imgs1 = []
imgs2 = []

# 将图片缩放到目标大小，不改变原数据，需将得到结果赋值给新变量
target_size = (256, 256)

# 遍历训练数据集1 将所有图片添加进一个数组
for img_path in all_img_paths1:
    # 读取图片
    img1 = Image.open(img_path)
    # resize 重新缩放 将图片大小调整到(256,256)
    new_image = img1.resize(target_size)
    # 创建数组用来存放图片
    X1 = np.array(new_image)
    # 3通道的图片取图片数据的其中一个通道，数据是列的结构
    X1 = X1[:,:,0:1]
    # 将列结构转化为行结构，变成行展开的一维数组
    X1 = X1.reshape(-1)
    #  将每个同一标签下的图片（转为一维之后的向量）向量纵向堆叠
    #  直白点就是第一次循环之后X01的结构是[[65535个0],[图片单通道的一维数组]]，循环就是以此叠加
    #  所有训练集1中的数据都放入了X01
    X01 = np.vstack((X01, X1))

# X01.shape返回[多少行一维度数组,每个维度的存储数字的数量]
# X01.shape[0]取出了X01中的一共有多少行一维数组（其中一行是全部0，其余行是真正的图片）
# np.ones创建全部值为1的一维数组（个数由参数决定）

# Y1的作用就是给训练数据集1 打上标签1作为记号
Y1= np.ones((X01.shape[0]))
# 可能是转置？疑问
Y1=np.transpose(Y1)

# 遍历训练数据集2 将所有图片添加进一个数组
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

# Y2的作用就是给训练数据集2 打上标签2作为记号
Y2 = np.ones((X02.shape[0]))*2
Y2 =np.transpose(Y2)

# 将2个数据集的图片一起堆积起来得到最后的X就是所有的 数据集
X = np.vstack((X01, X02))

# 将2个数据集的图片的标签一起堆积起来得到最后的Y就是所有的 标签集
Y = np.hstack((Y1, Y2))
Y = np.transpose(Y)


X012 = np.zeros(65536)
X022 = np.zeros(65536)

for img_path in all_img_paths01:
    img12=Image.open(img_path)
    new_image = img12.resize(target_size)
    X12 = np.array(new_image)
    X12=X12[:,:,0:1]
    X12=X12.reshape(-1)
    X012 = np.vstack((X012, X12))
Y12= np.ones((X012.shape[0]))
Y12=np.transpose(Y12)

for img_path in all_img_paths02:
    img22=Image.open(img_path)
    new_image = img22.resize(target_size)
    X22 = np.array(new_image)
    X22 = X22[:, :, 0:1]
    X22 = X22.reshape(-1)
    X022 = np.vstack((X022, X22))
Y22 = np.ones((X022.shape[0]))*2
Y22 = np.transpose(Y22)

# X2是所有测试集的数据
X2 = np.vstack((X012, X022))


# Y2是所有测试集的标签
Y2 = np.hstack((Y12, Y22))
Y2 = np.transpose(Y2)




traindata = X
trainlabel = Y
testdata = X2
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

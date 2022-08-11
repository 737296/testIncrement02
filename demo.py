import numpy as np
import glob
import os
from PIL import Image
X01 = np.zeros(65536)
X02 = np.zeros(65536)
X = np.zeros(65536)
imgs1 = []

TRAIN_IMG_PATH2=r'C:\Users\mlamp\Documents\03increase\crack'
TRAIN_IMG_PATH1=r'C:\Users\mlamp\Documents\03increase\damage'
Test_IMG_PATH2=r'C:\Users\mlamp\Documents\01basis\basicCrack'
Test_IMG_PATH1=r'C:\Users\mlamp\Documents\01basis\basicDamage'

all_img_paths1 = glob.glob(os.path.join(TRAIN_IMG_PATH1, "*.png"))
all_img_paths2 = glob.glob(os.path.join(TRAIN_IMG_PATH2, "*.png"))
all_img_paths01 = glob.glob(os.path.join(Test_IMG_PATH1, "*.png"))
all_img_paths02 = glob.glob(os.path.join(Test_IMG_PATH2, "*.png"))

target_size = (256, 256)
for img_path in all_img_paths1:
    img1 = Image.open(img_path)
    print("img1:",img1)
    print("img1:",type(img1))
    new_image = img1.resize(target_size)

    # 创建数组
    X1 = np.array(new_image)

    # 取图片数据的其中一个维度
    X1 = X1[:,:,0:1]
    print(X1)
    # 无论该数组形状是什么样的，统一变为一维,顺序默认为先行后列
    X1 = X1.reshape(-1)
    print(X1)

    X01 = np.vstack((X01, X1)) #  将每个同一标签下的图片（转为一维之后的向量）向量纵向堆叠
    print(X01)

    break;
Y1= np.ones((X01.shape[0]))*2
Y1=np.transpose(Y1)

for img_path in all_img_paths2:
    img2 = Image.open(img_path)
    print("img2:", img2)
    print("img2:", type(img2))
    new_image = img2.resize(target_size)
    # imgs1.append(img1)
    # img_gray = color.rgb2gray(new_image)
    X2 = np.array(new_image)
    X2 = X2[:, :, 0:1]
    X2 = X2.reshape(-1)
    print(X2.shape, X02.shape)
    X02 = np.vstack((X02, X2))
    break;
# Y2的作用就是给训练数据集2 打上标签2作为记号
Y2 = np.ones((X02.shape[0])) * 2
Y2 = np.transpose(Y2)

print('最终X01、X02形状')
print(X01.shape,X02.shape)
X=np.vstack((X01,X02))
print('X01、X02纵向堆积后X形状:')
print(X.shape)

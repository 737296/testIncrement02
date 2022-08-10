# JPG2ICO copy.py
# from PyQt5.Qt import *
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QPushButton, QLabel

import sys, os
from PySide6.QtCore import Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel
# from PySide6.QtWebKitWidgets import *
from myjgptoico import *
from sklearn import preprocessing
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

@Slot()
def say_hello():
    print("Button clicked, Hello!")
import numpy as np
import time
import sklearn

def tansig(x):
    return (2/(1+np.exp(-2*x)))-1

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
target_size = (200, 200)

class MyWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        # 继承自己设计的Ui界面
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # 调用自己的函数补充ui界面
        # self.initUi()
        self.setWindowTitle(QCoreApplication.translate("Form", u"QFileDialog文件选择对话框 - 图片转ico图标工具 by chunk", None))
        # QMainWindow.setObjectName(u"QFileDialog文件选择对话框 - 图片转ico图标工具 by chunk")
        self.pushButton.clicked.connect(self.print_filename)
        self.pushButton2.clicked.connect(self.pic_show1)
    def print_filename(self):
        FileDialog = QFileDialog(self, '文件选择', './', 'jpg(*.jpg *.jpeg);;png(*.png);;python(*.py)')
        FileDialog.setAcceptMode(QFileDialog.AcceptOpen)
        FileDirectory = FileDialog.getOpenFileNames(FileDialog, '请选择你要处理的图片', './',
                                                    'jpg(*.jpg *.jpeg);;png(*.png);;python(*.py)')

        FileDirectory = str(FileDirectory[0])
        FileName = FileDirectory.replace('[', '').replace(']', '').replace('\'', '')
        self.label1.setText(FileName)
        pixmap = QPixmap('%s' % FileName)
        self.label = QLabel()
        self.label.setPixmap(pixmap)
        self.label.show()
        img_path=FileName
        X03 = np.zeros(40000)
        img3 = Image.open(img_path)
        print("img1:", img3)
        print("img1:", type(img3))
        new_image = img3.resize(target_size)
        # imgs1.append(img1)
        # img_gray = color.rgb2gray(new_image)
        X3 = np.array(new_image)
        X3 = X3[:, :, 0:1]

        X3 = X3.reshape(-1)
        X03 = np.vstack((X03, X3))
        test_x=X03
        N1 = 10  # # of nodes belong to each window
        N2 = 10  # # of windows -------Feature mapping layer
        N3 = 500  # # of enhancement nodes -----Enhance layer
        L = 5  # # of incremental steps
        M1 = 50  # # of adding enhance nodes
        s = 0.8  # shrink coefficient
        C = 2 ** -30  # Regularization coefficient
        ymin = 0
        ymax = 1

        distOfMaxAndMin = np.load(r'C:\workcode\testIncrement02\distOf.npy')
        minOfEachWindow = np.load(r'C:\workcode\testIncrement02\minOfEac.npy')
        OutputWeight=np.load(r'C:\workcode\testIncrement02\outw.npy')
        Beta1OfEachWindow=np.load(r'C:\workcode\testIncrement02\Beta1O.npy')
        parameterOfShrink=np.load(r'C:\workcode\testIncrement02\outpara.npy')
        weightOfEnhanceLayer=np.load(r'C:\workcode\testIncrement02\weightOfEn.npy')
        # 测试过程
        test_x = preprocessing.scale(test_x)
        FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
        time_start = time.time()

        for i in range(N2):
            outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
            OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin

        InputOfEnhanceLayerWithBiasTest = np.hstack(
            [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
        tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

        OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

        InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

        OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
        for i in range(OutputOfTest.shape[0]):
            # print(OutputOfTest[i,0])
            OutputOfTest[i, 0] = round(OutputOfTest[i, 0])
            # print(OutputOfTest[i,0])
        time_end = time.time()
        testTime = time_end - time_start

        testAcc1 = show_accuracy(OutputOfTest, np.array([1,1]))
        testAcc2 = show_accuracy(OutputOfTest, np.array([2,2]))

        testAcc3 = show_accuracy(OutputOfTest, np.array([3,3]))
        max_num = 0
        n1=testAcc1
        n2=testAcc2
        n3=testAcc3
        if n1 > n2:
            max_num = n1
            if n1 > n3:
                max_num = 1
            else:
                max_num = 3
        else:
            max_num = 2
            if n2 > n3:
                max_num = 2
            else:
                max_num = 3

        if  max_num == 1:
            self.pic_show1()
            print(1)
        elif  max_num == 2:
            self.pic_show2()
            print(2)
        elif  max_num == 3:
            self.pic_show3()
            print(3)
        # return  FileName
    def  pic_show1(self):
        myWin = MyWindow()
        Filename1 = r'C:\Users\mlamp\Documents\03increase\normal\P(1)'
        Filename2 =r'C:\Windows\system32\SnippingTool.exe'

        pixmap = QPixmap('%s' % Filename1)
        self.label = QLabel()
        self.label.setPixmap(pixmap)
        self.label.show()

    def pic_show2(self):
        myWin = MyWindow()
        Filename1 = r'C:\Users\mlamp\Documents\03increase\normal\P(1)'
        # Filename2 = r'C:\Windows\system32\SnippingTool.exe'

        pixmap = QPixmap('%s' % Filename1)
        self.label = QLabel()
        self.label.setPixmap(pixmap)
        self.label.show()

    def pic_show3(self):
        myWin = MyWindow()
        Filename1 = r'C:\Users\mlamp\Documents\03increase\normal\P(1)'
        # Filename2 = r'C:\Windows\system32\SnippingTool.exe'

        pixmap = QPixmap('%s' % Filename1)
        self.label = QLabel()
        self.label.setPixmap(pixmap)
        self.label.show()
        # print(12357698760)
# Create the Qt Application
app = QApplication(sys.argv)
myWin = MyWindow()
# 展示界面

myWin.show()
# Filename1=myWin.print_filename()
# # app.exec_()
# # sys.exit(app.exec_())
# # app = QApplication(sys.argv)
# pixmap = QPixmap('%s'% Filename1)
# label = QLabel()
# label.setPixmap(pixmap)
# label.show()


app.exec_()
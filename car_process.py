from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform,data
from numpy import *
import  cv2
from scipy import misc
import os

from skimage import data, exposure, img_as_float

def AddFrame(res_end,img):#轮廓定位函数
    '''
    框选
    '''
    contours, hierarchy = cv2.findContours(res_end,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#找出所有的轮廓
    # print(contours)
    for i in range(len(contours)):##对所有的轮廓进行筛选
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        area=cv2.contourArea(cnt)
        leng = rect[1][1]
        high = rect[1][0]
        # 筛选掉小的2500,值需调整
        if area<1000:
            continue
 # if (leng/high<5 and leng/high>1) or (leng/high>0.2 and leng/high<0.74):
        if (leng/high<5 and leng/high>1) or (leng/high>0.2 and leng/high<0.74):
        # if (leng/high<5 and leng/high>1.5) or (leng/high>0.2 and leng/high<0.66):
        # if (leng/high<5 and leng/high>1.5):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 5)
 
 
    pic = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
    return pic#返回最后框选的结果

def color_Process(img):
    '''
    基于颜色
    '''
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转化为HSV空间
    H, S, V = cv2.split(HSV)

    # 蓝色的HSV空间值总是在[100,100,50]到[130,255,255]之间
    # LowerBlue = np.array([100, 100, 50])
    # UpperBlue = np.array([130, 255, 255])


    LowerBlue = np.array([100, 100, 50])
    UpperBlue = np.array([130, 255, 255])

    # HSV is [103 110  95]
    # HSV is [104 105 102]
    # HSV is [102  98  34]

    # HSV is [110 145 146]
    # HSV is [110 145 146]——————可
    # HSV is [109 115 159]

    # HSV is [103 103 158]
    # HSV is [106  64 179]
    # HSV is [104 101 157]


    # HSV图像中在LowerBlue, UpperBlue之间的变为255，之外的变为0
    mask = cv2.inRange(HSV, LowerBlue, UpperBlue)#进行蓝色的区分
    # cv2.imwrite("01BlueMask.png",mask)# _________________________________________________________
    # 利用掩膜（mask）进行“与”操作，mask中白色区域按位与，mask黑色区域剔除
    BlueThings = cv2.bitwise_and(img, img, mask=mask)

    # 转化为灰度图
    res_gray = cv2.cvtColor(BlueThings, cv2.COLOR_RGB2GRAY)
    # # cv2.imwrite("04Res_gray_binary.png",res_gray)# _________________________________________________________
    return res_gray

def edge_Process(img_gray):
    '''
    基于灰度——边缘检测
    '''
    # img_gray=grey_scale(img_gray)对比度拉伸____
 
    img_gray = cv2.medianBlur(img_gray, 5)  # 中值滤波，去除椒盐噪声
    # cv2.imwrite("11Median.png",img_gray)#————————————————————————————————————————————————————
    # gaussianResult = cv2.GaussianBlur(img_gray, (5, 5), 1.5)  # 高斯模糊____
    # img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化____
 
    x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3)  # sobel算子
    y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3)
    #laplacian = cv2.Laplacian(img_gray, cv2.CV_16S)____
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    #laplacian = cv2.convertScaleAbs(laplacian)
    img_gray = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # cv2.imwrite("12addWeighted.png",img_gray)#————————————————————————————————————————————————————
    retval, img_gray = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)  # 二值化方法
    # cv2.imwrite("13Two_threshold.png",img_gray)#————————————————————————————————————————————————————
 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    img_gray = cv2.dilate(img_gray, kernel, iterations=1)# 膨胀
    # cv2.imwrite("14Pengzhang.png",img_gray)#——————————————————————————————————————————————————
    img_gray = cv2.erode(img_gray, kernel)#腐蚀
    # cv2.imwrite("15fushi.png",img_gray)#—————————————————————————————————————————————
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)#开运算
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite("15Kai.png",img_gray)#—————————————————————————————————————————————
    return img_gray


if __name__=="__main__":
    img = []
    # 读入
    for root,dir,files in os.walk('./DataSource/'):
        for file in files:
            img.append(cv2.imread('./DataSource/' + str(file),-1))
    # 写出
    for i in range(len(img)):
        # img[i]= exposure.adjust_gamma(img[i], 0.73)  #调亮
        color_Temp = color_Process(img[i])
        edge_Temp =  edge_Process(color_Temp)
        res_img = AddFrame(edge_Temp,img[i])
        cv2.imwrite("./result/" + str(i)+".jpg",res_img)

# if __name__=="__main__":
#     img = []
#     # 读入
#     for root,dir,files in os.walk('./DataSource/'):
#         for file in files:
#             img.append(cv2.imread('./DataSource/' + str(file),-1))
#     i = 55
#     cv2.imwrite("00source.png",img[i])#_______________________________________________
#     color_Temp = color_Process(img[i])
#     edge_Temp =  edge_Process(color_Temp)
#     res_img = AddFrame(edge_Temp,img[i])
    
#     cv2.imwrite("res.jpg", res_img)


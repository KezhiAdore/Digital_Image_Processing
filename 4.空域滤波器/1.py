import cv2 as cv
import numpy as np
import math
img=cv.imread("test1.pgm")

sigma=1.5   #高斯滤波器的参数
def f(x,y):     #定义二维正态分布函数
    return 1/(math.pi*sigma**2)*math.exp(-(x**2+y**2)/(2*sigma**2))

def gauss(n):   #生成n*n的高斯滤波器
    mid=n//2
    filt=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            filt[i][j]=f(i-mid,j-mid)/f(-mid,-mid)
    return filt.astype(np.uint8)

def gauss_filter(img,n):    #对图像img进行n*n卷积块的高斯滤波
    filt=gauss(n)
    con=1/np.sum(filt)
    shape=img.shape
    mid=n//2
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))   #对边缘进行补0操作
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j][0]
    result=np.zeros((shape[0],shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            tmp=0
            for k in range(n):
                for l in range(n):
                    tmp+=filt[k][l]*temp[i+k][j+l]
            result[i][j]=con*tmp
    return result.astype(np.uint8)

def center_filter(img, n):  #使用n*n的中值滤波器对图像进行低通滤波
    mid=n//2
    shape=img.shape
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j][0]
    result=np.zeros((shape[0],shape[1]))
    tmp=np.zeros(n*n)
    for i in range (shape[0]):
        for j in range(shape[1]):
            for k in range(n):
                for l in range(n):
                    tmp[k*n+l]=temp[i+k][j+l]
            result[i][j]=np.median(tmp)
    return result.astype(np.uint8)

filename=["test1.pgm","test2.tif"]
size=[3,5,7]
for i in filename:
    img=cv.imread(i)
    for j in size:
        cv.imwrite(i+'gauss-'+str(j)+'.bmp',gauss_filter(img,j))
        cv.imwrite(i+'center-'+str(j)+'.bmp',center_filter(img,j))
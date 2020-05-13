from cv2 import cv2 as cv
import numpy as np
import math

def fft(img):       #对图像做二维快速傅里叶变换
    shape=img.shape
    new_img=np.zeros((shape[0]*2,shape[1]*2))
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_img[i][j]=img[i][j]*(-1)**(i+j)
    fimg=np.fft.fft2(new_img)
    return fimg

def ifft(f_img):
    img=np.real(np.fft.ifft2(f_img))
    shape=img.shape
    new_img=np.zeros((shape[0]//2,shape[1]//2))
    for i in range(shape[0]//2):
        for j in range(shape[1]//2):
            new_img[i][j]=img[i][j]*(-1)**(i+j)
    new_img=new_img-np.min(new_img)
    new_img=new_img/np.max(new_img)*255
    return new_img.astype(np.uint8)

def show_filter(filt):  #以图像的方式呈现出滤波器
    filt=np.abs(filt)
    filt=filt-np.min(filt)
    filt=filt/np.max(filt)*255
    filt=filt.astype(np.uint8)
    # cv.namedWindow("figure",cv.WINDOW_FREERATIO)
    # cv.imshow("figure",filt)
    # cv.waitKey(0)
    return filt

def laplace_filter(shape):
    filt=np.zeros(shape)
    P,Q=shape[0],shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            filt[i][j]=-4*math.pi**2*((i-P//2)**2+(j-Q//2)**2)
    return filt

def laplace(img):
    f_img=fft(img)
    shape=f_img.shape
    filt=laplace_filter(shape)
    f_result=f_img*filt
    return ifft(f_result)

def butter_filter(shape,n,D0):   #生成n阶，半径为D0的butterworth高通滤波器
    filt=np.zeros(shape)
    P,Q=shape[0],shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            D=math.sqrt((i-P//2)**2+(j-Q//2)**2)+0.01
            filt[i][j]=1/(1+(D0/D)**(2*n))
    return filt

def gauss_filter(shape,D0):    #生成n阶，滤波半径为D0的高斯高通滤波器
    filt=np.zeros(shape)
    P,Q=shape[0],shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            D2=(i-P//2)**2+(j-Q//2)**2
            filt[i][j]=1-math.exp(-D2/(2*D0**2))
    return filt

def unmask_filt(hp_filt,k1=1,k2=1):
    return k1+k2*hp_filt

def unmask_butter(img,n,D0):
    f_img=fft(img)
    f_result=f_img*unmask_filt(butter_filter(f_img.shape,n,D0))
    return ifft(f_result)

def unmask_gauss(img,D0):
    f_img=fft(img)
    f_result=f_img*unmask_filt(gauss_filter(f_img.shape,D0))
    return ifft(f_result)

files=["test3_corrupt.pgm","test4.tif"]

#产生拉普拉斯滤波器的图像
# cv.imwrite("laplace_filter.jpg",show_filter(laplace_filter((200,200))))

#对图像进行拉普拉斯滤波
# for i in files:
#     img=cv.imread(i,cv.IMREAD_UNCHANGED)
#     cv.imwrite(i+"laplace.jpg",laplace(img))

#对图像进行掩膜滤波(使用butterwroth滤波器)
# for i in files:
#     img=cv.imread(i,cv.IMREAD_UNCHANGED)
#     cv.imwrite(i+"unmask_butter.jpg",unmask_butter(img,3,10))

#对图像进行掩膜滤波（使用gauss滤波器）
for i in files:
    img=cv.imread(i,cv.IMREAD_UNCHANGED)
    cv.imwrite(i+"unmask_gauss.jpg",unmask_gauss(img,10))
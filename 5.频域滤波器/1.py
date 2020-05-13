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

def show_fft(f_img):    #将fft的结果对10取对数之后规整到0-255并显示出来
    f_img=np.abs(f_img)
    f_img=np.log10(f_img)
    f_img=(f_img-np.min(f_img))/np.max(f_img)*255
    cv.namedWindow("figure",cv.WINDOW_AUTOSIZE)
    cv.imshow("figure",f_img.astype(np.uint8))
    cv.waitKey(0)

def show_filter(filt):  #以图像的方式呈现出滤波器
    filt=np.abs(filt)
    filt=filt-np.min(filt)
    filt=filt/np.max(filt)*255
    filt=filt.astype(np.uint8)
    # cv.namedWindow("figure",cv.WINDOW_FREERATIO)
    # cv.imshow("figure",filt)
    # cv.waitKey(0)
    return filt

def butter_filter(shape,n,D0):   #生成n阶，半径为D0的butterworth低通滤波器
    filt=np.zeros(shape)
    P,Q=shape[0],shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            D=math.sqrt((i-P//2)**2+(j-Q//2)**2)
            filt[i][j]=1/(1+(D/D0)**(2*n))
    return filt
def butterwroth(img,n,D0):  #使用n阶，半径为D0的butterworth滤波器对图像进行滤波
    f_img=fft(img)
    shape=f_img.shape
    f_result=f_img*butter_filter(shape,n,D0)
    return ifft(f_result)

def gauss_filter(shape,D0):    #生成截止频率为D0的高斯低通滤波器
    filt=np.zeros(shape)
    P,Q=shape[0],shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            D2=(i-P//2)**2+(j-Q//2)**2
            filt[i][j]=math.exp(-D2/(2*D0**2))
    return filt

def gaussian(img,D0):  #使用n阶，截止频率为D0的高斯低通滤波器对图像进行滤波
    f_img=fft(img)
    shape=f_img.shape
    f_result=f_img*gauss_filter(shape,D0)
    return ifft(f_result)

def power(img,D0):
    f_img=np.abs(fft(img))
    f_img=f_img*f_img
    shape=f_img.shape
    P,Q=shape[0],shape[1]
    ans=0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if math.sqrt((i-P//2)**2+(j-Q//2)**2)<D0:
                ans=ans+f_img[i][j]
    return ans/np.sum(f_img)

files=["test2.tif"]
r=[10,20]
# for i in r:
#     cv.imwrite("guass"+str(i)+".jpg",show_filter(gauss_filter((200,200),i)))
# for i in range(4):
#     cv.imwrite("butter"+str(i+1)+".jpg",show_filter(butter_filter((200,200),i+1,10)))
for i in files:
    img=cv.imread(i,cv.IMREAD_UNCHANGED)
    for j in r:
        cv.imwrite(i+str(j)+".jpg",butterwroth(img,3,j))
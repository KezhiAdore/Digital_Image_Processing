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

def butter_filter(shape,n,D0):   #生成n阶，半径为D0的butterworth高通滤波器
    filt=np.zeros(shape)
    P,Q=shape[0],shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            D=math.sqrt((i-P//2)**2+(j-Q//2)**2)+0.01
            filt[i][j]=1/(1+(D0/D)**(2*n))
    return filt

def butterwroth(img,n,D0):  #使用n阶，半径为D0的butterworth高通滤波器对图像进行滤波
    f_img=fft(img)
    shape=f_img.shape
    f_result=f_img*butter_filter(shape,n,D0)
    return ifft(f_result)

def gauss_filter(shape,D0):    #生成n阶，滤波半径为D0的高斯高通滤波器
    filt=np.zeros(shape)
    P,Q=shape[0],shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            D2=(i-P//2)**2+(j-Q//2)**2
            filt[i][j]=1-math.exp(-D2/(2*D0**2))
    return filt

def gaussian(img,D0):  #使用n阶，滤波半径为D0的高斯高通滤波器对图像进行滤波
    f_img=fft(img)
    shape=f_img.shape
    f_result=f_img*gauss_filter(shape,D0)
    return ifft(f_result)

def power(img,D0):      #计算功率谱比
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

files=["test3_corrupt.pgm","test4.tif"]
r=[5,10,30,60]

#产生半径为10，阶数不同的巴特沃斯滤波器
# for i in range(4):
#     cv.imwrite("butter"+str(i+1)+".jpg",show_filter(butter_filter((200,200),i+1,10)))

#产生半径分别为5,10,30,60的高斯滤波器
# for i in r:
#     cv.imwrite("gauss"+str(i)+".jpg",show_filter(gauss_filter((200,200),i)))

#使用巴特沃斯滤波器对图像进行滤波
# for i in files:
#     img=cv.imread(i,cv.IMREAD_UNCHANGED)
#     for j in r:
#         print((1-power(img,j*2))*100)
        # for k in range(3):
        #     cv.imwrite(i+"butter-"+str(k+1)+"-"+str(j)+".jpg",butterwroth(img,k+1,j*2))

#使用高斯滤波对图像进行滤波
# for i in files:
#     img=cv.imread(i,cv.IMREAD_UNCHANGED)
#     for j in r:
#         cv.imwrite(i+"gauss-"+str(j)+".jpg",gaussian(img,j*2))
img=cv.imread("test3_corrupt.pgm",cv.IMREAD_UNCHANGED)
cv.imwrite("tmp.bmp",img)
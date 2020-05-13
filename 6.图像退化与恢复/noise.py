import numpy as np
from cv2 import cv2 as cv
import random
import math
import cmath

def gauss_noise(img,mu,sigma):      #给图像加入高斯噪声
    shape=img.shape
    new_img=np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_img[i][j]=img[i][j]+random.gauss(mu,sigma)
    new_img=np.clip(new_img,0,255)
    return new_img.astype(np.uint8)

def spicedsalt_noise(img,p_spiced,p_salt):  #给图像加入椒盐噪声
    shape=img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            rand=random.random()
            if rand<p_spiced+p_salt:
                if rand<p_spiced:
                    img[i][j]=0
                else:
                    img[i][j]=255
    return img

def fft(img):       #对图像做二维快速傅里叶变换
    shape=img.shape
    new_img=np.zeros((shape[0]*2,shape[1]*2))
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_img[i][j]=img[i][j]*(-1)**(i+j)
    fimg=np.fft.fft2(new_img)
    return fimg

def ifft(f_img):    #二维傅里叶反变换
    img=np.real(np.fft.ifft2(f_img))
    shape=img.shape
    new_img=np.zeros((shape[0]//2,shape[1]//2))
    for i in range(shape[0]//2):
        for j in range(shape[1]//2):
            new_img[i][j]=img[i][j]*(-1)**(i+j)
    new_img=new_img-np.min(new_img)
    new_img=new_img/np.max(new_img)*255
    return new_img.astype(np.uint8)

#运动模糊退化函数
def H_function(u,v,a,b,T):
    tmp=math.pi*(u*a+v*b)
    if tmp==0:
        tmp=1
    return T/tmp*math.sin(tmp)*cmath.exp(-tmp*1j)
#图像运动模糊
def move_noise(img,a,b,T):
    f_img=fft(img)
    shape=f_img.shape
    P,Q=shape[0]//2,shape[1]//2
    for u in range(shape[0]):
        for v in range(shape[1]):
            f_img[u][v]=(f_img[u][v]*H_function(u-P,v-Q,a,b,T))
    return ifft(f_img)
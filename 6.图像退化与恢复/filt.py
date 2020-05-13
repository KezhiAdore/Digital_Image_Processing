from cv2 import cv2 as cv
import numpy as np
import math
import cmath

##频域滤波器
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

def gauss_filter(shape,D0):    #生成n阶，截止频率为D0的高斯低通滤波器
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

##空域滤波器
##均值滤波器
def arithmetic_mean_filt(img, n):  #使用n*n的算数平均滤波器对图像进行低通滤波
    mid=n//2
    shape=img.shape
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j]
    result=np.zeros((shape[0],shape[1]))
    for i in range (shape[0]):
        for j in range(shape[1]):
            tmp=0
            for k in range(n):
                for l in range(n):
                    tmp+=temp[i+k][j+l]
            result[i][j]=tmp/(n*n)
    return result.astype(np.uint8)

def geometric_mean_filt(img, n):  #使用n*n的几何平均滤波器对图像进行低通滤波
    mid=n//2
    shape=img.shape
    temp=np.ones((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]+=img[i][j]
    result=np.zeros((shape[0],shape[1]))
    for i in range (shape[0]):
        for j in range(shape[1]):
            tmp=1
            for k in range(n):
                for l in range(n):
                    tmp*=temp[i+k][j+l]
            result[i][j]=tmp**(1/(n*n))
    return result.astype(np.uint8)

def Invertedharmonic_mean_filt(img, n, Q):  #使用n*n,阶数为Q的逆谐波均值滤波器对图像进行低通滤波
    mid=n//2
    shape=img.shape
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]+=img[i][j]
    result=np.zeros((shape[0],shape[1]))
    for i in range (shape[0]):
        for j in range(shape[1]):
            mole,deno=0,0
            for k in range(n):
                for l in range(n):
                    if temp[i+k][j+l]==0:
                        continue
                    mole+=(temp[i+k][j+l])**(Q+1)
                    deno+=(temp[i+k][j+l])**Q
            result[i][j]=mole/deno
    result=np.clip(result-1,0,255)
    return result.astype(np.uint8)
##统计排序滤波器
def med_filt(img, n):  #使用n*n的中值滤波器对图像进行低通滤波
    mid=n//2
    shape=img.shape
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j]
    result=np.zeros((shape[0],shape[1]))
    tmp=np.zeros(n*n)
    for i in range (shape[0]):
        for j in range(shape[1]):
            for k in range(n):
                for l in range(n):
                    tmp[k*n+l]=temp[i+k][j+l]
            result[i][j]=np.median(tmp)
    return result.astype(np.uint8)

def min_filt(img, n):  #使用n*n的最小值滤波器对图像进行低通滤波
    mid=n//2
    shape=img.shape
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j]
    result=np.zeros((shape[0],shape[1]))
    tmp=np.zeros(n*n)
    for i in range (shape[0]):
        for j in range(shape[1]):
            for k in range(n):
                for l in range(n):
                    tmp[k*n+l]=temp[i+k][j+l]
            result[i][j]=np.min(tmp)
    return result.astype(np.uint8)

def max_filt(img, n):  #使用n*n的中值滤波器对图像进行低通滤波
    mid=n//2
    shape=img.shape
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j]
    result=np.zeros((shape[0],shape[1]))
    tmp=np.zeros(n*n)
    for i in range (shape[0]):
        for j in range(shape[1]):
            for k in range(n):
                for l in range(n):
                    tmp[k*n+l]=temp[i+k][j+l]
            result[i][j]=np.max(tmp)
    return result.astype(np.uint8)

def mid_filt(img, n):  #使用n*n的中点滤波器对图像进行低通滤波
    mid=n//2
    shape=img.shape
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j]
    result=np.zeros((shape[0],shape[1]))
    tmp=np.zeros(n*n)
    for i in range (shape[0]):
        for j in range(shape[1]):
            for k in range(n):
                for l in range(n):
                    tmp[k*n+l]=temp[i+k][j+l]
            result[i][j]=(np.max(tmp)+np.min(tmp))/2
    return result.astype(np.uint8)

def alpha_filt(img, n, d):  #使用n*n,修正大小为d的修正阿尔法滤波器对图像进行低通滤波
    mid=n//2
    shape=img.shape
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j]
    result=np.zeros((shape[0],shape[1]))
    tmp=np.zeros(n*n)
    for i in range (shape[0]):
        for j in range(shape[1]):
            for k in range(n):
                for l in range(n):
                    tmp[k*n+l]=temp[i+k][j+l]
                    np.sort(tmp)
            result[i][j]=np.sum(tmp[d//2+1:n*n-d//2])/(n*n-d)
    result=np.clip(result,0,255)
    return result.astype(np.uint8)

##自适应滤波器
#自适应局部降噪滤波器
def adaptive_mean(img,n,sigma2):  #窗口大小为n*n，噪声方差为sigma2
    mid=n//2
    shape=img.shape
    temp=np.zeros((shape[0]+n-1,shape[1]+n-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j]
    result=np.zeros((shape[0],shape[1]))
    tmp=np.zeros(n*n)
    for i in range (shape[0]):
        for j in range(shape[1]):
            for k in range(n):
                for l in range(n):
                    tmp[k*n+l]=temp[i+k][j+l]
            sigma_tmp=np.var(tmp)
            if (sigma2<sigma2):
                result[i][j]=img[i][j]-sigma2/sigma_tmp*(img[i][j]-np.mean(tmp))
            else:
                result[i][j]=np.mean(tmp)
    result=np.clip(result,0,255)
    return result.astype(np.uint8)
#自适应中值滤波
def adaptive_mid(img,Smax):
    mid=Smax//2
    shape=img.shape
    temp=np.zeros((shape[0]+Smax-1,shape[1]+Smax-1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+mid][j+mid]=img[i][j]
    result=np.zeros(shape)
    tmp=np.zeros((Smax,Smax))
    for i in range (shape[0]):
        for j in range(shape[1]):
            for k in range(Smax):
                for l in range(Smax):
                    tmp[k][l]=temp[i+k][j+l]
            for k in range(1,Smax+1,2):
                a=tmp[mid-k//2:mid+k//2+1,mid-k//2:mid+k//2+1]
                a_min,a_max,a_mid=np.min(a),np.max(a),np.median(a)
                if a_mid-a_min>0 and a_mid-a_max<0:
                    if img[i][j]-a_min>0 and img[i][j]-a_max<0:
                        result[i][j]=img[i][j]
                        break
                    else:
                        result[i][j]=a_mid
                        break
                elif k==Smax:
                    result[i][j]=a_mid
    return result.astype(np.uint8)

#运动模糊退化函数
def H_function(u,v,a,b,T):
    tmp=math.pi*(u*a+v*b)
    if tmp==0:
        tmp=1
    return T/tmp*math.sin(tmp)*cmath.exp(-tmp*1j)
#拉普拉斯滤波器
def P_function(u,v):
    return -4*(math.pi)**2*(u**2+v**2)
#最小均方误差滤波
def Minimum_mean_variance_filt(img,K,a,b,T):
    f_img=fft(img)
    shape=f_img.shape
    P,Q=shape[0]//2,shape[1]//2
    for u in range(shape[0]):
        for v in range(shape[1]):
            H_tmp=H_function(u-P,v-Q,a,b,T)
            f_img[u][v]=f_img[u][v]*(abs(H_tmp)**2/H_tmp/(abs(H_tmp)**2+K))
    return ifft(f_img)
#约束最小二乘方滤波
def Constraint_minimum_square_filt(img,gamma,a,b,T):
    f_img=fft(img)
    shape=f_img.shape
    P,Q=shape[0]//2,shape[1]//2
    for u in range(shape[0]):
        for v in range(shape[1]):
            H_tmp=H_function(u-P,v-Q,a,b,T)
            P_tmp=P_function(u-P,v-Q)
            f_img[u][v]=f_img[u][v]*(H_tmp.conjugate()/(abs(H_tmp)**2+gamma*abs(P_tmp)**2))
    return ifft(f_img)
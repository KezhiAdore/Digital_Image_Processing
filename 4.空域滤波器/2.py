from cv2 import cv2 as cv
import numpy as np
import math

sigma=1.5   #高斯滤波器的参数

def add_zeros(img,edge):    #图像边缘补零
    shape=img.shape
    temp=np.zeros((shape[0]+2*edge,shape[1]+2*edge))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp[i+edge][j+edge]=img[i][j][0]
    return temp

def f(x,y):     #定义二维正态分布函数
    return 1/(math.pi*sigma**2)*math.exp(-(x**2+y**2)/(2*sigma**2))

def gauss(n):   #生成n*n的高斯滤波器
    mid=n//2
    filt=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            filt[i][j]=f(i-mid,j-mid)/f(-mid,-mid)
    return filt.astype(np.uint8)

def gauss_filter(img,n):    #对图像进行n*n卷积块的高斯滤波
    filt=gauss(n)
    con=1/np.sum(filt)
    shape=img.shape
    temp=add_zeros(img,n//2)
    result=np.zeros((shape[0],shape[1],1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            tmp=0
            for k in range(n):
                for l in range(n):
                    tmp+=filt[k][l]*temp[i+k][j+l]
            result[i][j][0]=con*tmp
    return result.astype(np.uint8)

def unsharp_mask(img,n,is_mask=0):        #用n*n高斯模糊后非锐化掩膜,is_mask为0返回叠加后图像，为1返回掩膜
    shape=img.shape
    new_img=np.zeros((shape[0],shape[1],1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_img[i][j][0]=img[i][j][0]
    mask=new_img-gauss_filter(img,n)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i==0 or j==0 or i==shape[0]-1 or j==shape[1]-1:
                mask[i][j][0]=0
    result=new_img+mask
    result=result-np.min(result)
    result=result/np.max(result)*255
    mask=mask-np.min(mask)
    mask=mask/np.max(mask)*255
    if is_mask:
        return mask.astype(np.uint8)    #返回掩膜
    return result.astype(np.uint8)  #返回结果

sobelx=[[-1,0,1],[-2,0,2],[-1,0,1]]
sobely=[[-1,-2,-1],[0,0,0],[1,2,1]]
laplace4=[[0,-1,0],[-1,4,-1],[0,-1,0]]
laplace8=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

def filt_3(img,filt):          #任意3*3滤波器(图像,算子)
    shape=img.shape
    temp=add_zeros(img,1)
    result=np.zeros((shape[0],shape[1],1))
    for i in range(shape[0]):
        for j in range(shape[1]):
            tmp=0
            for k in range(3):
                for l in range(3):
                    tmp+=filt[k][l]*temp[i+k][j+l]
            result[i][j][0]=tmp
    result=np.abs(result)     #分别输出x方向和y方向的边缘信息
    result=result/np.max(result)*255
    cv.imwrite("tmp.bmp",result.astype(np.uint8))
    return result

def laplace_edge(img,filt): #规整后的拉普拉斯算子边缘
    tmp=filt_3(img,filt)
    tmp=tmp-np.min(tmp)
    shape=tmp.shape
    for i in range(shape[0]):
        for j in range (shape[1]):
            if i==0 or j==0 or i==shape[0]-1 or j==shape[1]-1:
                tmp[i][j][0]=0
    tmp=tmp/np.max(tmp)*255
    return tmp.astype(np.uint8)

def laplace(img,filt):  #原图像与拉普拉斯边缘叠加的结果图像
    tmp=filt_3(img,filt)
    shape=img.shape
    result=np.zeros((shape[0],shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i][j]=tmp[i][j][0]+img[i][j][0]
            if i==0 or j==0 or i==shape[0]-1 or j==shape[1]-1:
                result[i][j]=0
    result-=np.min(result)
    result=result/np.max(result)*255
    return result.astype(np.uint8)

def sobel(img):     #提取图像的sobel边缘
    shape=img.shape
    sobx=filt_3(img,sobelx)
    soby=filt_3(img,sobely)
    result=np.zeros((shape[0],shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i==0 or j==0 or i==shape[0]-1 or j==shape[1]-1:
                result[i][j]=0
            else:
                result[i][j]=math.sqrt(sobx[i][j][0]**2+soby[i][j][0]**2)
    result=result/np.max(result)*255
    return result.astype(np.uint8)

def canny(img,n=3):     #使用canny算法提取图像边缘（使用n*n的高斯滤波器进行模糊操作）
    de=[[1,0,-1,0],[1,1,-1,-1],[0,1,0,-1],[-1,1,1,-1]]
    shape=img.shape
    tmp=gauss_filter(img,n)
    sobx=filt_3(tmp,sobelx)
    soby=filt_3(tmp,sobely)
    weight,angle,result=np.zeros((shape[0],shape[1])),np.zeros((shape[0],shape[1])),np.zeros((shape[0],shape[1]))
    angle=angle.astype(np.int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            weight[i][j]=math.sqrt(sobx[i][j][0]**2+soby[i][j][0]**2)
            if sobx[i][j][0]:
                angle[i][j]=round((math.atan(soby[i][j][0]/sobx[i][j][0])/(math.pi/4)-0.5))%4
    for i in range(shape[0]-2):
        for j in range(shape[1]-2):
            tmp_i,tmp_j=i+1,j+1
            if weight[tmp_i][tmp_j]<=weight[tmp_i+de[angle[tmp_i][tmp_j]][0]][tmp_j+de[angle[tmp_i][tmp_j]][1]] and weight[tmp_i][tmp_j]<=weight[tmp_i+de[angle[tmp_i][tmp_j]][2]][tmp_j+de[angle[tmp_i][tmp_j]][3]]:
                result[tmp_i][tmp_j]=0
            else:
                result[tmp_i][tmp_j]=weight[tmp_i][tmp_j]
    result=result/np.max(result)*255
    mean=np.mean(img)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if result[i][j]<100:
                result[i][j]=0
    return result.astype(np.uint8)


filename=["test3_corrupt.pgm","test4.tif"]
for i in filename:
    img=cv.imread(i)
    # cv.imwrite(i+"_mask.bmp",unsharp_mask(img,3,1))
    # cv.imwrite(i+"_unsharp_mask.bmp",unsharp_mask(img,3))
    cv.imwrite(i+"_sobel.bmp",sobel(img))
    # cv.imwrite(i+"_canny.bmp",canny(img,3))
    cv.imwrite(i+"laplace4_edge.bmp",laplace_edge(img,laplace4))
    cv.imwrite(i+"laplace8_edge.bmp",laplace_edge(img,laplace8))
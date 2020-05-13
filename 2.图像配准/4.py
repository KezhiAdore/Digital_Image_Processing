import cv2 as cv
import numpy as np

def trans(imname):
    img=cv.imread(imname)
    shape=img.shape
    cnt_i=shape[0]//7+1
    cnt_j=shape[0]//7+1
    result=np.zeros((shape[0],shape[1]))
    for i in range(cnt_i):
        for j in range(cnt_j):
            num=np.zeros(256)
            for point_i in range(i*7,i*7+7):
                for point_j in range(j*7,j*7+7):
                    if point_i<shape[0] and point_j<shape[1]:
                        num[img[point_i][point_j][0]]+=1
            fre=num/np.sum(num)
            T=np.zeros((256))
            T[0]=fre[0]*256
            for k in range(256):
                if k:
                    T[k]=T[k-1]+fre[k]*255
            for point_i in range(i*7,i*7+7):
                for point_j in range(j*7,j*7+7):
                    if point_i<shape[0] and point_j<shape[1]:
                        result[point_i][point_j]=T[img[point_i][point_j][0]]
    result=result.astype(np.uint8)
    cv.imwrite("trans_"+imname,result)

trans('lena.bmp')
trans('elain.bmp')
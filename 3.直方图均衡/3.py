import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams
def find(a,x):
    shape_tmp=a.shape
    for i in range(shape_tmp[0]):
        if x-a[i]<=0:
            return i
    return shape_tmp[0]-1
def graybar_trans(img,cnt,flag):
    file_tem=['citywall.bmp','elain.bmp','lena.bmp','woman.bmp']
    img_tem=cv.imread(file_tem[cnt])
    num,num_temp=np.zeros((256)),np.zeros((256))
    img_shape=img.shape
    if flag:
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                num[img[i][j][0]]+=1
                num_temp[img_tem[i][j][0]]+=1
    else:
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if img[i][j][0]==0:
                    img[i][j][0]=255
                num[img[i][j][0]]+=1
                num_temp[img_tem[i][j][0]]+=1
    fre=num/np.sum(num)
    fre_temp=num_temp/np.sum(num_temp)
    z,z_temp=np.zeros((256)),np.zeros((256))
    z[0],z_temp[0]=fre[0]*256,fre_temp[0]*256
    for i in range(256):
        if i:
            z[i]=z[i-1]+fre[i]*255
            z_temp[i]=z_temp[i-1]+fre_temp[i]*255
    trans=np.zeros((256))
    for i in range(256):
        trans[i]=find(z_temp,z[i])
    trans=trans.astype(np.uint8)
    new_img=np.zeros((img_shape[0],img_shape[1]))
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            new_img[i][j]=trans[img[i][j][0]]
    return new_img.astype(np.uint8)
def cnt(str):
    if 'citywall' in str:
        return 0
    if 'elain' in str:
        return 1
    if 'lena' in str:
        return 2
    if 'woman' in str:
        return 3
def flag(str):
    if '1' in str:
        return 0
    return 1
filename=['citywall.bmp','citywall1.bmp','citywall2.bmp','elain.bmp','elain1.bmp','elain2.bmp','elain3.bmp','lena.bmp','lena1.bmp','lena2.bmp','lena3.bmp','woman.BMP','woman1.bmp','woman2.bmp']
for i in filename:
    img=cv.imread(i)
    result=graybar_trans(img,cnt(i),flag(i))
    # cv.namedWindow(i,cv.WINDOW_FREERATIO)
    # cv.imshow(i,result)
    # cv.waitKey(0)
    cv.imwrite("aim_bar_"+i,result)
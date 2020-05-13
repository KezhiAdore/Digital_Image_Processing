from cv2 import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams
def graybar_trans(img,flag):
    num=np.zeros((256))
    img_shape=img.shape
    if flag:
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                num[img[i][j]]+=1
    else:
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if img[i][j][0]==0:
                    img[i][j][0]=255
                num[img[i][j][0]]+=1
    fre=num/np.sum(num)
    trans=np.zeros((256))
    trans[0]=fre[0]*256
    for i in range(256):
        if i:
            trans[i]=trans[i-1]+fre[i]*255
    trans=trans.astype(np.uint8)
    new_img=np.zeros((img_shape[0],img_shape[1]))
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            new_img[i][j]=trans[img[i][j]]
    return new_img.astype(np.uint8)
def flag(str):
    if '1' in str:
        return 0
    return 1
# filename=['citywall.bmp','citywall1.bmp','citywall2.bmp','elain.bmp','elain1.bmp','elain2.bmp','elain3.bmp','lena.bmp','lena1.bmp','lena2.bmp','lena3.bmp','woman.BMP','woman1.bmp','woman2.bmp']
# for i in filename:
#     img=cv.imread(i)
#     result=graybar_trans(img,flag(i))
#     # cv.namedWindow(i,cv.WINDOW_FREERATIO)
#     # cv.imshow(i,result)
#     # cv.waitKey(0)
#     cv.imwrite('bar_'+i,result)
img=cv.imdecode(np.fromfile(r"C:\Users\于钊\Desktop\线上学习\数字图像处理\作业\第5次作业\test4.tiflaplace.jpg",dtype=np.uint8),-1)
cv.imwrite("tmp.jpg",graybar_trans(img,1))
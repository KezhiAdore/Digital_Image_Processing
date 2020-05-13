import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams
def graybar(img,win_num,img_name,flag):
    num=np.zeros((256))
    img_shape=img.shape
    if flag:
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                num[img[i][j][0]]+=1
    else:
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if img[i][j][0]==0:
                    img[i][j][0]=255
                num[img[i][j][0]]+=1
    fig=plt.figure(win_num)
    rects=plt.bar(range(256),num,0.2)
    plt.title('gray_bar_'+img_name)
    plt.show()
def flag(str):
    if '1' in str:
        return 0
    return 1
filename=['citywall.bmp','citywall1.bmp','citywall2.bmp','elain.bmp','elain1.bmp','elain2.bmp','elain3.bmp','lena.bmp','lena1.bmp','lena2.bmp','lena3.bmp','woman.BMP','woman1.bmp','woman2.bmp']
for i in filename:
    img=cv.imread(i)
    graybar(img,0,i,flag(i))

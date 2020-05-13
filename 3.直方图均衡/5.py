import cv2 as cv
import numpy as np
import string
def division(imname,threshold):
    img=cv.imread(imname)
    shape=img.shape
    img1,img2=np.zeros((shape[0],shape[1])),np.zeros((shape[0],shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i][j][0]<threshold:
                img1[i][j]=img[i][j][0]
            else:
                img2[i][j]=img[i][j][0]
    cv.imwrite('div'+str(threshold)+'-'+imname,img1.astype(np.uint8))
    cv.imwrite('div2_'+str(threshold)+'+'+imname,img2.astype(np.uint8))

division('elain.bmp',210)
division('woman.bmp',90)
division('woman.bmp',170)
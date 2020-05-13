import cv2 as cv
import numpy as np
import math
def weight(x):
    a=-0.5
    x=abs(x)
    if x<=1:
        return  (a+2)*x**3-(a+3)*x**2+1
    if x<2:
        return a*x**3-5*a*x**2+8*a*x-4*a
    return 0
def zoom(source,filename):
    result=[np.zeros((2048,2048)),np.zeros((2048,2048)),np.zeros((2048,2048))]
    range_list=range(2048)
    range_tmp=range(4)
    for i in range_list:
        for j in range_list:
            if i>4 and i<2040 and j>4 and j<2040:
                result[0][i][j]=source[round(i/4)][round(j/4)]
                m,n,a,b=i//4,j//4,i/4-i//4,j/4-j//4
                result[1][i][j]=source[m][n]*(1-a)*(1-b)+source[m+1][n]*a*(1-b)+source[m][n+1]*(1-a)*b+source[m+1][n+1]*a*b
                for k in range_tmp:
                    for w in range_tmp:
                        result[2][i][j]+=source[i//4+k-1][j//4+w-1]*weight(i/4-(i//4+k-1))*weight(j/4-(j//4+w-1))
            else:
                result[0][i][j]=source[i//4][j//4]
                result[1][i][j]=source[i//4][j//4]
                result[2][i][j]=source[i//4][j//4]
    result[0]=result[0].astype(np.uint8)
    cv.imwrite(filename+"_nearest.bmp",result[0])
    result[1]=result[1].astype(np.uint8)
    cv.imwrite(filename+"_Bilinear.bmp",result[1])
    result[2]=result[2].astype(np.uint8)
    cv.imwrite(filename+"_Bicubic.bmp",result[2])

img1=cv.imread("lena.bmp")
img2=cv.imread("elain1.bmp")
img_lena_shear,img_elain_shear,img_lena_rotate,img_elain_rotate=np.zeros((512,512)),np.zeros((512,512)),np.zeros((512,512)),np.zeros((512,512))
for i in range(512):
    for j in range(512):
        if math.floor(i+0.5*j)<512:
            img_lena_shear[math.floor(i+0.5*j)][j]=img1[i][j][0]
            img_elain_shear[math.floor(i+0.5*j)][j]=img2[i][j][0]
img_lena_shear=img_lena_shear.astype(np.uint8)
zoom(img_lena_shear,"lena_shear")
img_elain_shear=img_elain_shear.astype(np.uint8)
zoom(img_elain_shear,"elain_shear")
a,b=math.cos(math.pi/6),math.sin(math.pi/6)
T=np.array([[a,b,0],[-b,a,0],[0,0,1]])
T=np.linalg.inv(T)
for i in range(512):
    for j in range(512):
        A=np.dot(np.array([i-256,j-256,1]),T)+np.array([256,256,0])
        A=A.astype(np.int)
        if A[0]>0 and A[0]<512 and A[1]>0 and A[1]<512:
            img_lena_rotate[i][j]=img1[A[0]][A[1]][0]
            img_elain_rotate[i][j]=img2[A[0]][A[1]][0]
img_lena_rotate=img_lena_rotate.astype(np.uint8)
zoom(img_lena_rotate,"lena_rotate")
img_elain_rotate=img_elain_rotate.astype(np.uint8)
zoom(img_elain_rotate,"elain_rotate")
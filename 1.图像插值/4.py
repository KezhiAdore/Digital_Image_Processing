import cv2 as cv
import numpy as np
import math
img=cv.imread("lena.bmp")
result=[np.zeros((2048,2048)),np.zeros((2048,2048)),np.zeros((2048,2048))]

for i in range(2048):
    for j in range(2048):
        if i<=2044 and j<=2044:
            result[0][i][j]=img[round(i/4)][round(j/4)][0]
        else:
            result[0][i][j]=img[math.floor(i/4)][math.floor(j/4)][0]
result[0]=result[0].astype(np.uint8)
cv.namedWindow("test",cv.WINDOW_NORMAL)
cv.imshow("test",result[0])
cv.waitKey(0)
cv.imwrite("lena_nearest.bmp",result[0])

for i in range(2048):
    for j in range(2048):
        if i<2044 and j<2044:
            l=i//4
            k=j//4
            a=i/4-i//4
            b=j/4-j//4
            result[1][i][j]=img[l][k][0]*(1-a)*(1-b)+img[l+1][k][0]*a*(1-b)+img[l][k+1][0]*(1-a)*b+img[l+1][k+1][0]*a*b
        else:
            result[1][i][j]=img[i//4][j//4][0]
result[1]=result[1].astype(np.uint8)
cv.namedWindow("test",cv.WINDOW_NORMAL)
cv.imshow("test",result[1])
cv.waitKey(0)
cv.imwrite("lena_Bilinear.bmp",result[1])

def weight(x):
    a=-0.5
    x=abs(x)
    if x<=1:
        return  (a+2)*x**3-(a+3)*x**2+1
    if x<2:
        return a*x**3-5*a*x**2+8*a*x-4*a
    return 0

for i in range(2048):
    for j in range(2048):
        if i>4 and i<2040 and j>4 and j<2040:
            for k in range(4):
                for w in range(4):
                    result[2][i][j]+=img[i//4+k-1][j//4+w-1][0]*weight(i/4-(i//4+k-1))*weight(j/4-(j//4+w-1))
        else:
            result[2][i][j]=img[i//4][j//4][0]
result[2]=result[2].astype(np.uint8)
cv.namedWindow("test",cv.WINDOW_NORMAL)
cv.imshow("test",result[2])
cv.waitKey(0)
cv.imwrite("lena_Bicubic.bmp",result[2])
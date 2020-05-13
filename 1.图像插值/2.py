import cv2 as cv
import numpy as np
img=cv.imread("lena.bmp")
print(img.shape)
def trans(img,depth):      #给出图像矩阵img，返回图像深度为depth的结果矩阵
    result=np.rint((2**depth-1)*(img/np.max(img)))
    result=result.astype(np.uint8)
    return result*(2**(8-depth))
for i in range(1,9):        #循环生成8个不同灰度级的图像
    cv.imshow("img"+str(i),trans(img,i))
    cv.waitKey(0)
img=np.rint(img/np.max(img))#对灰度级为1即黑白图像另行处理
img=img.astype(np.uint8)*255
cv.imshow("img1",img)
cv.waitKey(0)
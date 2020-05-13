import cv2 as cv
import numpy as np
import math
img=cv.imread("lena.bmp")
ave=np.average(img)
print("average:",ave)
s=np.float(0)
size=img.shape
for i in range(size[0]):
    for j in range(size[1]):
        s+=(img[i][j][0]-ave)**2
print("variance:",s/(size[0]*size[1]))
print("standard deviation",math.sqrt(s/(size[0]*size[1])))
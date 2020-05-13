import cv2 as cv
import numpy as np
import math
#np.set_printoptions(threshold=np.inf)

imgA=cv.imread("Image A.jpg")
imgB=cv.imread("Image B.jpg")
point_A=np.array([[1448,1694,1756,383,2290,2035,2150],[1308,1198,2744,2516,933,2693,1968],[1,1,1,1,1,1,1]])
point_B=np.array([[1042,1252,1708,323,1761,1966,1890],[1077,907,2387,2519,498,2265,1535],[1,1,1,1,1,1,1]])
H=np.dot(point_B,np.linalg.pinv(point_A))
print(H)
shape_A=imgA.shape
shape_B=imgB.shape
result,mistake=np.zeros(shape_A),np.zeros(shape_A)
result=result.astype(np.uint8)
mistake=mistake.astype(np.uint8)
for i in range(shape_A[0]):
    for j in range(shape_A[1]):
        address=np.dot(H,np.array([i,j,1]))
        address=address.astype(np.int)
        if address[0]>0 and address[0]<shape_B[0] and address[1]>0 and address[1]<shape_B[1]:
            result[i][j]=imgB[address[0]][address[1]]
            mistake[i][j]=imgA[i][j]-result[i][j]
cv.namedWindow("test",cv.WINDOW_NORMAL)
cv.imshow("test",mistake)
cv.waitKey(0)
cv.imwrite("result.jpg",result)
cv.imwrite("mistake.jpg",mistake)

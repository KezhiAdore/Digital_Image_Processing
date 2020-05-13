import noise
import filt
from cv2 import cv2 as cv

img=cv.imread("lena.bmp",cv.IMREAD_UNCHANGED)
# fuzzy_img=noise.gauss_noise(img,0,25)                 #加入高斯噪声
# fuzzy_img=noise.spicedsalt_noise(img,0.1,0.1)         #加入椒盐噪声
fuzzy_img=noise.move_noise(img,0.03,0.03,1)             #加入运动模糊
# filt_img=filt.gaussian(fuzzy_img,50)                  #高斯低通滤波
# filt_img=filt.butterwroth(fuzzy_img,3,100)            #巴特沃斯低通滤波
# filt_img=filt.arithmetic_mean_filt(fuzzy_img,n)       #算数平均滤波
# filt_img=filt.geometric_mean_filt(fuzzy_img,n)        #几何平均滤波
# filt_img=filt.Invertedharmonic_mean_filt(fuzzy_img,n,-1)  #谐波均值滤波
# filt_img=filt.Invertedharmonic_mean_filt(fuzzy_img,n,2)    #逆谐波均值滤波
# filt_img=filt.med_filt(fuzzy_img,n)               #中值滤波
# filt_img=filt.max_filt(fuzzy_img,n)               #最大值值滤波
# filt_img=filt.min_filt(fuzzy_img,n)               #最小值值滤波
# filt_img=filt.mid_filt(fuzzy_img,n)               #中点滤波
# filt_img=filt.alpha_filt(fuzzy_img,n,n//2)        #修正alpha滤波器
# filt_img=filt.adaptive_mean(fuzzy_img,n,25*25)    #自适应局部滤波
# filt_img=filt.adaptive_mid(fuzzy_img,n)           #自适应中值滤1101波

filt_img=filt.Minimum_mean_variance_filt(fuzzy_img,5,0.03,0.03,1)
cv.imwrite("filt_tmp"+str(n)+".bmp",filt_img)
cv.imwrite("fuzzy_tmp.bmp",fuzzy_img)
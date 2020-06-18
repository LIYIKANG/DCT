# DCT
 import numpy as np
from scipy import ndimage
import cv2
import cv2 as cv
import random

kernel_3 = np.array([[-1,-1,-1],
                      [-1,8,-1],
                      [-1,-1,-1]]) 
					  


kernel_5 = np.array([[-1,-1,-1,-1,-1],
                    [-1,1,2,1,-1],
                    [-1,2,4,2,-1],
                    [-1,1,2,1,-1],
                    [-1,-1,-1,-1,-1]])
					  
#a = np.arange(256).reshape((16,16))
#print("ori data: \n{}".format(a))
img = cv2.imread(r'C:\Users\Owner\Desktop\so\sample.png',0)

#print(img.shape)
img1 = img.astype(np.float32)
m,n = img.shape
#print(m,n)

# Y = np.zeros(256).reshape((16,16))

hdata = np.vsplit(img,n/8) # 垂直分成高度度为8 的块
for i in range(0, n//8):
     blockdata = np.hsplit(hdata[i],m/8) 
     #垂直分成高度为8的块后,在水平切成长度是8的块, 也就是8x8 的块
     for j in range(0, m//8):
         b = random.randint(-2,2)
         block = blockdata[j]+b
         #print("block[{},{}] data \n{}".format(i,j,blockdata[j]))
         Yb = cv2.dct(block.astype(np.float))
         #print("dct data\n{}".format(Yb))
         iblock = cv2.idct(Yb)
         #print("idct data\n{}".format(iblock))
img_dct = cv2.dct(img1)  # 使用dct获得img的频域图像
k3=ndimage.convolve(img_dct,kernel_3)
k5=ndimage.convolve(img_dct,kernel_5)
blurred = cv.GaussianBlur(img_dct,(11,11),0)
#g_hpf = img_dct - blurred

img_recor2 = cv2.idct(img_dct)  # 使用反dct从频域图像恢复出原图像(有损)
#print(img_dct)
#print(img_recor2)
cv2.imshow("3*3",k3)
cv2.imshow("5*5",k5)
cv2.imshow("sample1", img_dct)

#cv2.imshow("sample01", img_recor2)

cv2.waitKey(0)


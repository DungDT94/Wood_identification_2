import numpy as np
from skimage import feature
import cv2
import os
path = 'D:/xu li anh van go/woodentify-master/imgs/cs_sycamore'
new_path = 'D:/xu li anh van go/woodentify-master/imgs/cs_sycamore_out'
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):  
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method = 'default')
        return lbp
'''
img = cv2.imread("D:/xu li anh van go/woodentify-master/imgs/cs_sycamore/20170925_151112.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lbp =  LocalBinaryPatterns(3,3)
lbp_img = lbp.describe(img_gray)


cv2.imshow('lbp',lbp_img )
cv2.waitKey(0)
cv2.imwrite( 'D:/xu li anh van go/woodentify-master/imgs/cs_sycamore/2.jpg', lbp_img)'''



for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        filename_path = os.path.join(path, filename)
        filename_full_path = filename_path.replace('\\', '/')
        
        
        img = cv2.imread(filename_full_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        lbp =  LocalBinaryPatterns(8,1)
        lbp_img = lbp.describe(image = img_gray)
        
       # lbp = feature.local_binary_pattern(image=img_gray, P=8, R=1, method='default')
        
        fullpath = os.path.join(new_path, filename)
        full_path_final = fullpath.replace('\\', '/')
        cv2.imwrite( full_path_final, lbp_img)
    



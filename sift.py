import cv2
import numpy as np 
import os

path = 'D:/xu li anh van go/woodentify-master/imgs/cs_sycamore'
new_path = 'D:/xu li anh van go/woodentify-master/imgs/cs.sycamore_out_sift'
def sift(img):

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(img_gray, None)
    img=cv2.drawKeypoints(img_gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

    
for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        filename_path = os.path.join(path, filename)
        filename_full_path = filename_path.replace('\\', '/')
        img = cv2.imread(filename_full_path)
        img_sift = sift(img)
   
       # lbp = feature.local_binary_pattern(image=img_gray, P=8, R=1, method='default')
        fullpath = os.path.join(new_path, filename)
        full_path_final = fullpath.replace('\\', '/')
        cv2.imwrite( full_path_final, img_sift)
    
'''

img = cv2.imread('D:/xu li anh van go/woodentify-master/imgs/cs_sycamore/20170925_151112.jpg')   
img_gray =   cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(img_gray, None)
img=cv2.drawKeypoints(img_gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('D:/xu li anh van go/code/image_sift/sift_keypoints.jpg', img)'''
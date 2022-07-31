import cv2
import numpy as np 
import os

path = 'D:/xu li anh van go/woodentify-master/imgs/cs_sycamore'
new_path = 'D:/xu li anh van go/woodentify-master/imgs/cs_sycamore_out_suft'
def suft(img):

    img_gray =   cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF_create(400)
    
    surf.setHessianThreshold(50000)
    kp, des = surf.detectAndCompute(img_gray,None)
    img = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    
    return img

def writing_file(path, new_path):
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            filename_path = os.path.join(path, filename)
            filename_full_path = filename_path.replace('\\', '/')
            img = cv2.imread(filename_full_path)

            img_suft = suft(img)

            fullpath = os.path.join(new_path, filename)
            full_path_final = fullpath.replace('\\', '/')
            cv2.imwrite( full_path_final, img_suft)


writing_file(path , new_path)
        


        
        
        
       
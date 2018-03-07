from sklearn.externals import joblib
import cv2
import numpy as np
from scipy import misc
lr = joblib.load('Neural_ocr.sav')

kernel = np.ones((3,3),np.uint8)

img = cv2.imread('all10.jpg')
img = np.array(img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,img1 = cv2.threshold(img , 100 , 255 , cv2.THRESH_BINARY) # Take care of the inv in thresholding
img1 = cv2.dilate(img1,kernel,iterations = 1)
# cv2.imshow('frame' , img1)
# cv2.waitKey(0)
img2 = img1.copy()
_,cnts,_ = cv2.findContours(img2 , cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in cnts:
    (x,y,w,h) = cv2.boundingRect(cnt)
    img3 = img[y:y+h,x:x+w]
    img4 = cv2.resize(img3, (20,20) )
    img5 = img4.reshape(-1,400)
    result = lr.predict(img5)
    print result
    cv2.imshow('frame' , img4)
    cv2.waitKey(0)
# print cnts
cv2.destroyAllWindows()











 



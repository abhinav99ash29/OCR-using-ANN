from sklearn.externals import joblib
import cv2
import numpy as np
from scipy import misc
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPClassifier
import pandas as pd

i=0
lr =  LogisticRegression()
train = []
kernel = np.ones((3,3),np.uint8)
kernel1 = np.ones((1,2),np.uint8)

label = []
inp = 0




def im_process(img,inp):
    img = np.array(img) 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     img = cv2.GaussianBlur(img,(3,3),0)##################################################################
    _,img1 = cv2.threshold(img , 100 , 255 , cv2.THRESH_BINARY)
    img1 = cv2.dilate(img1,kernel,iterations = 1)
#     img1 = cv2.erode(img1,kernel1,iterations = 1)
    img2 = img1.copy()
    _,cnts,_ = cv2.findContours(img2 , cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in cnts:
        (x,y,w,h) = cv2.boundingRect(cnt)
        img3 = img1[y:y+h,x:x+w]
        img4 = cv2.resize(img3, (20,20) )
    #     inp  = int(raw_input('Enter label'))
        img5 = img4.reshape(-1,400)
        train.append(img5)
        label.append(inp)
    return None


# k = np.arange(10)
# train_labels = np.repeat(k,1)[:,np.newaxis]


images = ['zero','one','two','three','four','five','six','seven','eight','nine']
for image in images:
    
    filep = 'Digi_temp/'+image+'.png'
    img = cv2.imread(filep)
    cv2.imshow('frame' , img)
    cv2.waitKey(0)
    im_process(img, inp)
    
    inp = inp+1
tr = np.array(train).reshape(500,400)
label = np.array(label).reshape(500,-1)
lr.fit(tr,label)
filename = 'OCR_Log_reg.sav'
joblib.dump(lr, filename)



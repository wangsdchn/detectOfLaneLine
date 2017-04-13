import os
import numpy as np
import cv2
path='E:/wsd/detectOfLaneLine'
files=os.listdir(path)
imgs=[]
for f in files:
    fs=path+"/"+f
    if fs[-3:] not in '.jpg' and fs[-3:] not in '.png':
        continue    
    #print(fs)
    imgs.append(fs)
i=0
for img in imgs:
    if not os.path.isfile(img):
        continue
    print(img)
    dst=path+'/'+str(i)+'.bmp'
    i=i+1
    src=cv2.imread(img.encode('u8').decode('gbk'))
    cv2.imwrite(dst,src)
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:38:01 2017
This is for 武汉巡检
@author: WSD
"""

import numpy as np
import cv2

#reload(sys)  
#sys.setdefaultencoding('utf-8') 
def ZhizhengDetect(src):
    if(src.ndim==3):
        gray=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    else:
        gray=src.copy()
    cols,rows=src.shape[:2]
    thresh=0
    thresh,dst=cv2.threshold(gray,thresh,255,cv2.THRESH_OTSU)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel_1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    img2=cv2.dilate(dst,kernel_1)
    #img2=cv2.dilate(img2,kernel_1)
    cv2.imshow('dst',img2)
    dst1=img2.copy()
    contours,hieracy=cv2.findContours(dst1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(src,contours,-1,(0,255,0),2)
    size=np.size(contours)
    for i in range(size):
        (x,y),radius=cv2.minEnclosingCircle(contours[i])
        if x<cols/3 or x>cols*2/3 or y<rows/3 or y>rows*2/3:
            continue
        area=radius**2*np.pi
        if area<10 or area>5000:
            continue
        x_final,y_final,radius_final=x,y,radius+15
        center=(int(x_final),int(y_final))
        r=int(radius_final)
        cv2.circle(img2,center,r,255,-1)
    img2=cv2.dilate(img2,kernel_1)    
    img=cv2.erode(img2,kernel_1)
    img=cv2.erode(img,kernel_1)
    cv2.imshow('img',img)
    contours,hieracy=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(src,contours,0,(0,255,0),2)
    cv2.imshow('src',src)
    cv2.waitKey(0)
    size=np.size(contours)
    print (size)
    for i in range(size):
        area=cv2.contourArea(contours[i])
        if area<200 or area>150000:
            continue
        rect=cv2.minAreaRect(contours[i])
        box=cv2.cv.BoxPoints(rect)
        box=np.int0(box)
        
        if box[1,0]<cols/5 or box[3,0]>cols*14/15:
            continue
        if box[2,1]<rows/15 or box[0,1]>rows*14/15:
            continue
        w=np.sqrt(np.power((box[0,0]-box[1,0]),2)+np.power((box[0,1]-box[1,1]),2))
        h=np.sqrt(np.power((box[1,0]-box[2,0]),2)+np.power((box[1,1]-box[2,1]),2))
        if h<5*w and h>0.2*w:
            continue
            
        [vx,vy,x,y] = cv2.fitLine(contours[i],cv2.cv.CV_DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(src,(cols-1,righty),(0,lefty),(0,0,255),2)
        print (np.arctan(-vy/vx)*180/np.pi)
        cv2.drawContours(src,[box],0,(255,0,0),1)        
        cv2.imshow('contours',src) 
        k=cv2.waitKey(0)&0xFF
        if k==27:
            break
      
    

if __name__ == '__main__':
    imgpath="F:/download/武汉巡检记录2016/9.8上午巡检记录/"
    imgname='表11___1___2016_9_8_10_4_41/0___srcImg___eCircle_homo_filter.jpg'
    img=imgpath+imgname
    #img= 'F:/ime/1test/我.jpg'
    src=cv2.imread(img.decode('u8').encode('gbk'))
    #ZhizhengDetect(src)
    b,g,r=src[:,:,0],src[:,:,1],src[:,:,2]
    src1=np.dstack([b,g,r])
    cv2.imwrite(imgname,src)
    cv2.destroyAllWindows()
# Detect Of Lane Line
# By Shudong Wang, 2017/04/11
import numpy as np
import cv2

def detect(src):
    if src.ndim==3:
        gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    else:
        gray=src
    rows,cols=src.shape[:2]
    print(cols,rows)
    thresh=250
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,5),(0,2))
    while thresh>=50:
        thresh,binImg=cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)
        thresh-=5
        img0=cv2.erode(binImg,kernel,2)
        binImg=cv2.dilate(img0,kernel,2)
        img,contours,hieracy=cv2.findContours(binImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        size=np.size(contours)
        for i in range(size):
            rect=cv2.minAreaRect(contours[i])
            box=cv2.boxPoints(rect)
            box=np.int0(box)
            
            if box[0,0]<cols/20 or box[0,0]>cols*19/20:
                continue
            #if box[0,1]<rows*3/5:
            #    continue
            w=np.sqrt(np.power((box[0,0]-box[1,0]),2)+np.power((box[0,1]-box[1,1]),2))
            h=np.sqrt(np.power((box[1,0]-box[2,0]),2)+np.power((box[1,1]-box[2,1]),2))
            if h<10*w and h>0.1*w:
                continue
            if w*h<200:
                continue            
            [vx,vy,x,y] = cv2.fitLine(contours[i],cv2.DIST_L2,0,0.01,0.01)
            if vx==0:
                slidRitio=0xFFFF
            else:
                slidRitio=vy/vx
            if slidRitio<0.25 and slidRitio>-0.25:
                continue
            print(box)
            lefty = int((-y/slidRitio) + x)
            righty = int(((rows-y)/slidRitio)+x)
            print(lefty,righty,x,y)
            cv2.line(src,(righty,rows),(lefty,0),(0,0,255),2)
        cv2.imshow('gray',gray)
        cv2.imshow('src',src)
        cv2.imshow('binImg',binImg)
        k=cv2.waitKey(0)&0xFF
        if k==27:
            break
    
if __name__=='__main__':
    imgPath='./laneLine.png'
    src=cv2.imread(imgPath)
    detect(src)
    
    cv2.destroyAllWindows()

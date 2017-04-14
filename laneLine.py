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
    roi=gray[rows//2:rows,:]
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,5),(0,2))
    while thresh>=150:
        thresh,binImg=cv2.threshold(roi,thresh,255,cv2.THRESH_OTSU)
        thresh=5
        img0=cv2.erode(binImg,kernel,2)
        binImg=cv2.dilate(img0,kernel,2)
        img,contours,hieracy=cv2.findContours(binImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        size=np.size(contours)
        print(size)
        for i in range(size//4-1):
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
            if w*h>cols*rows/20:
                continue
            [vx,vy,x,y] = cv2.fitLine(contours[i],cv2.DIST_L2,0,0.01,0.01)
            if vx==0:
                slidRitio=0xFFFF
            else:
                slidRitio=vy/vx
            if slidRitio<0.25 and slidRitio>-0.25:
                continue
            maxX=0
            maxY=0
            maxSlidRitio=0
            minSlidRitio=9999
            minX=9999
            minY=9999
            if x>cols/2:
                if minX>x:
                    minX=x
                    minSlidRitio=slidRitio
                    minY=y
            else:
                if maxX<x:
                    maxX=x
                    maxSlidRitio=slidRitio
                    maxY=y
            cv2.imshow('binImg',binImg)
            k=cv2.waitKey(0)&0xFF
            if k==27:
                break
    #minLefty = int((-minY/minSlidRitio) + minX)
    #minRighty = int(((rows//2-minY)/minSlidRitio)+minX)
    #maxLefty = int((-maxY/maxSlidRitio) + maxX)
    #maxRighty = int(((rows//2-maxY)/maxSlidRitio)+maxX)
    #cv2.line(src,(minRighty,rows),(minLefty,rows//2),(0,0,255),2)
    #cv2.line(src,(maxRighty,rows),(maxLefty,rows//2),(0,0,255),2)
    cv2.imshow('src',src)
    cv2.imshow('binImg',binImg)
    k=cv2.waitKey(0)&0xFF
    #if k==27:
    #    break

def imgPerspective(src):
    roi=src[270:445,50:766]
    cv2.imshow('roi',roi)
    rows,cols=roi.shape[:2]
    origin_pts=np.float32([[cols//3,0],[cols*2//3,0],[0,rows-1],[cols-1,rows-1]]) #(x,y)
    destiny_pts=np.float32([[0,0],[300,0],[0,300],[300,300]])
    transform_mat=cv2.getPerspectiveTransform(origin_pts,destiny_pts)
    transform_mat=np.array(transform_mat)
    print(transform_mat)
    dst=src
    dst=cv2.warpPerspective(roi,transform_mat,(cols,rows))
    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    
    
    
if __name__=='__main__':
    imgPath='./0.bmp'
    src=cv2.imread(imgPath)
    #detect(src)
    imgPerspective(src)
    cv2.destroyAllWindows()

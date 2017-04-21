# Detect Of Lane Line
# By Shudong Wang, 2017/04/11
import numpy as np
import cv2

def detect(src):
    if src.ndim==3:
        gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    else:
        gray=src
    rects=[]
    rows,cols=src.shape[:2]
    thresh=250
    roi=gray[rows//2:rows-50,:]
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,1),(1,0))

    thresh,binImg=cv2.threshold(roi,thresh,255,cv2.THRESH_OTSU)
    thresh=5
    img0=cv2.erode(binImg,kernel,2)
    binImg=cv2.dilate(img0,kernel,2)
    img,contours,hieracy=cv2.findContours(binImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    size=len(contours)
    print(size)
    for i in range(size):
        rect=cv2.minAreaRect(contours[i])
        box=cv2.boxPoints(rect)
        box=np.int0(box)

        w=np.sqrt(np.power((box[0,0]-box[1,0]),2)+np.power((box[0,1]-box[1,1]),2))
        h=np.sqrt(np.power((box[1,0]-box[2,0]),2)+np.power((box[1,1]-box[2,1]),2))
        if h<4*w and h>0.25*w:
            continue
        if w*h<100:
            continue
        if w*h>cols*rows/10:
            continue
        [vx,vy,x,y] = cv2.fitLine(contours[i],cv2.DIST_L2,0,0.01,0.01)
        if vx==0:
            slidRitio=0xFFFF
        else:
            slidRitio=vy/vx
        if slidRitio<0.35 and slidRitio>-0.35:
            continue
        up = int(((-y)/slidRitio) + x)
        low = int(((rows//2-y)/slidRitio)+x)
        rects.append([up,(rows-1)//2,low,rows-1])
        cv2.line(src,(low,rows-1),(up,((rows-1)//2)),(0,0,255),2)
    #print(rects)
    cv2.imshow('src',src)
    cv2.imshow('binImg',binImg)
    #cv2.waitKey(0)

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
    dst=cv2.warpPerspective(roi,transform_mat,(300,300))
    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    
    
"""
     ----------->  X
    |
    |
    |
    Y
"""
if __name__=='__main__':
    #imgPath='./imgs/0.bmp'
    #src=cv2.imread(imgPath)
    videoPath='./lane.avi'
    video=cv2.VideoCapture(videoPath)
    if video.isOpened():
        while True:
            ret,src=video.read()
            if ret==True:
                detect(src)
                #cv2.imshow('video',src)
            else:
                break
            
            if cv2.waitKey(30)&0xffff==27:
                break
    
    #imgPerspective(src)
    cv2.destroyAllWindows()

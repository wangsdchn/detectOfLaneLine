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
    thresh=250
    rects=[0,0,0,0]
    roi=gray[rows//2:rows-50,cols//8:cols*7//8]
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,1),(1,0))

    thresh,binImg=cv2.threshold(roi,thresh,255,cv2.THRESH_OTSU)
    thresh,binImg=cv2.threshold(roi,thresh+10,255,cv2.THRESH_BINARY)
    img0=cv2.erode(binImg,kernel,1)
    binImg=cv2.dilate(img0,kernel,1)
    img,contours,hieracy=cv2.findContours(binImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    size=len(contours)
    k=0
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
        if slidRitio<0.3 and slidRitio>-0.3:
            continue
        if (box[0,0]>cols/2+100 or box[0,0]>cols/2+100) and np.abs(slidRitio)>1.2:
               continue
        #print(box[0,0],slidRitio)
        #print(box[0,0])
        up = int(((rows//10-y)/slidRitio) + x)
        low = int(((rows//2-y)/slidRitio)+x)
        k += 1
        if slidRitio<0:
            rects[0]=up
            rects[1]=low
        else:
            rects[2]=up
            rects[3]=low
    
    #cv2.imshow('binImg',binImg)
    return rects,k

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
def videoDetect(videoPath):
    video=cv2.VideoCapture(videoPath)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 30.0, (856,480))
    
    kalman = cv2.KalmanFilter(8,4,0)
    
    #kalman.measurementMatrix = 1. * np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]])
    kalman.measurementMatrix = 1.*np.eye(4,8)
    kalman.transitionMatrix = 1.*np.array([[1,0,0,0,0.1,0,0,0],[0,1,0,0,0,0.1,0,0],[0,0,1,0,0,0,0.1,0],[0,0,0,1,0,0,0,0.1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
    #kalman.transitionMatrix = 1.*np.eye(8)
    kalman.processNoiseCov = 1e-5 * np.eye(8)
    kalman.measurementNoiseCov = 1e-1 * np.eye(4)
    kalman.errorCovPost = 1. * np.eye(8)
    #kalman.statePost=0.1 * np.random.randn(8, 1)

    k=0
    rects=[1,1,1,1]
    if video.isOpened():
        while True:
                        
            ret,src=video.read()
            if ret==True:
                rects,ret=detect(src)
                rows,cols=src.shape[:2]                
                if ret==2:
                    x0,x1,x2,x3=rects[:4]
                    if k==0:
                        k+=1
                    for i in range(0,len(rects),2):
                        up,low=rects[i:i+2]
                        cv2.line(src,(low+cols//8,rows-1),(up+cols//8,((rows)*3//5)),(0,0,255),2)
                    
                if k==1:
                    print(k)
                    k=2
                    kalman.statePost=np.transpose(1.*np.array([[x0,x1,x2,x3,0.1,0.1,0.1,0.1]]))
                if k>0:
                    tp = kalman.predict()
                    if ret==2:
                        measurement = np.sqrt(kalman.measurementNoiseCov[0,0])+np.random.randn(4,1)
                        measurement = np.dot(kalman.measurementMatrix, np.transpose(1.*np.array([[x0,x1,x2,x3,0,0,0,0]]))) + measurement
                        kalman.correct(measurement)
                    for i in range(0,4,2):
                        up,low=tp[i:i+2]
                        cv2.line(src,(low+cols//8,rows-1),(up+cols//8,((rows)*3//5)),(0,255,0),2)
                #cv2.imshow('video',src)
            else:
                break
            #if k<3000:
            out.write(src)
            if cv2.waitKey(20)&0xffff==27:
                break
    video.release()
    out.release()
def tracking(kalman2d,rects):
    x0,x1,x2,x3=rects[:4]
    kalman_points = []
    # Update the Kalman filter with the mouse point  
    kalman2d.update(x0,x1,x2,x3)
  
    # Get the current Kalman estimate and add it to the trajectory  
    estimated = [int (c) for c in kalman2d.getEstimate()]  
    kalman_points.append(estimated)
    return kalman_points
        
    
    
if __name__=='__main__':
    #imgPath='./imgs/0.bmp'
    #src=cv2.imread(imgPath)
    
    videoPath='./lane.avi'
    videoDetect(videoPath)
    
    #imgPerspective(src)
    cv2.destroyAllWindows()

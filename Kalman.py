import cv2
import numpy as np
class Kalman2D(object):  
    ''''' 
    A class for 2D Kalman filtering 
    '''  
  
    def __init__(self, processNoiseCovariance=1e-4, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1):  
        ''''' 
        Constructs a new Kalman2D object.   
        For explanation of the error covariances see 
        http://en.wikipedia.org/wiki/Kalman_filter 
        '''  
        # 状态空间：位置--2d,速度--2d
        self.kalman = cv2.KalmanFilter(8, 4, 0)
        self.kalman_state = np.array((8,1),np.float32)
        self.kalman_process_noise = np.array((8,1),np.float32)   
        self.kalman_measurement = np.array((4,1),np.float32)   
  
        for j in range(8):
            for k in range(8):
                self.kalmantransitionMatrix[j,k] = 0  
            self.kalman.transitionMatrix[j,j] = 1  
        #加入速度 x = x + vx, y = y + vy  
        # 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1  
        #如果把下面两句注释掉，那么位置跟踪kalman滤波器的状态模型就是没有使用速度信息  
#        self.kalman.transition_matrix[0, 2]=1  
#        self.kalman.transition_matrix[1, 3]=1  
          
        cv2.SetIdentity(self.kalman.measurement_matrix)  
        #初始化带尺度的单位矩阵  
        cv2.SetIdentity(self.kalman.processNoiseCov , cv2.RealScalar(processNoiseCovariance))  
        cv2.SetIdentity(self.kalman.measurementNoiseCov , cv2.RealScalar(measurementNoiseCovariance))  
        cv2.SetIdentity(self.kalman.error_cov_post, cv2.RealScalar(errorCovariancePost))  
  
        self.predicted = None  
        self.esitmated = None  
  
    def update(self, x0, x1,x2,x3):  
        ''''' 
        Updates the filter with a new X,Y measurement 
        '''  
  
        self.kalman_measurement[0, 0] = x0 
        self.kalman_measurement[1, 0] = x1
        self.kalman_measurement[2, 0] = x2 
        self.kalman_measurement[3, 0] = x3 
  
        self.predicted = cv2.KalmanPredict(self.kalman)  
        self.corrected = cv2.KalmanCorrect(self.kalman, self.kalman_measurement)  
  
    def getEstimate(self):  
        ''''' 
        Returns the current X,Y estimate. 
        '''  
  
        return self.corrected[0,0], self.corrected[1,0], self.corrected[2,0], self.corrected[3,0]
  
    def getPrediction(self):  
        ''''' 
        Returns the current X,Y prediction. 
        '''  
  
        return self.predicted[0,0], self.predicted[1,0], self.predicted[2,0], self.predicted[3,0]
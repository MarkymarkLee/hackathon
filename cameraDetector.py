import cv2
import argparse
import sys
import imutils
import time
import math
import numpy as np

#constants
CAM_TO_TAG_THRESHOLD , CENTER_X_THRESHOLD , CENTER_Y_THRESHOLD ,CENTER_ROTATE_THRESHOLD  = (10 , 200 , 100 , 10)
DISTANCE_DIFF_TO_SPEED , X_SPEED , CENTER_X_DIFF_TO_SPEED , CENTER_Y_DIFF_TO_SPEED, ROTATION_SPEED= (1 , 20 , 0.1 , 0.2 , math.pi/2)
ARUCO_DICT = {
    "DICT_4X4_50" : cv2.aruco.DICT_4X4_50 ,
    "DICT_4X4_100" : cv2.aruco.DICT_4X4_100 ,
    "DICT_4X4_250" : cv2.aruco.DICT_4X4_250 ,
    "DICT_4X4_1000" : cv2.aruco.DICT_4X4_1000 ,
    "DICT_5X5_50" : cv2.aruco.DICT_5X5_50 ,
    "DICT_5X5_100" : cv2.aruco.DICT_5X5_100 ,
    "DICT_5X5_250" : cv2.aruco.DICT_5X5_250 ,
    "DICT_5X5_1000" : cv2.aruco.DICT_5X5_1000 ,
    "DICT_6X6_50" : cv2.aruco.DICT_6X6_50 ,
    "DICT_6X6_100" : cv2.aruco.DICT_6X6_100 ,
    "DICT_6X6_250" : cv2.aruco.DICT_6X6_250 ,
    "DICT_6X6_1000" : cv2.aruco.DICT_6X6_1000 ,
    "DICT_7X7_50" : cv2.aruco.DICT_7X7_50 ,
    "DICT_7X7_100" : cv2.aruco.DICT_7X7_100 ,
    "DICT_7X7_250" : cv2.aruco.DICT_7X7_250 ,
    "DICT_7X7_1000" : cv2.aruco.DICT_7X7_1000 
}
class CameraDetector:
    def __init__(self, cap, arucoDict, arucoParams, FPS):
        self.cap = cap
        self.arucoDict = arucoDict
        self.arucoParams = arucoParams
        self.corners = dict()
        self.ID_Recorded = []
        self.rvecs = None
        self.tvecs = None
        self.center = dict()
        self.cameraWidth = 1000
        self.cameraCenter = (1000/2, cap.get(4)/cap.get(3)*1000/2)
        self.cameraYawDeg = 0
        self.edge = dict()
        self.standardDistance = {492: 40}
        # self.standardCenter = {}
        self.X_speed = {}
        self.Y_speed = {}
        self.Z_speed = {}
        self.pitch_speed = {}
        self.roll_speed = {}
        self.yaw_speed = {}
        self.mtx = []
        self.dist = []
        self.rvec = []
        self.tvec = []
        self.FPS = FPS
    
    def saveCurrentInfo(self , markerCorner , markerID):
        self.corners[markerID] = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = self.corners[markerID]
        topRight = (int(topRight[ 0 ]), int(topRight[ 1 ]))
        bottomRight = (int(bottomRight[ 0 ]), int(bottomRight[ 1 ]))
        bottomLeft = (int(bottomLeft[ 0 ]), int(bottomLeft[ 1 ]))
        topLeft = (int(topLeft[ 0 ]), int(topLeft[ 1 ]))
        self.center[markerID] = [int((topLeft[ 0 ] + bottomRight[ 0 ]) /  2.0 ),
                        int((topLeft[ 1 ] + bottomRight[ 1 ]) /  2.0 )]
        x_length = abs(topRight[ 0 ] - topLeft[ 0 ])
        y_length = abs(topRight[ 1 ] - topLeft[ 1 ])
        self.edge[markerID] = ((x_length*x_length) + (y_length*y_length))**(0.5)
        return topRight, bottomRight, bottomLeft, topLeft
    
    def cameraCalibration(self):
        cv_file = cv2.FileStorage("webcam.yaml", cv2.FILE_STORAGE_READ)
        self.mtx = cv_file.getNode("camera_matrix").mat()
        self.dist = cv_file.getNode("dist_coeff").mat()
        cv_file.release()
    
    def calculate_X_speed(self , markerID):
        index = self.ID_Recorded.index(markerID)
        distance = self.tvec[index][0][2]*100
        x_speed, y_speed = 0, 0
        if distance - self.standardDistance[markerID] < (-1) * CAM_TO_TAG_THRESHOLD:
            # self.X_speed[markerID] = (CAM_TO_TAG_THRESHOLD-distance) * DISTANCE_DIFF_TO_SPEED
            x_speed += X_SPEED * math.cos(self.cameraYawDeg * math.pi / 180)
            y_speed += X_SPEED * math.sin(self.cameraYawDeg * math.pi / 180)
            # print( "[INFO] Marker {} moved in positive X direction , speed : '{}'".format(markerID , self.X_speed[markerID]))
        elif distance - self.standardDistance[markerID] > CAM_TO_TAG_THRESHOLD:
            # self.X_speed[markerID] = (CAM_TO_TAG_THRESHOLD-distance) * DISTANCE_DIFF_TO_SPEED
            x_speed += X_SPEED * (-1) * math.cos(self.cameraYawDeg * math.pi / 180)
            y_speed += X_SPEED * (-1) * math.sin(self.cameraYawDeg * math.pi / 180)
            # print( "[INFO] Marker {} moved in positive X direction , speed : '{}'".format(markerID , self.X_speed[markerID]))
        
        return x_speed, y_speed
        
    def calculate_Y_speed(self , markerID):
        
        x_speed, y_speed = 0, 0
        diffCenter = self.center[markerID][0] - self.cameraCenter[0]
        # print( "diff center : {}" .format(diffCenter))
        if diffCenter > CENTER_X_THRESHOLD:
            y_speed += (diffCenter-CENTER_X_THRESHOLD) * CENTER_X_DIFF_TO_SPEED * math.cos(-self.cameraYawDeg * math.pi / 180) * (-1)
            x_speed += (diffCenter-CENTER_X_THRESHOLD) * CENTER_X_DIFF_TO_SPEED * math.sin(-self.cameraYawDeg * math.pi / 180) * (-1)
            # print( "[INFO] Marker {} moved in negative Y direction , speed : '{}'".format(markerID , self.Y_speed[markerID]))
        elif diffCenter < (-1)*CENTER_X_THRESHOLD:
            y_speed += (diffCenter+CENTER_X_THRESHOLD) * CENTER_X_DIFF_TO_SPEED * math.cos(-self.cameraYawDeg * math.pi / 180) * (-1)
            x_speed += (diffCenter+CENTER_X_THRESHOLD) * CENTER_X_DIFF_TO_SPEED * math.sin(-self.cameraYawDeg * math.pi / 180) * (-1)
            # print( "[INFO] Marker {} moved in positive Y direction , speed : '{}'".format(markerID , self.Y_speed[markerID]))
        return x_speed, y_speed

    def calculate_Z_speed(self , markerID):
        diffCenter = self.center[markerID][1] - self.cameraCenter[1]
        # print( "diff center : {}" .format(diffCenter))
        if diffCenter > CENTER_Y_THRESHOLD:
            self.Z_speed[markerID] = (diffCenter-CENTER_Y_THRESHOLD) * CENTER_Y_DIFF_TO_SPEED * (-1)
            # print( "[INFO] Marker {} moved in negative Y direction , speed : '{}'".format(markerID , self.Y_speed[markerID]))
        elif diffCenter < (-1)*CENTER_Y_THRESHOLD:
            self.Z_speed[markerID] = (diffCenter+CENTER_Y_THRESHOLD) * CENTER_Y_DIFF_TO_SPEED * (-1)
            # print( "[INFO] Marker {} moved in positive Y direction , speed : '{}'".format(markerID , self.Y_speed[markerID]))
        else:
            self.Z_speed[markerID] =  0

    def calculate_Rotate_speed(self , markerID):
        index = self.ID_Recorded.index(markerID)
        
        # degx = self.rvec[index][0][0]/math.pi*180
        # degy = self.rvec[index][0][1]/math.pi*180
        # degz = self.rvec[index][0][2]/math.pi*180*90/104
        
        # Euler angle to rotation matrix
        R = np.zeros((3,3),dtype=np.float64)
        cv2.Rodrigues(self.rvec[index],R)
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular:#偏航，俯仰，滚动
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        # # 偏航，俯仰，滚动换成角度
        rx = x * 180.0 / math.pi
        ry = y * 180.0 / math.pi
        rz = z * 180.0 / math.pi
        # print("pitch angle:" , rx, end = ' ')
        # print("yaw angle:" , ry ,end = ' ')
        # print("roll angle:" , rz)
        
        pitch_angle = (rx + 360) % 360
        if (180 - CENTER_ROTATE_THRESHOLD < pitch_angle < 180 + CENTER_ROTATE_THRESHOLD):
            self.pitch_speed[markerID] = 0
        else:
            self.pitch_speed[markerID] = ROTATION_SPEED * (-1 if pitch_angle >= 180 else 1)
            self.Z_speed[markerID] = 0
            
        yaw_angle = ry
        # print(ry)
        if (-CENTER_ROTATE_THRESHOLD < yaw_angle < CENTER_ROTATE_THRESHOLD):
            self.yaw_speed[markerID] = 0
        else:
            self.yaw_speed[markerID] = ROTATION_SPEED * (1 if yaw_angle >= 0 else -1)
            self.Y_speed[markerID] = 0
            self.X_speed[markerID] = 0
        self.cameraYawDeg += self.yaw_speed[markerID] / self.FPS
            
        roll_angle = rz
        if (-CENTER_ROTATE_THRESHOLD < roll_angle < CENTER_ROTATE_THRESHOLD):
            self.roll_speed[markerID] = 0
        else:
            self.roll_speed[markerID] = ROTATION_SPEED * (1 if roll_angle >= 0 else -1)
        
    
        
    def drawBoundingBox(self , frame , markerID , topLeft, topRight, bottomRight, bottomLeft):
        cv2.line(frame, topLeft, topRight, ( 0 , 255 , 0 ),  2 )
        cv2.line(frame, topRight, bottomRight, ( 0 , 255 , 0 ),  2 )
        cv2.line(frame, bottomRight, bottomLeft, ( 0 , 255 , 0 ),  2 )
        cv2.line(frame, bottomLeft, topLeft, ( 0 , 255 , 0 ),  2 )
        cv2.circle(frame, (self.center[markerID][0], self.center[markerID][1]),  4 , ( 0 , 0 , 255 ), - 1 )
        cv2.putText(frame, str(markerID), (topLeft[ 0 ] , topLeft[ 1 ] -  15 ), cv2.FONT_HERSHEY_SIMPLEX,  0.5 , ( 0 , 255 , 0 ),  2 )
    
    def getMovementFromFrame(self , showFrame):
        # print("in getMovementFromFrame")
        ret, frame = self.cap.read()
        self.cameraCalibration()
        # gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return False , {} , {} , {} , {} , []
        frame , x_speed , y_speed , z_speed , pitch_speed , yaw_speed, roll_speed, id_recorded= self.detect(frame)
        if showFrame:
            cv2.imshow( 'frame' , frame)
        print( "x_speed : {} , y_speed : {} , z_speed : {} , yaw_speed : {} id_recorded : {}"
                .format(x_speed , y_speed , z_speed , yaw_speed , id_recorded))
        return True , x_speed , y_speed , z_speed , yaw_speed ,pitch_speed, id_recorded  

    def videoStream(self , showFrame):
        while True:
            # show the output image
            ret, frame = self.cap.read()
            # print(frame.shape[1], frame.shape[0])
            self.cameraCalibration()
            # gray
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame , x_speed , y_speed , z_speed , pitch_speed , yaw_speed, roll_speed, id_recorded= self.detect(frame)
            print( "x_speed : {} , y_speed : {} , z_speed : {} , yaw_speed : {} , cameraDeg : {} , id_recorded : {}"
                .format(x_speed , y_speed , z_speed , yaw_speed ,self.cameraYawDeg, id_recorded)) 
            # print(self.rvec,self.tvec)
            if showFrame:
                cv2.imshow( "Frame" , frame)
            key = cv2.waitKey( 1 ) & 0xFF
            if key == ord( "q" ):
                break
        cv2.destroyAllWindows()
        self.cap.release()

    def detect(self, frame):
        frame = imutils.resize(frame, self.cameraWidth)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.arucoDict, parameters=self.arucoParams)
        # print(corners)
        if len(corners) > 0:
            ids = ids.flatten()
            self.rvec, self.tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.035, self.mtx, self.dist)
            (self.rvec-self.tvec).any()
            
            for (markerCorner, markerID) in zip(corners, ids):
                if markerID not in self.ID_Recorded:
                    self.ID_Recorded.append(markerID)
                (topLeft, topRight, bottomRight, bottomLeft) = self.saveCurrentInfo(markerCorner , markerID)
                # check if moving in x direction
                x1, y1 = self.calculate_X_speed(markerID)

                # check if moving in y direction
                x2, y2 = self.calculate_Y_speed(markerID)
                self.X_speed[markerID], self.Y_speed[markerID] = x1+x2, y1+y2

                # check if moving in y direction
                self.calculate_Z_speed(markerID)

                # rotation
                self.calculate_Rotate_speed(markerID)
                
                
                #draw the bounding box of the ArUCo detection
                self.drawBoundingBox(frame, markerID, topLeft, topRight, bottomRight, bottomLeft)
                
        return frame , self.X_speed , self.Y_speed , self.Z_speed , self.pitch_speed, self.yaw_speed, self.roll_speed , self.ID_Recorded


def add_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument( "-t" , "--type" , type=str, default= "DICT_4X4_50" ,
        help= "type of ArUCo tag to detect" )
    args = vars(ap.parse_args())
    return args

def checkArucoDict(arucoType):
    if(ARUCO_DICT.get(arucoType, None) is None):
        print("[ERROR] AruCo tag of '{}' is not supported".format(arucoType))
        sys.exit(0)

def setup():
    # construct the argument parser and parse the arguments
    args = add_arguments()

    # check the names of aruco type fits the possible ArUco tag OpenCV supports
    checkArucoDict(args["type"])
    print( "[INFO] detecting '{}' tags...".format(args[ "type" ]))
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args[ "type" ]])
    arucoParams = cv2.aruco.DetectorParameters_create()
    return arucoDict,arucoParams



def main():
    arucoDict,arucoParams = setup()
    print( "[INFO] starting video stream..." )
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    cameraDetector = CameraDetector(cap , arucoDict, arucoParams, 60)
    cameraDetector.videoStream(True)

if __name__ == '__main__':
    main()
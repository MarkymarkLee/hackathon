import cv2
import argparse
import sys
import imutils
from imutils.video import VideoStream
import time

#constants
(EDGE_THRESHOLD , CENTER_THRESHOLD , PHI_THRESHOLD)  = (30 , 300 , 0)
(EDGE_DIFF_TO_SPEED , CENTER_DIFF_TO_SPEED , PHI_DIFF_TO_SPEED) = (2 , 0.1 , 0.1)

ap = argparse.ArgumentParser()
ap.add_argument("-t" , "--type" , type = str , default = "DICT_ARUCO_ORIGINAL" , help = "Type of AruCo tag to detect" )
args = vars(ap.parse_args())

# load the AruCo dictionary, grab the AruCo parameters, and detect the markers
ARUCO_DICT = {
    "DICT_4X4_50" : cv2.aruco.DICT_4X4_50 , "DICT_4X4_100" : cv2.aruco.DICT_4X4_100 ,
    "DICT_4X4_250" : cv2.aruco.DICT_4X4_250 , "DICT_4X4_1000" : cv2.aruco.DICT_4X4_1000 ,
    "DICT_5X5_50" : cv2.aruco.DICT_5X5_50 , "DICT_5X5_100" : cv2.aruco.DICT_5X5_100 ,
    "DICT_5X5_250" : cv2.aruco.DICT_5X5_250 , "DICT_5X5_1000" : cv2.aruco.DICT_5X5_1000 ,
    "DICT_6X6_50" : cv2.aruco.DICT_6X6_50 , "DICT_6X6_100" : cv2.aruco.DICT_6X6_100 ,
    "DICT_6X6_250" : cv2.aruco.DICT_6X6_250 , "DICT_6X6_1000" : cv2.aruco.DICT_6X6_1000 ,
    "DICT_7X7_50" : cv2.aruco.DICT_7X7_50 , "DICT_7X7_100" : cv2.aruco.DICT_7X7_100 ,
    "DICT_7X7_250" : cv2.aruco.DICT_7X7_250 , "DICT_7X7_1000" : cv2.aruco.DICT_7X7_1000 ,
    "DICT_ARUCO_ORIGINAL" : cv2.aruco.DICT_ARUCO_ORIGINAL ,
    "DICT_APRILTAG_16h5" : cv2.aruco.DICT_APRILTAG_16h5 ,
    "DICT_APRILTAG_25h9" : cv2.aruco.DICT_APRILTAG_25h9 ,
    "DICT_APRILTAG_36h10" : cv2.aruco.DICT_APRILTAG_36h10 ,
    "DICT_APRILTAG_36h11" : cv2.aruco.DICT_APRILTAG_36h11 ,
}

if(ARUCO_DICT.get(args[ "type" ], None) is None):
    print("[ERROR] AruCo tag of '{}' is not supported".format(args[ "type" ]))
    sys.exit(0)

print( "[INFO] detecting '{}' tags...".format(args[ "type" ]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args[ "type" ]])
arucoParams = cv2.aruco.DetectorParameters_create()

print( "[INFO] starting video stream..." )
vs=VideoStream(src=0).start()
time.sleep(2.0)

standardEdge = {}
standardPhi = {}
standardCenter = {}
while(True):
    X_speed , Y_speed , Z_speed = (0 , 0 , 0)
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):

            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[ 0 ]), int(topRight[ 1 ]))
            bottomRight = (int(bottomRight[ 0 ]), int(bottomRight[ 1 ]))
            bottomLeft = (int(bottomLeft[ 0 ]), int(bottomLeft[ 1 ]))
            topLeft = (int(topLeft[ 0 ]), int(topLeft[ 1 ]))
            centerX = int((topLeft[ 0 ] + bottomRight[ 0 ]) /  2.0 )
            # centerY = int((topLeft[ 1 ] + bottomRight[ 1 ]) /  2.0 )
            x_length = abs(topRight[ 0 ] - topLeft[ 0 ])
            y_length = abs(topRight[ 1 ] - topLeft[ 1 ])
            curEdge = ((x_length*x_length) + (y_length*y_length))**(0.5)
            
            # check if moving in x direction
            if standardEdge.get(markerID) is not None:
                diffEdge = curEdge - standardEdge[markerID]
                print( "diff edge : {}" .format(diffEdge))
                if diffEdge > EDGE_THRESHOLD:
                    X_speed = (diffEdge-EDGE_THRESHOLD) * EDGE_DIFF_TO_SPEED
                    print( "[INFO] Marker {} moved in positive X direction , speed : '{}'".format(markerID , X_speed))
                elif diffEdge < (-1)*EDGE_THRESHOLD:
                    X_speed = (diffEdge-EDGE_THRESHOLD) * EDGE_DIFF_TO_SPEED
                    print( "[INFO] Marker {} moved in negative X direction , speed : '{}'".format(markerID , X_speed))
            elif standardEdge.get(markerID) is None:
                standardEdge[markerID] = curEdge

            # check if moving in y direction
            if standardCenter.get(markerID) is not None:
                diffCenter = centerX - standardCenter[markerID]
                # diffCenterY = centerY - standardCenter[markerID][1]
                # diffCenter = ((diffCenterX*diffCenterX) + (diffCenterY*diffCenterY))**(0.5)
                print( "diff center : {}" .format(diffCenter))
                if diffCenter > CENTER_THRESHOLD:
                    Y_speed = (diffCenter-CENTER_THRESHOLD) * CENTER_DIFF_TO_SPEED
                    print( "[INFO] Marker {} moved in negative Y direction , speed : '{}'".format(markerID , Y_speed))
                elif diffCenter < (-1)*CENTER_THRESHOLD:
                    Y_speed = (diffCenter-CENTER_THRESHOLD) * CENTER_DIFF_TO_SPEED
                    print( "[INFO] Marker {} moved in positive Y direction , speed : '{}'".format(markerID , Y_speed))
            elif standardCenter.get(markerID) is None:
                standardCenter[markerID] = centerX

            
            # cv2.line(frame, topLeft, topRight, ( 0 , 255 , 0 ),  2 )
            # cv2.line(frame, topRight, bottomRight, ( 0 , 255 , 0 ),  2 )
            # cv2.line(frame, bottomRight, bottomLeft, ( 0 , 255 , 0 ),  2 )
            # cv2.line(frame, bottomLeft, topLeft, ( 0 , 255 , 0 ),  2 )
            # cv2.circle(frame, (cX, cY),  4 , ( 0 , 0 , 255 ), - 1 )
            # cv2.putText(frame, str(markerID), (topLeft[ 0 ] , topLeft[ 1 ] -  15 ), cv2.FONT_HERSHEY_SIMPLEX,  0.5 , ( 0 , 255 , 0 ),  2 )
            # print( "[INFO] ID: {}  Center: {},{}".format(markerID, cX, cY))

            (lastTopLeft, lastTopRight, lastBottomRight, lastBottomLeft) = corners
        
    # show the output image
    cv2.imshow( "Frame" , frame)
    key = cv2.waitKey( 1 ) & 0xFF
    if key == ord( "q" ):
        break

    # else :
    #     print( "[INFO] No AruCo markers found" )
        
cv2.destroyAllWindows()
vs.stop()
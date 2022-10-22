import cv2
import argparse
import sys
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i" , "--image" , required = True , help = "Path to the image containing AruCo tag" )
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
print("[INFO] Loading image...")
image = cv2.imread(args[ "image" ])
image = imutils.resize(image, width = 600)

if(ARUCO_DICT.get(args[ "type" ], None) is None):
    print("[ERROR] Invalid AruCo tag type")
    sys.exit(0)

print( "[INFO] detecting '{}' tags...".format(args[ "type" ]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args[ "type" ]])
arucoParams = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters = arucoParams)
# print( "[INFO] (corners , ids , rejected) ='{}' , '{}' , '{}'.".format(corners , ids, rejected))
# verify *at least* one AruCo marker was detected
if len(corners) > 0:
    ids = ids.flatten()
    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        topRight = (int(topRight[ 0 ]), int(topRight[ 1 ]))
        bottomRight = (int(bottomRight[ 0 ]), int(bottomRight[ 1 ]))
        bottomLeft = (int(bottomLeft[ 0 ]), int(bottomLeft[ 1 ]))
        topLeft = (int(topLeft[ 0 ]), int(topLeft[ 1 ]))

        cv2.line(image, topLeft, topRight, ( 0 , 255 , 0 ),  2 )
        cv2.line(image, topRight, bottomRight, ( 0 , 255 , 0 ),  2 )
        cv2.line(image, bottomRight, bottomLeft, ( 0 , 255 , 0 ),  2 )
        cv2.line(image, bottomLeft, topLeft, ( 0 , 255 , 0 ),  2 )

        cX = int((topLeft[ 0 ] + bottomRight[ 0 ]) /  2.0 )
        cY = int((topLeft[ 1 ] + bottomRight[ 1 ]) /  2.0 )
        cv2.circle(image, (cX, cY),  4 , ( 0 , 0 , 255 ), - 1 )

        cv2.putText(image, str(markerID), (topLeft[ 0 ] , topLeft[ 1 ] -  15 ), cv2.FONT_HERSHEY_SIMPLEX,  0.5 , ( 0 , 255 , 0 ),  2 )
        print( "[INFO] AruCo marker ID: {}".format(markerID))
    
    # show the output image
    cv2.imshow( "Image" , image)
    cv2.waitKey( 0 )
else :
    print( "[INFO] No AruCo markers found" )


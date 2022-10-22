import pygame
import math
from cameraDetector import CameraDetector
import cv2
import argparse
import sys
import time

pygame.init()
# constants
size = (width, height) = (500, 500)
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

screen = pygame.display.set_mode(size)
WHITE = (0,0,0)
BLUE = (11, 179, 191)
RED = (196, 24, 58)

FPS = 60

class Ball:
    def __init__(self , ballID):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.xspeed = 0
        self.yspeed = 0
        self.zspeed = 0
        self.omega = 0
        self.size = 30
        self.thetaA = 1
        self.ID = ballID

    def move(self,t):
        self.x += self.xspeed * t
        self.y += self.yspeed * t
        self.theta += self.omega * t * self.thetaA

    def reset(self):
        self.x = 0
        self.y = 0
        self.theta = 0

    def draw(self):
        center = (width/2,height/2)
        pos = (center[0]+self.x,center[1]+self.y)
        pygame.draw.circle(screen,BLUE,pos,self.size)

        end = (pos[0]+2*self.size*math.cos(self.theta-math.pi/2),pos[1]+2*self.size*math.sin(self.theta-math.pi/2))
        pygame.draw.line(screen,RED,pos,end)


def draw_window(ball):
    screen.fill(WHITE)
    ball.draw()
    pygame.display.update()

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
    cameraDetector = CameraDetector(cap , arucoDict, arucoParams)

    run = True
    showFrame = False
    ball = Ball(492)
    clock = pygame.time.Clock()
    while run:
        clock.tick(FPS)
        run , x_speed , y_speed , z_speed , omega , id_list= cameraDetector.getMovementFromFrame(showFrame)
        # print(id_list)
        if not run :
            break
        else:
            for id in id_list:
                if(ball.ID == id):
                    ball.yspeed = x_speed[id]
                    ball.yspeed = y_speed[id]
                    ball.zspeed = z_speed[id]
                    ball.omega = omega[id]
                    ball.move(1/FPS)
                    draw_window(ball)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                run = False
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_q]:
            run = False
        # pressed = pygame.key.get_pressed()
        # if pressed[pygame.K_d]:
        #     ball.xspeed += 0.01
        # elif pressed[pygame.K_a]:
        #     ball.xspeed -= 0.01
        # if pressed[pygame.K_s]:
        #     ball.yspeed += 0.01
        # if pressed[pygame.K_w]:
        #     ball.yspeed -= 0.01
        # if pressed[pygame.K_e]:
        #     ball.omega += 0.001
        # if pressed[pygame.K_q]:
        #     ball.omega -= 0.001
        # if pressed[pygame.K_r]:
        #     ball.reset()

        # ball.move(1)
        

        # draw_window(ball)

        

if __name__ == "__main__":
    main()
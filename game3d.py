from http.client import ImproperConnectionState
from ursina import *
from cameraDetector import CameraDetector
import cv2
import argparse
import time
import math


#constants
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

def average(speed_dict):
    total = 0
    count = 0
    for key in speed_dict:
        total += speed_dict[key]
        count += 1
    return total / count


class FirstPersonControllera(Entity):
    def __init__(self, **kwargs):
        self.cursor = Entity(parent=camera.ui, model='quad', color=color.pink, scale=.008, rotation_z=45)
        super().__init__()
        self.speed = 5
        self.height = 2
        self.camera_pivot = Entity(parent=self, y=self.height)

        camera.parent = self.camera_pivot
        camera.position = (0,0,0)
        camera.rotation = (0,0,0)
        camera.fov = 90
        mouse.locked = True
        self.mouse_sensitivity = Vec2(40, 40)

        self.gravity = 1
        self.grounded = False
        self.jump_height = 2
        self.jump_up_duration = .5
        self.fall_after = .35 # will interrupt jump up
        self.jumping = False
        self.air_time = 0

        self.time = time.time_ns()

        self.cd = 0

        for key, value in kwargs.items():
            setattr(self, key ,value)

        # make sure we don't fall through the ground if we start inside it
        if self.gravity:
            ray = raycast(self.world_position+(0,self.height,0), self.down, ignore=(self,))
            if ray.hit:
                self.y = ray.world_point.y


    def update(self):

        ttime = time.time_ns()

        data = self.cd.getMovementFromFrame(False,(ttime-self.time)/1000000000)
        self.time = ttime
        print(data)

        if data[0] and data[1]:

            x_speed , y_speed , z_speed ,pitch_speed ,  yaw_speed , roll_speed , theta  = data[2:]


            self.camera_pivot.rotation_x += average(pitch_speed) * 0.2 * math.pi/90
            self.camera_pivot.rotation_x = clamp(self.camera_pivot.rotation_x, -90, 90)
            # self.camera_pivot.rotation_y += average(yaw_speed) * 0.2
            self.camera_pivot.rotation_y = theta
            # theta = self.camera_pivot.rotation_y / 180 * pi + pi/2

            axspeed = average(x_speed)
            ayspeed = average(y_speed)

            self.direction = Vec3(
                self.forward * axspeed
                +self.right * ayspeed
                ).normalized()

            self.speed = sqrt(axspeed**2+ayspeed**2) * 0.05

            feet_ray = raycast(self.position+Vec3(0,0.5,0), self.direction, ignore=(self,), distance=.5, debug=False)
            head_ray = raycast(self.position+Vec3(0,self.height-.1,0), self.direction, ignore=(self,), distance=.5, debug=False)
            if not feet_ray.hit and not head_ray.hit:
                move_amount = self.direction * time.dt * self.speed

                if raycast(self.position+Vec3(-.0,1,0), Vec3(1,0,0), distance=.5, ignore=(self,)).hit:
                    move_amount[0] = min(move_amount[0], 0)
                if raycast(self.position+Vec3(-.0,1,0), Vec3(-1,0,0), distance=.5, ignore=(self,)).hit:
                    move_amount[0] = max(move_amount[0], 0)
                if raycast(self.position+Vec3(-.0,1,0), Vec3(0,0,1), distance=.5, ignore=(self,)).hit:
                    move_amount[2] = min(move_amount[2], 0)
                if raycast(self.position+Vec3(-.0,1,0), Vec3(0,0,-1), distance=.5, ignore=(self,)).hit:
                    move_amount[2] = max(move_amount[2], 0)
                self.position += move_amount

            self.position += self.direction * self.speed * time.dt

            if average(z_speed)<0:
                self.jump()


        if self.gravity:
            # gravity
            ray = raycast(self.world_position+(0,self.height,0), self.down, ignore=(self,))
            # ray = boxcast(self.world_position+(0,2,0), self.down, ignore=(self,))

            if ray.distance <= self.height+.1:
                if not self.grounded:
                    self.land()
                self.grounded = True
                # make sure it's not a wall and that the point is not too far up
                if ray.world_normal.y > .7 and ray.world_point.y - self.world_y < .5: # walk up slope
                    self.y = ray.world_point[1]
                return
            else:
                self.grounded = False

            # if not on ground and not on way up in jump, fall
            self.y -= min(self.air_time, ray.distance-.05) * time.dt * 100
            self.air_time += time.dt * .25 * self.gravity


    def jump(self):
        if not self.grounded:
            return

        self.grounded = False
        self.animate_y(self.y+self.jump_height, self.jump_up_duration, resolution=int(1//time.dt), curve=curve.out_expo)
        invoke(self.start_fall, delay=self.fall_after)


    def start_fall(self):
        self.y_animator.pause()
        self.jumping = False

    def land(self):
        # print('land')
        self.air_time = 0
        self.grounded = True


    def on_enable(self):
        mouse.locked = True
        self.cursor.enabled = True

    def reset(self):
        self.cd.reset()


    def on_disable(self):
        mouse.locked = False
        self.cursor.enabled = False


def add_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument( "-t" , "--type" , type=str, default= "DICT_7X7_1000" ,
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



def initcamera():
    arucoDict,arucoParams = setup()
    print( "[INFO] starting video stream..." )
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    cd = CameraDetector(cap , arucoDict, arucoParams)
    return cd


if __name__ == '__main__':
    cd = initcamera()
    window.vsync = False
    app = Ursina()
    # Sky(color=color.gray)
    ground = Entity(model='plane', scale=(100,1,100), color=color.yellow.tint(-.2), texture='white_cube', texture_scale=(100,100), collider='box')
    e = Entity(model='cube', scale=(1,5,10), x=3, y=.01,z=5, rotation_y=90, collider='box', texture='white_cube')
    e.texture_scale = (e.scale_z, e.scale_y)
    e = Entity(model='cube', scale=(1,5,10), x=-2, y=.01, collider='box', texture='white_cube')
    e.texture_scale = (e.scale_z, e.scale_y)
    e = Entity(model='cube', scale=(1,5,10), x=3, y=.01,z=-5, collider='box', texture='white_cube')
    e.texture_scale = (e.scale_z, e.scale_y)
    e = Entity(model='cube', scale=(1,5,10), x=8, y=.01, collider='box', texture='white_cube')
    e.texture_scale = (e.scale_z, e.scale_y)

    player = FirstPersonControllera(cd=cd, y=2, origin_y=-.5)
    player.gun = None

    print(player.cd)

    def input(key):
        if key == 'q':
            quit()
        elif key=='r':
            player.reset()


    # player.add_script(NoclipMode())
    app.run()